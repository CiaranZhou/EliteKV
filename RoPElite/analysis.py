import torch
from dataclasses import dataclass
import pickle
import os
import torch.multiprocessing as mp

#rewrite
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import torch.nn.functional as F
from torch import nn
from transformers.utils import logging
from transformers.models.llama.modeling_llama_init import repeat_kv, apply_rotary_pos_emb
import numpy as np
logger = logging.get_logger(__name__)

task = None

def init_task(model, args):
    global task
    task = TASK_CLASSES[args.task](model, args)

def rewrite_to_get_attn_score_RopeDimPreserveGreedy(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states_origin = query_states.clone()
        key_states_origin = key_states.clone()

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # search next fixed dim
        query_states_new = query_states_origin.clone()
        key_states_new = key_states_origin.clone()
        half_head_dim = query_states.shape[-1] // 2
        # change qk_new for fixed_dim of every head (new<-rope) 
        dim_idx = task.last_round_rank[self.layer_idx][:, :task.fixed_dim_num]
        dim_idx = torch.cat((dim_idx, dim_idx+half_head_dim), dim=1)
        head_idx = torch.arange(self.num_heads).unsqueeze(1).expand(-1, task.fixed_dim_num*2)
        query_states_new[:, head_idx, :, dim_idx] = query_states[:, head_idx, :, dim_idx]
        key_states_new[:, head_idx, :, dim_idx] = key_states[:, head_idx, :, dim_idx]

        # loop unfixed dim
        for rank_idx in range(task.fixed_dim_num, half_head_dim):
            dim_idx = task.last_round_rank[self.layer_idx][:, rank_idx]
            dim_idx = torch.stack((dim_idx, dim_idx+half_head_dim), dim=0)
            head_idx = torch.arange(self.num_heads).expand(2,-1)
            # change qk_new for dim of every head of current rank (new<-rope) 
            query_states_new[:, head_idx, :, dim_idx] = query_states[:, head_idx, :, dim_idx]
            key_states_new[:, head_idx, :, dim_idx] = key_states[:, head_idx, :, dim_idx]

            # calc attn_score
            attn_weights_new = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights_new = attn_weights_new + causal_mask
            attn_weights_distance = (attn_weights_new - attn_weights)
            attn_weights_distance = attn_weights_distance.reshape(attn_weights_distance.shape[1], -1).abs().mean(dim=1)

            task.update(self.layer_idx, rank_idx, attn_weights_distance)

            # recover qk_new for dim of every head of current rank (new<-origin) 
            query_states_new[:, head_idx, :, dim_idx] = query_states_origin[:, head_idx, :, dim_idx]
            key_states_new[:, head_idx, :, dim_idx] = key_states_origin[:, head_idx, :, dim_idx]

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def rewrite_to_get_attn_score_RopeDimContribution(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        

        # sample dim
        pairs_num = 100 * task.data_length
        q = query_states[:,:,task.q_idx,:].view(bsz, self.num_heads, pairs_num, 2, self.head_dim // 2).permute(0, 1, 4, 2, 3).contiguous()
        k = key_states[:,:,task.k_idx,:].view(bsz, self.num_heads, pairs_num, 2, self.head_dim // 2).permute(0, 1, 4, 2, 3).contiguous()
        g = q * k
        dim_contribution = g.sum(dim=(3, 4)).squeeze().transpose(0,1)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        task.update(self.layer_idx, dim_contribution)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


@dataclass
class ModelDatasetArgs(dict):
    model_name: str = ""
    model_layer: int = 32
    intermediate_size: int = 11008
    data_length: int = 32
    bsz: int = 1
    iters: int = 0
    token_sum: int = 0

# find the top-k important pairs of rope dim using greedy method, which preserves a pair of rope dim and compares the change of attention score
# save attn_score_distance
class RopeDimPreserveGreedy(ModelDatasetArgs):
    def __init__(self, model, args) -> None:
        super().__init__(f"{args.model_path.split('/')[-1]}", model.config.num_hidden_layers, model.config.intermediate_size, args.data_length)
        self.task_name = "RoPElite"
        self.head_num = model.config.num_attention_heads
        self.half_head_dim = (model.config.hidden_size / self.head_num) // 2
        self.device = next(model.parameters()).device
        self.fixed_dim_num = args.fixed_dim_num
        def get_last_round_rank(rank_file):
            with open(rank_file, 'rb') as file:
                data = pickle.load(file)
            new_data = {}
            for layer_idx, heads_rank in data["rank"].items():
                new_data[layer_idx] = heads_rank.to(self.device)
            return new_data
        self.last_round_rank = get_last_round_rank(args.rank_file)
        self.attn_weights_distance = {layer_idx: torch.zeros(int(self.half_head_dim), self.head_num, device=self.device) for layer_idx in range(self.model_layer)}
        self.save_dir = args.save_dir
        model.config.attn_implementation = "eager"
        for layer in model.model.layers:
            layer.self_attn.forward = rewrite_to_get_attn_score_RopeDimPreserveGreedy.__get__(layer.self_attn, type(layer.self_attn))

    def update(self, layer_idx, rank_idx, attn_weights_distance):
        dim_idx = self.last_round_rank[layer_idx][:,rank_idx]
        self.attn_weights_distance[layer_idx][dim_idx, torch.arange(self.head_num)] += attn_weights_distance

    def save(self):
        save_result = {}
        # process the result
        for layer_idx in range(self.model_layer):
            self.attn_weights_distance[layer_idx] /= self.iters
            save_result[layer_idx] = self.attn_weights_distance[layer_idx]

        save_dir = os.path.join(f"{self.save_dir}", f"{self.model_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{self.task_name}_{self.fixed_dim_num}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(save_result, f)
        print(f"has saved to {save_path}")
        

# find the top-k important pairs of rope dim using method proposed by On the token distance modeling ability of higher RoPE attention dimension(arXiv:2410.08703v2)
# save dimension contribution
class RopeDimContribution(ModelDatasetArgs):
    def __init__(self, model, args) -> None:
        super().__init__(f"{args.model_path.split('/')[-1]}", model.config.num_hidden_layers, model.config.intermediate_size, args.data_length)
        self.task_name = "Contribution"
        self.head_num = model.config.num_attention_heads
        self.half_head_dim = (model.config.hidden_size / self.head_num) // 2
        self.device = next(model.parameters()).device
        def generate_qk(max_value):
            num_pairs = 100 * max_value
            x_values = np.random.randint(0, max_value, size=num_pairs)
            y_values = np.random.randint(0, max_value, size=num_pairs)
            for i in range(num_pairs):
                if y_values[i] < x_values[i]:
                    y_values[i] = np.random.randint(x_values[i], max_value)
            return x_values, y_values
        self.q_idx, self.k_idx = generate_qk(self.data_length)
        self.dim_contribution = {layer_idx: torch.zeros(int(self.half_head_dim), self.head_num, device=self.device) for layer_idx in range(self.model_layer)}
        self.save_dir = args.save_dir
        model.config.attn_implementation = "eager"
        for layer in model.model.layers:
            layer.self_attn.forward = rewrite_to_get_attn_score_RopeDimContribution.__get__(layer.self_attn, type(layer.self_attn))

    def update(self, layer_idx, dim_contribution):
        self.dim_contribution[layer_idx] += dim_contribution

    def save(self):
        save_result = {}
        # process the result
        for layer_idx in range(self.model_layer):
            self.dim_contribution[layer_idx] /= self.iters
            save_result[layer_idx] = self.dim_contribution[layer_idx]

        save_dir = os.path.join(f"{self.save_dir}", f"{self.model_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, f"{self.task_name}.pkl"), 'wb') as f:
            pickle.dump(save_result, f)


TASK_CLASSES = {
    "Contribution": RopeDimContribution,
    "RoPElite": RopeDimPreserveGreedy
}