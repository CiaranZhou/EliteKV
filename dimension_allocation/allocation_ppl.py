# dimension allocation
'''
python dimension_allocation/allocation_ppl.py \
    --model_path path/to/your/model \
    --data_path path/to/your/data \
    --file_path RoPElite/rank/RoPElite_1.pkl \
    --start 1 \
    --end 32 \
    --eval_iters 32
'''
import os
import math
from tqdm.auto import tqdm
import pickle
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half, repeat_kv, LlamaAttention, LlamaSdpaAttention

def load_model(model_path, attn_implementation="sdpa"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, 
            trust_remote_code=True
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=attn_implementation
        )
    except NameError:
        assert KeyError
    
    return model, tokenizer

def get_logits(model, inputs):
    model.eval()
    
    with torch.no_grad():        
        outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
        return outputs.logits

def get_avg_l2_norm_difference(output_before: torch.Tensor, output_after: torch.Tensor):
    return torch.norm(output_before - output_after).div(output_before.numel()).item()

def calculate_perplexity(model, tokenize):
    model.eval()  # 设定模型为评估模式
    
    with torch.no_grad():
        inputs = tokenize
        
        # 计算模型输出
        outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)

        # 获取log likelihood (损失)
        log_likelihood = outputs.loss.item() * inputs["input_ids"].size(1)
        
        # 累加log likelihood和token数量
        total_log_likelihood = log_likelihood
        total_token_count = inputs["input_ids"].size(1)
        
    # 计算交叉熵
    average_log_prob = total_log_likelihood / total_token_count
    perplexity = torch.exp(torch.tensor(average_log_prob))
    
    return perplexity.item()

def forward_eager(
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

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
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

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

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

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_eager_decoupling(
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

    q_nope = self.q_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    q_pe = self.q_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_pe = self.k_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_nope = self.k_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings


    cos_shape = cos.shape
    sin_shape = sin.shape
    cos = cos.view(*cos_shape[:-1], 2, -1)
    sin = sin.view(*sin_shape[:-1], 2, -1)
    rope_mask_cols = self.importance_rank[:, :self.half_of_rope_dim]
    rope_mask_rows = torch.arange(self.num_heads).unsqueeze(1).expand(-1, self.half_of_rope_dim)
    cos = cos.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    sin = sin.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    nope_mask = torch.ones_like(cos, dtype=torch.bool)
    nope_mask[:, rope_mask_rows, :, :, rope_mask_cols] = False
    cos_nope = cos[nope_mask].view(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], 2, -1)
    # cos_nope = torch.ones(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], 2, self.half_of_nope_dim, device=cos.device, dtype=cos.dtype)
    cos_rope = cos[~nope_mask].view(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], 2, -1)
    cos = torch.cat([cos_nope, cos_rope], dim=-1).view(*cos_shape[:1], self.num_heads, *cos_shape[1:])
    sin_nope = sin[nope_mask].view(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], 2, -1)
    # sin_nope = torch.zeros(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], 2, self.half_of_nope_dim, device=sin.device, dtype=sin.dtype)
    sin_rope = sin[~nope_mask].view(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], 2, -1)
    sin = torch.cat([sin_nope, sin_rope], dim=-1).view(*sin_shape[:1], self.num_heads, *sin_shape[1:])

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, 2, self.half_of_head_dim)
    query_states[:, :, :, :, :self.half_of_nope_dim] = q_nope.view(*q_nope.shape[:-1], 2, -1)
    query_states[:, :, :, :, self.half_of_nope_dim:] = q_pe.view(*q_pe.shape[:-1], 2, -1)
    query_states = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim)

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, 2, self.half_of_head_dim)
    key_states[:, :, :, :, :self.half_of_nope_dim] = k_nope.view(*k_nope.shape[:-1], 2, -1)
    key_states[:, :, :, :, self.half_of_nope_dim:] = k_pe.view(*k_pe.shape[:-1], 2, -1)
    key_states = key_states.reshape(bsz, self.num_heads, q_len, self.head_dim)

    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)

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

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_eager_decoupling_svd(
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

    q_nope = self.q_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    q_pe = self.q_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_pe = self.k_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_proj(hidden_states))
        .view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2 + self.head_dim)
        .transpose(1, 2)
    )
    k_nope, value_states = kv.split([self.half_of_nope_dim * 2, self.head_dim], dim=-1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings


    cos_shape = cos.shape
    sin_shape = sin.shape
    cos = cos.view(*cos_shape[:-1], 2, -1)
    sin = sin.view(*sin_shape[:-1], 2, -1)
    rope_mask_cols = self.importance_rank[:, :self.half_of_rope_dim]
    rope_mask_rows = torch.arange(self.num_heads).unsqueeze(1).expand(-1, self.half_of_rope_dim)
    cos = cos.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    sin = sin.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)

    rope_mask = torch.zeros_like(cos, dtype=torch.bool)
    rope_mask[:, rope_mask_rows, :, :, rope_mask_cols] = True
    cos = cos[rope_mask].reshape(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], self.half_of_rope_dim * 2)
    sin = sin[rope_mask].reshape(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], self.half_of_rope_dim * 2)

    q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
    k_pe = (k_pe * cos) + (rotate_half(k_pe) * sin)

    query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    query_states[:, :, :, :self.half_of_nope_dim * 2] = q_nope
    query_states[:, :, :, self.half_of_nope_dim * 2:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    key_states[:, :, :, :self.half_of_nope_dim * 2] = k_nope
    key_states[:, :, :, self.half_of_nope_dim * 2:] = k_pe

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

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_sdpa_decoupling(
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

    q_nope = self.q_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    q_pe = self.q_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_pe = self.k_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_nope = self.k_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    cos_shape = cos.shape
    sin_shape = sin.shape
    cos = cos.view(*cos_shape[:-1], 2, -1)
    sin = sin.view(*sin_shape[:-1], 2, -1)
    rope_mask_cols = self.importance_rank[:, :self.half_of_rope_dim]
    rope_mask_rows = torch.arange(self.num_heads).unsqueeze(1).expand(-1, self.half_of_rope_dim)
    cos = cos.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    sin = sin.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    # print(f"{cos.shape=}")

    # 只算rope部分，还没跑通
    # print(f"{cos[:, rope_mask_rows, :, :, rope_mask_cols].shape=}")
    # print(f"{cos[:, rope_mask_rows, :, :, rope_mask_cols].permute(2, 0, 3, 4, 1).shape=}")
    # print(f"{torch.equal(cos[:, rope_mask_rows, :, :, rope_mask_cols].flatten(), cos[:, rope_mask_rows, :, :, rope_mask_cols].permute(2, 0, 3, 4, 1).flatten())}")
    # cos = cos[:, rope_mask_rows, :, :, rope_mask_cols].permute(2, 0, 3, 4, 1).reshape(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], self.half_of_rope_dim * 2)
    # sin = sin[:, rope_mask_rows, :, :, rope_mask_cols].permute(2, 0, 3, 4, 1).reshape(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], self.half_of_rope_dim * 2)
    # print(f"{cos.shape=}")
    # # print('q_pe:', q_pe.shape)

    # q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
    # k_pe = (k_pe * cos) + (rotate_half(k_pe) * sin)

    # query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    # query_states[:, :, :, :self.half_of_nope_dim * 2] = q_nope
    # query_states[:, :, :, self.half_of_nope_dim * 2:] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    # key_states[:, :, :, :self.half_of_nope_dim * 2] = k_nope
    # key_states[:, :, :, self.half_of_nope_dim * 2:] = k_pe

    # 通过选择矩阵提取rope部分，验证成功
    rope_mask = torch.zeros_like(cos, dtype=torch.bool)
    rope_mask[:, rope_mask_rows, :, :, rope_mask_cols] = True
    cos = cos[rope_mask].reshape(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], self.half_of_rope_dim * 2)
    sin = sin[rope_mask].reshape(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], self.half_of_rope_dim * 2)

    q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
    k_pe = (k_pe * cos) + (rotate_half(k_pe) * sin)

    query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    query_states[:, :, :, :self.half_of_nope_dim * 2] = q_nope
    query_states[:, :, :, self.half_of_nope_dim * 2:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    key_states[:, :, :, :self.half_of_nope_dim * 2] = k_nope
    key_states[:, :, :, self.half_of_nope_dim * 2:] = k_pe

    # 通过直接sin和cos置0/1来实现，验证正确
    # nope_mask = torch.one_like(cos, dtype=torch.bool)
    # nope_mask[:, rope_mask_rows, :, :, rope_mask_cols] = False
    # cos_nope = torch.ones(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], 2, self.half_of_nope_dim, device=cos.device, dtype=cos.dtype)
    # cos_rope = cos[~nope_mask].view(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], 2, -1)
    # cos = torch.cat([cos_nope, cos_rope], dim=-1).view(*cos_shape[:1], self.num_heads, *cos_shape[1:])
    # sin_nope = torch.zeros(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], 2, self.half_of_nope_dim, device=sin.device, dtype=sin.dtype)
    # sin_rope = sin[~nope_mask].view(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], 2, -1)
    # sin = torch.cat([sin_nope, sin_rope], dim=-1).view(*sin_shape[:1], self.num_heads, *sin_shape[1:])

    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, 2, self.half_of_head_dim)
    # query_states[:, :, :, :, :self.half_of_nope_dim] = q_nope.view(*q_nope.shape[:-1], 2, -1)
    # query_states[:, :, :, :, self.half_of_nope_dim:] = q_pe.view(*q_pe.shape[:-1], 2, -1)
    # query_states = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim)

    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, 2, self.half_of_head_dim)
    # key_states[:, :, :, :, :self.half_of_nope_dim] = k_nope.view(*k_nope.shape[:-1], 2, -1)
    # key_states[:, :, :, :, self.half_of_nope_dim:] = k_pe.view(*k_pe.shape[:-1], 2, -1)
    # key_states = key_states.reshape(bsz, self.num_heads, q_len, self.head_dim)

    # query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    # key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def forward_sdpa_decoupling_svd(
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

    q_nope = self.q_nope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2).transpose(1, 2)
    q_pe = self.q_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    k_pe = self.k_rope_proj(hidden_states).view(bsz, q_len, self.num_heads, self.half_of_rope_dim * 2).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_proj(hidden_states))
        .view(bsz, q_len, self.num_heads, self.half_of_nope_dim * 2 + self.head_dim)
        .transpose(1, 2)
    )
    k_nope, value_states = kv.split([self.half_of_nope_dim * 2, self.head_dim], dim=-1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    cos_shape = cos.shape
    sin_shape = sin.shape
    cos = cos.view(*cos_shape[:-1], 2, -1)
    sin = sin.view(*sin_shape[:-1], 2, -1)
    rope_mask_cols = self.importance_rank[:, :self.half_of_rope_dim]
    rope_mask_rows = torch.arange(self.num_heads).unsqueeze(1).expand(-1, self.half_of_rope_dim)
    cos = cos.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
    sin = sin.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)

    rope_mask = torch.zeros_like(cos, dtype=torch.bool)
    rope_mask[:, rope_mask_rows, :, :, rope_mask_cols] = True
    cos = cos[rope_mask].reshape(*cos_shape[:1], self.num_heads, *cos_shape[1:-1], self.half_of_rope_dim * 2)
    sin = sin[rope_mask].reshape(*sin_shape[:1], self.num_heads, *sin_shape[1:-1], self.half_of_rope_dim * 2)

    q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
    k_pe = (k_pe * cos) + (rotate_half(k_pe) * sin)

    query_states = q_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    query_states[:, :, :, :self.half_of_nope_dim * 2] = q_nope
    query_states[:, :, :, self.half_of_nope_dim * 2:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.head_dim)
    key_states[:, :, :, :self.half_of_nope_dim * 2] = k_nope
    key_states[:, :, :, self.half_of_nope_dim * 2:] = k_pe

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def main(args):
    attn_implementation = "eager"
    decoupling = True
    svd = True
    svd &= decoupling

    rope_dim_list = [rope_dim for rope_dim in range(args.start, args.end + 1, args.step)]
    args_file_path = args.file_path

    step_size = 256
    sample_length = 2048
    model, tokenizer = load_model(args.model_path, attn_implementation=attn_implementation)
    model.eval()

    hidden_size = model.config.hidden_size

    dataset = load_dataset("parquet", data_files=args.data_path)
    samples = []
    for val_data in tqdm(dataset["train"]):
        if len(samples) >= args.eval_iters:
            break
        tokenize = tokenizer(val_data["text"], return_tensors="pt", truncation=True, max_length=sample_length).to('cuda')
        if tokenize["input_ids"].shape[-1] >= sample_length:
            samples.append(tokenize)

    perplexity, data_num = 0, 0
    with tqdm(samples) as pbar:
        for tokenize in pbar:
            perplexity += calculate_perplexity(model, tokenize)
            data_num += 1
            pbar.set_postfix({
                "avg_ppl": f"{perplexity / data_num:.6f}"
            })

    ppl = {}
    for rope_dim in rope_dim_list:
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        hidden_size = num_heads * head_dim
        half_of_head_dim = head_dim // 2
        half_of_rope_dim = rope_dim
        half_of_nope_dim = half_of_head_dim - half_of_rope_dim

        args.file_path = args_file_path
        args.file_path = f"{args.file_path.split('.')[0][:-1]}{rope_dim}.{args.file_path.split('.')[1]}"
        with open(args.file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        importance_rank = loaded_data["rank"]

        for layer_idx, model_layer in tqdm(enumerate(model.model.layers), total=len(model.model.layers)):
            self_attn = model_layer.self_attn
            self_attn.importance_rank = importance_rank[layer_idx]
            self_attn.half_of_head_dim = half_of_head_dim
            self_attn.half_of_rope_dim = half_of_rope_dim
            self_attn.half_of_nope_dim = half_of_nope_dim

            if attn_implementation == "eager":
                if svd:
                    self_attn.forward = forward_eager_decoupling_svd.__get__(self_attn, type(self_attn))
                elif decoupling:
                    self_attn.forward = forward_eager_decoupling.__get__(self_attn, type(self_attn))
                else:
                    self_attn.forward = forward_eager.__get__(self_attn, type(self_attn))
                    continue
            elif attn_implementation == "sdpa":
                if svd:
                    self_attn.forward = forward_sdpa_decoupling_svd.__get__(self_attn, type(self_attn))
                elif decoupling:
                    self_attn.forward = forward_sdpa_decoupling.__get__(self_attn, type(self_attn))
                else:
                    continue
            
            rope_mask_cols = self_attn.importance_rank[:, :half_of_rope_dim]
            rope_mask_rows = torch.arange(num_heads).unsqueeze(1).expand(-1, half_of_rope_dim)
            nope_mask = torch.ones(hidden_size, num_heads, 2, half_of_head_dim, dtype=torch.bool)
            nope_mask[:, rope_mask_rows, :, rope_mask_cols] = False

            # W_q
            W_q = self_attn.q_proj.weight.data.t()
            W_q_nope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[nope_mask].reshape(hidden_size, -1)
            self_attn.q_nope_proj = nn.Linear(*W_q_nope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_nope_proj.weight.data = W_q_nope.t().contiguous()
            W_q_rope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[~nope_mask].reshape(hidden_size, -1)
            self_attn.q_rope_proj = nn.Linear(*W_q_rope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_rope_proj.weight.data = W_q_rope.t().contiguous()

            # W_k_rope
            W_k = self_attn.k_proj.weight.data.t()
            W_k_rope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[~nope_mask].reshape(hidden_size, -1)
            self_attn.k_rope_proj = nn.Linear(*W_k_rope.shape, bias=False, device=self_attn.k_proj.weight.device, dtype=self_attn.k_proj.weight.dtype)
            self_attn.k_rope_proj.weight.data = W_k_rope.t().contiguous()

            if svd:
                # W_k_nope, W_v
                W_k_nope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[nope_mask].reshape(hidden_size, num_heads, -1)
                W_v = self_attn.v_proj.weight.data.t().view(hidden_size, num_heads, head_dim)
                W_kv = torch.cat([W_k_nope, W_v], dim=-1).reshape(hidden_size, -1)
                self_attn.kv_U, self_attn.kv_S, self_attn.kv_V = torch.linalg.svd(W_kv, full_matrices=False)
            else:
                # W_k_nope
                W_k_nope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[nope_mask].reshape(hidden_size, -1)
                self_attn.k_nope_proj = nn.Linear(*W_k_nope.shape, bias=False, device=self_attn.k_proj.weight.device, dtype=self_attn.k_proj.weight.dtype)
                self_attn.k_nope_proj.weight.data = W_k_nope.t().contiguous()

        if svd:
            min_kv_cache_ratio = 0.1
            step_size = 128
            pbar = tqdm(range(math.ceil((1 - min_kv_cache_ratio) * hidden_size / step_size)))
            thred_lora_dim = (256-rope_dim*2)*model.config.num_attention_heads*128 / (384-rope_dim*2)

            ppl[rope_dim] = {}
            for step in pbar:
                lora_size = hidden_size - step * step_size
                if lora_size > thred_lora_dim:
                    continue
                for layer_idx, model_layer in enumerate(model.model.layers):
                    self_attn = model_layer.self_attn
                    W_kv_a = self_attn.kv_U[:, :lora_size] @ torch.diag(self_attn.kv_S[:lora_size])
                    W_kv_b = self_attn.kv_V[:lora_size]
                    self_attn.kv_a_proj = nn.Linear(*W_kv_a.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
                    self_attn.kv_a_proj.weight.data = W_kv_a.t().contiguous()
                    self_attn.kv_b_proj = nn.Linear(*W_kv_b.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
                    self_attn.kv_b_proj.weight.data = W_kv_b.t().contiguous()

                perplexity, data_num = 0, 0
                for tokenize in samples:
                    perplexity += calculate_perplexity(model, tokenize)
                ppl[rope_dim][lora_size] = perplexity / args.eval_iters
                pbar.set_postfix({
                    "lora_size": lora_size,
                    "lora_ratio": f"{lora_size / hidden_size:.6f}",
                    "avg_ppl": ppl[rope_dim]
                })
        else:
            perplexity, data_num = 0, 0
            with tqdm(samples) as pbar:
                for tokenize in pbar:
                    perplexity += calculate_perplexity(model, tokenize)
                    data_num += 1
                    pbar.set_postfix({
                        "class name": self_attn.__class__.__name__,
                        "attn_implementation": model.config._attn_implementation,
                        "decoupling": decoupling,
                        "SVD": svd,
                        "rope_dim": rope_dim,
                        "avg_ppl": f"{perplexity / data_num:.6f}"
                    })
            ppl[rope_dim] = perplexity / data_num

    save_path = f"dimension_allocation/result/{args.model_path.split('/')[-1]}_{args.start}-{args.end}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(ppl, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path/to/your/model")
    parser.add_argument("--file_path", type=str, default="RoPElite/rank/Llama-2-7b-hf/RoPElite_1.pkl")
    parser.add_argument("--data_path", type=str, default="path/to/your/data")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=2)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--eval_iters", type=int, default=32)
    args = parser.parse_args()
    main(args)
