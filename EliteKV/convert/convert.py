import os
from tqdm import tqdm

import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer
import pickle
import argparse

device = "cuda:0"


class FUSE:
    @classmethod
    def linspace_concat(cls, model, half_of_head_dim, half_of_rope_dim, kv_dim):
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        for idx, layer in tqdm(enumerate(model.model.layers)):
            self_attn = layer.self_attn
            
            # new config
            self_attn.half_of_head_dim = half_of_head_dim
            self_attn.half_of_rope_dim = half_of_rope_dim
            self_attn.half_of_nope_dim = half_of_head_dim - half_of_rope_dim
            self_attn.kv_dim = kv_dim

            nope_mask = torch.ones(half_of_head_dim, dtype=torch.bool)
            nope_mask[torch.linspace(0, half_of_head_dim - 1, half_of_rope_dim, dtype=torch.int)] = False

            # q_nope_proj q_rope_projv
            W_q = self_attn.q_proj.weight.data.t()
            W_q_nope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[..., nope_mask].reshape(hidden_size, -1)
            self_attn.q_nope_proj = nn.Linear(*W_q_nope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_nope_proj.weight.data = W_q_nope.t().contiguous()
            W_q_rope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[..., ~nope_mask].reshape(hidden_size, -1)
            self_attn.q_rope_proj = nn.Linear(*W_q_rope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_rope_proj.weight.data = W_q_rope.t().contiguous()
            
            # k_rope_proj
            W_k = self_attn.k_proj.weight.data.t()
            W_k_rope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[..., ~nope_mask].reshape(hidden_size, -1)
            self_attn.k_rope_proj = nn.Linear(*W_k_rope.shape, bias=False, device=self_attn.k_proj.weight.device, dtype=self_attn.k_proj.weight.dtype)
            self_attn.k_rope_proj.weight.data = W_k_rope.t().contiguous()

            # kv_a_proj kv_b_proj
            W_k_nope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[..., nope_mask].reshape(hidden_size, num_heads, -1)
            W_v = self_attn.v_proj.weight.data.t().view(hidden_size, num_heads, head_dim)
            W_kv = torch.cat([W_k_nope, W_v], dim=-1).reshape(hidden_size, -1)
            kv_U, kv_S, kv_V = torch.linalg.svd(W_kv, full_matrices=False)
            kv_U = kv_U[:, :kv_dim]
            kv_S = torch.diag(kv_S[:kv_dim])
            kv_V = kv_V[:kv_dim]
            W_kv_b = kv_S @ kv_V
            self_attn.kv_a_proj = nn.Linear(*kv_U.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
            self_attn.kv_a_proj.weight.data = kv_U.t().contiguous()
            self_attn.kv_b_proj = nn.Linear(*W_kv_b.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
            self_attn.kv_b_proj.weight.data = W_kv_b.t().contiguous()

            del self_attn.q_proj, self_attn.k_proj, self_attn.v_proj

    @classmethod
    def rank_lh_concat(cls, model, half_of_head_dim, half_of_rope_dim, kv_dim):
        dim_path = os.path.join('RoPElite/rank', args.model_path.split('/')[-1], f"RoPElite_{half_of_rope_dim}.pkl")
        with open(dim_path, 'rb') as file:
            loaded_data = pickle.load(file)
        importance_rank = loaded_data["rank"]

        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = hidden_size // num_heads

        for idx, layer in tqdm(enumerate(model.model.layers)):
            self_attn = layer.self_attn
            
            # new config
            self_attn.half_of_head_dim = half_of_head_dim
            self_attn.half_of_rope_dim = half_of_rope_dim
            self_attn.half_of_nope_dim = half_of_head_dim - half_of_rope_dim
            self_attn.kv_dim = kv_dim
            self_attn.importance_rank = importance_rank[idx]

            rope_mask_cols = self_attn.importance_rank[:, :half_of_rope_dim]
            rope_mask_rows = torch.arange(num_heads).unsqueeze(1).expand(-1, half_of_rope_dim)
            nope_mask = torch.ones(hidden_size, num_heads, 2, half_of_head_dim, dtype=torch.bool)
            nope_mask[:, rope_mask_rows, :, rope_mask_cols] = False

            # q_nope_proj q_rope_proj
            W_q = self_attn.q_proj.weight.data.t()
            W_q_nope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[nope_mask].reshape(hidden_size, -1)
            self_attn.q_nope_proj = nn.Linear(*W_q_nope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_nope_proj.weight.data = W_q_nope.t().contiguous()
            W_q_rope = W_q.view(hidden_size, num_heads, 2, half_of_head_dim)[~nope_mask].reshape(hidden_size, -1)
            self_attn.q_rope_proj = nn.Linear(*W_q_rope.shape, bias=False, device=self_attn.q_proj.weight.device, dtype=self_attn.q_proj.weight.dtype)
            self_attn.q_rope_proj.weight.data = W_q_rope.t().contiguous()
            
            # k_rope_proj
            W_k = self_attn.k_proj.weight.data.t()
            W_k_rope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[~nope_mask].reshape(hidden_size, -1)
            self_attn.k_rope_proj = nn.Linear(*W_k_rope.shape, bias=False, device=self_attn.k_proj.weight.device, dtype=self_attn.k_proj.weight.dtype)
            self_attn.k_rope_proj.weight.data = W_k_rope.t().contiguous()

            # kv_a_proj kv_b_proj
            W_k_nope = W_k.view(hidden_size, num_heads, 2, half_of_head_dim)[nope_mask].reshape(hidden_size, num_heads, -1)
            W_v = self_attn.v_proj.weight.data.t().view(hidden_size, num_heads, head_dim)
            W_kv = torch.cat([W_k_nope, W_v], dim=-1).reshape(hidden_size, -1)
            kv_U, kv_S, kv_V = torch.linalg.svd(W_kv, full_matrices=False)
            kv_U = kv_U[:, :kv_dim]
            kv_S = torch.diag(kv_S[:kv_dim])
            kv_V = kv_V[:kv_dim]
            W_kv_b = kv_S @ kv_V
            self_attn.kv_a_proj = nn.Linear(*kv_U.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
            self_attn.kv_a_proj.weight.data = kv_U.t().contiguous()
            self_attn.kv_b_proj = nn.Linear(*W_kv_b.shape, bias=False, device=self_attn.v_proj.weight.device, dtype=self_attn.v_proj.weight.dtype)
            self_attn.kv_b_proj.weight.data = W_kv_b.t().contiguous()

            del self_attn.q_proj, self_attn.k_proj, self_attn.v_proj
    
FUSE_METHOD = {
    "Uniform": FUSE.linspace_concat,
    "Contribution": FUSE.rank_lh_concat,
    "EliteKV": FUSE.rank_lh_concat,
}

def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, device_map=device)
    model = LlamaForCausalLM.from_pretrained(args.model_path, device_map=device)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    half_of_head_dim = head_dim // 2
    
    FUSE_METHOD[f"{args.pe_mode}"](model, half_of_head_dim, args.half_of_rope_dim, args.kv_dim)
    model.config.half_of_rope_dim = args.half_of_rope_dim
    model.config.kv_dim = args.kv_dim
    model.config.auto_map = {
        "AutoModel": f"{args.modeling_llama}.LlamaModel",
        "AutoModelForCausalLM": f"{args.modeling_llama}.LlamaForCausalLM"
    }

    save_path = f"{args.save_dir}/{args.model_path.split('/')[-1]}-{args.half_of_rope_dim}-{args.kv_dim}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    modelling_llama = f"modeling_llama_{args.pe_mode}"
    os.symlink(f"convert/{modelling_llama}.py", os.path.join(save_path, f"{modelling_llama}.py"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/ssr/model/Llama-2-7b-hf")
    parser.add_argument("--pe_mode", type=str, default="EliteKV")
    parser.add_argument("--half_of_rope_dim", type=int, default=4)
    parser.add_argument("--kv_dim", type=int, default=960)
    parser.add_argument("--save_dir", type=str, default="convert/model")
    args = parser.parse_args()
    main(args)