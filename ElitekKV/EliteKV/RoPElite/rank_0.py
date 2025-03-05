import argparse
import torch
from transformers import AutoConfig
from tqdm.auto import tqdm
import pickle
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

def main(args):
    config = AutoConfig.from_pretrained(args.model_path)

    result = {}
    result["note"] = f"Initial rank, save top-0 chunks"
    half_head_dim = (config.hidden_size // config.num_attention_heads) // 2
    rank_dict = {layer:torch.zeros(config.num_attention_heads, half_head_dim, dtype=int, device=torch.device('cuda')) for layer in range(config.num_hidden_layers)}
    result["rank"] = rank_dict
    save_dir = os.path.join(f"{args.save_dir}", f"{args.model_path.split('/')[-1]}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"RoPElite_0.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/ssr/model/Llama-2-7b-hf")
    parser.add_argument("--save_dir", type=str, default="RoPElite/rank")
    args = parser.parse_args()
    main(args)