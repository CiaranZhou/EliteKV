"""
python RoPElite/cal_attn_distance.py
    --model_path path/to/your/model
    --task RoPElite (Contribution RoPElite)
    --data_path path/to/your/data
    --data_length 2048
    --eval_iters 2000
    --fixed_dim_num 0
    --rank_file "RoPElite/rank/RoPElite_0.pkl"
    --save_dir "RoPElite/result"
"""
import argparse
import torch
from tqdm.auto import tqdm
import pickle
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM

import analysis

device = torch.device("cuda:0")

FILE_TYPE = {
    'json': 'json',
    'jsonl': 'json',
    'parquet': 'parquet'
}

def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False, 
            trust_remote_code=True
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": f"cuda:0"},
            attn_implementation="eager",
        )
    except NameError:
        assert KeyError
    return model, tokenizer

def main(args):
    # Load model & init result.global
    model, tokenizer = load_model(args.model_path)
    analysis.init_task(model, args)

    # load data
    print("dataset loading")
    dataset = load_dataset(FILE_TYPE[args.data_path.split('.')[-1]], data_files = args.data_path)
    filter_count = {'value': 0}
    def filter_with_limit(example):
        if filter_count['value'] >= args.eval_iters * analysis.task.bsz: 
            return False
        if tokenizer(example["text"], return_tensors="pt", truncation=False)["input_ids"].shape[-1] >= args.data_length:
            filter_count['value'] += 1
            return True
        return False
    dataset = dataset.filter(filter_with_limit)

    # forward...
    pbar = tqdm(dataset["train"], total=args.eval_iters)
    for k, val_data in enumerate(pbar):
        if analysis.task.iters >= args.eval_iters:
            break
        tokenize = tokenizer(val_data["text"], return_tensors="pt", truncation=True, max_length=args.data_length)
        if tokenize["input_ids"].shape[-1] < args.data_length:
            continue
        analysis.task.iters += 1
        analysis.task.token_sum += tokenize["input_ids"].shape[-1]
        pbar.set_postfix(total_iter=args.eval_iters, iter=analysis.task.iters)

        model.eval()
        with torch.no_grad():
            output = model(
                input_ids = tokenize["input_ids"].to(device)
            )

    analysis.task.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path/to/your/model")
    parser.add_argument("--task", type=str, default="RoPElite")
    parser.add_argument("--data_path", type=str, default="path/to/your/data")
    parser.add_argument("--data_length", type=int, default=2048)
    parser.add_argument("--eval_iters", type=int, default=2000)
    parser.add_argument("--fixed_dim_num", type=int, default=0)
    parser.add_argument("--rank_file", type=str, default="RoPElite/rank/RoPElite_0.pkl")
    parser.add_argument("--save_dir", type=str, default="RoPElite/result")
    args = parser.parse_args()
    main(args)