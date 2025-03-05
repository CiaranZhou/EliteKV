# rope_dim
import pickle
import torch
import argparse
import os

rank_dict = {}
def main(args):
    with open(args.file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    for layer_idx in range(len(loaded_data)):
        rank_dict[layer_idx] = torch.argsort(loaded_data[layer_idx].T, dim=1)
    num_fixed_dim = int(args.file_path.split('.')[-2].split('_')[-1]) + 1
    result = {}
    result["note"] = f"Calculate importance rank based on {args.file_path}, save top-{num_fixed_dim} chunks"
    result["rank"] = rank_dict

    
    save_dir = os.path.join(args.save_dir, args.file_path.split('/')[-2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{args.file_path.split('/')[-1].split('_')[0]}_{num_fixed_dim}.pkl"), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="./result/Llama-2-7b-hf/RoPElite_3.pkl")
    parser.add_argument("--save_dir", type=str, default="./rank")
    args = parser.parse_args()
    main(args)