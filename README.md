# EliteKV
EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection

### Dataset
[RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)

### Model
[LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)

[LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)

## Start

### Installation
```bash
conda create --name EliteKV python=3.10 -y
conda activate EliteKV
git clone --depth 1 https://github.com/CiaranZhou/EliteKV.git
cd EliteKV
pip install -r requirements.txt
```

### Quickstart
RoPElite
```bash
bash RoPElite/cal_then_rank.sh
```

model conversion
```bash
python convert/convert.py \
    --model_path path/to/model \
    --pe_mode EliteKV \
    --half_of_rope_dim 4 \
    --kv_dim 960 \
    --save_dir convert/model
```

