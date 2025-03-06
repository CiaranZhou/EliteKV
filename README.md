# EliteKV
【paper】[EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection](https://arxiv.org/abs/2503.01586)

🚧 *Work in Progress* 🚧
This repository is under active development. Feedback are welcome! More updates coming soon—stay tuned! 

---

### Dataset
[RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)

### Model
[LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)

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
**RoPElite**
```bash
bash RoPElite/cal_then_rank.sh
```

**Dimension Allocation**
```bash
python dimension_allocation/allocation_ppl.py \
    --model_path path/to/your/model \
    --data_path path/to/your/data \
    --file_path RoPElite/rank/RoPElite_1.pkl \
    --start 1 \
    --end 32 \
    --eval_iters 32
```
You can visualize the result using *dimension_allocation/draw_fig.ipynb*

**Model Conversion**
```bash
python convert/convert.py \
    --model_path path/to/model \
    --pe_mode EliteKV \
    --half_of_rope_dim 12 \
    --kv_dim 2048 \
    --save_dir convert/model
```

## Acknowledgements

Some code in this project is cited and modified from [transformers](https://github.com/huggingface/transformers).
We train model using [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), an easy and efficient LLM training framework.
we evaluate our method by using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Citation
```bibtex
@misc{zhou2025elitekvscalablekvcache,
      title={EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection}, 
      author={Yuhao Zhou and Sirui Song and Boyang Liu and Zhiheng Xi and Senjie Jin and Xiaoran Fan and Zhihao Zhang and Wei Li and Xuanjing Huang},
      year={2025},
      eprint={2503.01586},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01586}, 
}
```

