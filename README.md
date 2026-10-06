# HW1 Q4: nanoGPT architecture ablations (char-level Shakespeare)

Fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The original nanoGPT README is kept in `README_nanogpt.md`.

## What changed
All variants are flags on `GPTConfig` in `model.py` (also exposed in `train.py`):

| flag | values | question |
|---|---|---|
| `norm_type` | `layernorm` (default) / `rmsnorm` | Q4.2 |
| `mlp_type` | `gelu` (default) / `swiglu` (hidden = 8d/3, same param count) | Q4.3 |
| `pos_enc` | `learned` (default) / `nope` / `rope` | Q4.4 |
| `n_kv_head` | `0` = MHA (default) / `3` = GQA, group size 2 | Q4.5 |

One config per run in `config/hw1_*.py`. Each one is `train_shakespeare_char.py` plus exactly one change.
`*_lr2e-3.py` are the same runs at learning rate 2e-3 (min_lr 2e-4).
Seed (1337) and iteration count (5000) are the same for every run.

## Reproduce
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build (what I used: torch 2.11.0+cu128) (Windows/Linux + NVIDIA)
pip install numpy tiktoken matplotlib typst
python data/shakespeare_char/prepare.py
python run_all.py --device=cuda          # all 12 runs, logs -> logs/<run>.log
python hw1/plot_losses.py                # plots -> plots/, table -> results/summary.md
python hw1/q2_rope_plot.py               # Q2 plot
```

## Where things are
- `logs/`: raw training logs (train/val loss every 250 iters)
- `plots/`: loss curves for every question, plus the Q2 RoPE plot
- `results/summary.md`: final/best val loss per run
- `hw1/report.pdf`: the written solutions (compiled from `hw1/report.typ`)
