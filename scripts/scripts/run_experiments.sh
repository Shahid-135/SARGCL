#!/usr/bin/env bash
# Reproduce the SARGCL v5 experiments. Run from the repository root.
# Each training run uses early stopping (max 50 epochs, patience 10).
# Choose GPUs with nvidia-smi; CUDA_DEVICE_ORDER=PCI_BUS_ID keeps GPU ids
# consistent with nvidia-smi.
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ---- 1. Main model (B = 8). The first run also builds the caption feature cache.
CUDA_VISIBLE_DEVICES=0 python3 SARGCL_v5.py > v5_bs8.out 2>&1

# ---- 2. Batch-size study (Table IX)
for B in 4 16 32; do
  CUDA_VISIBLE_DEVICES=0 SARGCL_BATCH_SIZE=$B python3 SARGCL_v5.py > v5_bs$B.out 2>&1
done

# ---- 3. Without difficulty-regulated alignment (fixed temperature)
CUDA_VISIBLE_DEVICES=0 SARGCL_FIXED_TAU=1 python3 SARGCL_v5.py > v5_fixedtau.out 2>&1

# ---- 4. Noisy correspondence (Table VI)
for R in 0.4 0.6; do
  CUDA_VISIBLE_DEVICES=0 SARGCL_NOISE_RATIO=$R python3 SARGCL_v5.py > v5_noise$R.out 2>&1
  CUDA_VISIBLE_DEVICES=0 SARGCL_NOISE_RATIO=$R SARGCL_FIXED_TAU=1 python3 SARGCL_v5.py > v5_noise${R}f.out 2>&1
done

# ---- 5. Diagnostic heads
CUDA_VISIBLE_DEVICES=0 SARGCL_HEAD=global_only     python3 SARGCL_v5.py > v5_global_only.out 2>&1
CUDA_VISIBLE_DEVICES=0 SARGCL_HEAD=structured_only python3 SARGCL_v5.py > v5_structured_only.out 2>&1

# ---- 6. Evaluation (test metrics, difficulty bins, tau tables)
CUDA_VISIBLE_DEVICES=0 python3 SARGCL_v5_EVAL.py \
  sargcl_results_v5 sargcl_results_v5_fixedtau:fixed \
  sargcl_results_v5_bs4 sargcl_results_v5_bs16 sargcl_results_v5_bs32 \
  sargcl_results_v5_global_only sargcl_results_v5_structured_only
CUDA_VISIBLE_DEVICES=0 python3 SARGCL_v5_EVAL.py \
  sargcl_results_v5_noise40 sargcl_results_v5_fixedtau_noise40:fixed \
  sargcl_results_v5_noise60 sargcl_results_v5_fixedtau_noise60:fixed

# ---- 7. Diagnostics
CUDA_VISIBLE_DEVICES=0 python3 SARGCL_probe.py sargcl_results_v5
python3 SARGCL_complementarity.py
CUDA_VISIBLE_DEVICES=0 python3 SARGCL_NMAX_SCALABILITY_v5.py
python3 make_fig4_v5.py
