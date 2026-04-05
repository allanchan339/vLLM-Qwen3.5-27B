#!/bin/bash

# --------------------------
# CUDA PATH SETTINGS
# --------------------------
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/.local/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
# ------------------------------
# Safe, Speed-Focused Env Vars
# ------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=0

export OMP_NUM_THREADS=4

# NCCL tuning for SYS/PCIe topology (DO NOT REMOVE)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

# --------------------------
# FIX: Clean Stale FlashInfer Cache
# --------------------------
rm -rf ~/.cache/flashinfer

# Activate virtual environment
source /home/cychan/vllm/.venv/bin/activate

# Start ParoQuant (vLLM backend) with reduced swap space
# python -m paroquant.cli.serve --model z-lab/Qwen3.5-27B-PARO --port 8000

python -m paroquant.cli.serve --model z-lab/Qwen3.5-27B-PARO --port 8000 \
  --served-model-name vllm/Qwen3.5-27B \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0
#  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
# current hardware setting is not allowed to have 80BA3B model as speculator
  # --attention-backend FLASHINFER \ 
  # --enable-auto-tool-choice \
  # --enable-chunked-prefill \
  # --enable-prefix-caching \
  # --kv-cache-dtype fp8 \
  # --tool-call-parser qwen3_coder \
  # --reasoning-parser qwen3 \
  # --attention-backend FLASH_ATTN \ 
  # --tensor-parallel-size 2 \
