#!/bin/bash

# --------------------------
# CUDA PATH SETTINGS
# --------------------------
# Since nvcc is in /usr/bin, we set CUDA_HOME to /usr
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# ------------------------------
# Safe, Speed-Focused Env Vars
# ------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Your original mixed-GPU safeguard
#export CUDA_VISIBLE_DEVICES=0,1        # Explicit device selection
export NCCL_CUMEM_ENABLE=0             # Fixes WSL2 NCCL issues
#export VLLM_ENABLE_CUDAGRAPH_GC=1      # Prevents VRAM leaks from CUDA graphs
#export VLLM_USE_FLASHINFER_SAMPLER=1   # Faster sampling (great for speculative decoding)

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

# Start vLLM with reduced swap space
/home/cychan/vllm/.venv/bin/vllm serve QuantTrio/Qwen3.5-27B-AWQ \
  --served-model-name vllm/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 220000 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
    --attention-backend FLASHINFER \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000
#  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
# current hardware setting is not allowed to have 80BA3B model as speculator

