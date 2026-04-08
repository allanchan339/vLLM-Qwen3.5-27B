#!/bin/bash
# THIS IS THE ONLY CONFIG THAT IS WORKING FOR NOW.
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
export VLLM_ENABLE_CUDAGRAPH_GC=1      # Prevents VRAM leaks from CUDA graphs

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
source /home/cychan/vLLM/.venv/bin/activate

# Enable memory profiler to estimate CUDA graphs v0.19 functionality
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export MODEL_NAME="QuantTrio/Qwopus3.5-27B-v3-AWQ"
export VLLM_USE_FLASHINFER_SAMPLER=1   # Faster sampling (great for speculative decoding)
# Start vLLM with reduced swap space
vllm serve $MODEL_NAME \
  --served-model-name vllm/Qwen3.5-27B \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --pipeline-parallel-size 2 \
  --max-model-len 219520 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 12288 \
  --max-num-seqs 6 \
  --kv-cache-dtype fp8 \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only 

