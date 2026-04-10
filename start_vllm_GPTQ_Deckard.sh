#!/bin/bash
# ⚠️ DEPRECATED: 40B model - too large for 48GB VRAM setup
#
# This script is kept for reference only. Model requires >48GB VRAM.
# For production use: ./start_vllm_FP8.sh (Qwen3.5-27B-FP8)
# --------------------------
# CUDA PATH SETTINGS
# --------------------------
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Environment Variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export OMP_NUM_THREADS=4

# NCCL tuning for mixed GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

# Clean cache
rm -rf ~/.cache/flashinfer

# Activate virtual environment
source /home/cychan/vLLM/.venv/bin/activate

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export MODEL_NAME="raydelossantos/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-GPTQ-Int4"
export VLLM_USE_FLASHINFER_SAMPLER=1
# Start vLLM with reduced swap space
vllm serve $MODEL_NAME \
  --served-model-name vllm/Qwen3.5-27B \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --pipeline-parallel-size 2 \
  --max-model-len 90000 \
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

