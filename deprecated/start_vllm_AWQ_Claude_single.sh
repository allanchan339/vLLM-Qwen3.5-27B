#!/bin/bash
# ⚠️ DEPRECATED: Uses Qwopus3.5 series (SFT-distilled)
#
# WARNING: This model has format drift issues after 65K tokens
# See README.md "Qwopus3.5 Series is a TRAP for Long Context"
#
# For production use: ./start_vllm_FP8.sh (official Qwen3.5-27B-FP8)
# --------------------------
# CUDA PATH SETTINGS
# --------------------------
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Single GPU configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Clean cache
rm -rf ~/.cache/flashinfer

# Activate virtual environment
source /home/cychan/vLLM/.venv/bin/activate

export MODEL_NAME="QuantTrio/Qwopus3.5-27B-v3-AWQ"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# Start vLLM with reduced swap space and forced eager mode
vllm serve $MODEL_NAME \
  --served-model-name vllm/Qwen3.5-27B \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.94 \
  --enforce-eager \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only
