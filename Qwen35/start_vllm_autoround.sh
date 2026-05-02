#!/bin/bash
# AutoRound INT4 Quantization - For Limited VRAM (<48GB Available)
#
# ⚠️ TRADEOFFS:
# - ✅ Stable tool calling WITH qwen3.5-enhanced.jinja template
# - ✅ Saves ~4GB VRAM vs FP8 (good for GUI/other tasks)
# - ⚠️ Higher perplexity than BF16/FP8 (some accuracy loss)
# - ⚠️ INT4 quantization not lossless (vs FP8 near-lossless)
#
# Use this if: You need to save VRAM and accept accuracy tradeoffs
# For best quality: ./start_vllm_FP8.sh (official Qwen3.5-27B-FP8)

# Environment Variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=4

# NCCL tuning for mixed GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

export MODEL_NAME="Intel/Qwen3.5-27B-int4-AutoRound"

# Clean cache
rm -rf ~/.cache/flashinfer

# Activate virtual environment
source /home/cychan/vLLM/.venv/bin/activate

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
# Start vLLM with reduced swap space
vllm serve $MODEL_NAME \
  --served-model-name Qwen3.5-27B \
  --chat-template qwen3.5-enhanced.jinja \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 219520 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --no-use-tqdm-on-load \
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only 

#  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
# current hardware setting is not allowed to have 80BA3B model as speculator

  # --attention-backend FLASHINFER \
