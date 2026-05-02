#!/bin/bash
# SUCCESS: Proven stable configuration for Qwen3.5-27B on mixed GPU setup
# 
# Verified: 138.2k tokens over 1h 9m continuous agentic work
# Project: https://github.com/allanchan339/qwen_own_project (Knowledge Platform)
#
# Key fixes:
# - NCCL tuning for mixed GPU (RTX 3090 SM80 + RTX 4090 SM89)
# - Custom Jinja template for stable tool calling with reasoning
# - FP8 KV cache for 219k context length
#
# Requires: qwen3.5-enhanced.jinja template in working directory 

# ------------------------------
# Safe, Speed-Focused Env Vars
# ------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Your original mixed-GPU safeguard
export CUDA_VISIBLE_DEVICES=0,1        # Explicit device selection
export NCCL_CUMEM_ENABLE=0             # Fixes WSL2 NCCL issues
export VLLM_ENABLE_CUDAGRAPH_GC=1      # Prevents VRAM leaks from CUDA graphs
export VLLM_USE_FLASHINFER_SAMPLER=1   # Faster sampling (great for speculative decoding)

export OMP_NUM_THREADS=4

# NCCL tuning for SYS/PCIe topology (DO NOT REMOVE)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export MODEL_NAME="Qwen/Qwen3.5-27B-FP8"
# --------------------------
# FIX: Clean Stale FlashInfer Cache
# --------------------------
rm -rf ~/.cache/flashinfer

# Activate virtual environment
source /home/cychan/vLLM/.venv/bin/activate

# Enable memory profiler to estimate CUDA graphs v0.19 functionality
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SLEEP_WHEN_IDLE=1
# Start vLLM with reduced swap space
vllm serve $MODEL_NAME \
  --served-model-name Qwen3.5-27B \
  --chat-template qwen3.5-enhanced.jinja \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 219520 \
  --gpu-memory-utilization 0.91 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 12288 \
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
