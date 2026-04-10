#!/bin/bash

# vLLM Benchmark Script
# Tests server performance with concurrent requests

set -e

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Configuration
MODEL_NAME="Qwen/Qwen3.5-27B-FP8"
SERVED_MODEL_NAME="vllm/Qwen3.5-27B"
BASE_URL="http://localhost:8000"
ENDPOINT="/v1/chat/completions"
INPUT_LEN=2048
OUTPUT_LEN=512
NUM_PROMPTS=50
REQUEST_RATE=1

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ Error: vLLM is not installed or not in PATH"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check if server is running
echo "🔍 Checking if server is running at $BASE_URL..."
if ! curl -s "$BASE_URL/models" > /dev/null; then
    echo "❌ Error: Server is not responding at $BASE_URL"
    echo "Make sure vLLM server is started first"
    exit 1
fi
echo "✅ Server is running"

echo ""
echo "🚀 Starting vLLM Benchmark..."
echo "━━━━━━━━ Parameters ="
echo "  HF Model:       $MODEL_NAME"
echo "  API Model Name: $SERVED_MODEL_NAME"
echo "  Endpoint:       $ENDPOINT"
echo "  Input tokens:   $INPUT_LEN"
echo "  Output tokens:  $OUTPUT_LEN"
echo "  Num prompts:    $NUM_PROMPTS"
echo "  Request rate:   $REQUEST_RATE req/s"
echo ""

# Run benchmark
vllm bench serve \
    --backend openai-chat \
    --endpoint "$ENDPOINT" \
    --model "$MODEL_NAME" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_PROMPTS \
  

echo ""
echo "✅ Benchmark completed!"
