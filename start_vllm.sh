#!/bin/bash
# Wrapper script that delegates to start_vllm_AWQ_Claude.sh (recommended config)
# Uses AWQ quantization + PP mode for mixed GPU stability
# Kept for compatibility with services that expect start_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/start_vllm_FP8.sh" "$@"
