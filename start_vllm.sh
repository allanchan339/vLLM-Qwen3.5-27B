#!/bin/bash
# Wrapper script that delegates to start_paroquant.sh
# Kept for compatibility with services that expect start_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/start_vllm_autoround.sh" "$@"
