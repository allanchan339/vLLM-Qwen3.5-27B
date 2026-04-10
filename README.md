# Running Qwen3.5-27B on Mixed GPU Setup: From Pain to Success

## Success Story: Knowledge Platform

**This configuration now powers a production knowledge graph extraction system** - see [qwen_own_project](https://github.com/allanchan339/qwen_own_project) for the complete working example.

**Results**: 138.2k tokens over 1h 9m of continuous agentic work with stable tool calling and reasoning.

---

## Hardware & Environment

### Mixed GPU Setup
- **RTX 4090** (SM89) - 24GB VRAM
- **RTX 3090** (SM80) - 24GB VRAM
- Intel i9-13900K - 32GB RAM
- **Total VRAM**: 48GB

### Software Stack
- Windows 11 + WSL2 Ubuntu 22.04
- NVIDIA Driver: 591.86
- CUDA: 12.8
- vLLM: 0.19.0
- Transformers: 5.5+

---

## Quick Start (Working Configuration)

### For FP8 Quantization (Recommended for 48GB VRAM)

```bash
./start_vllm_FP8.sh
```

This uses:
- **Tensor Parallelism (TP=2)** - splits model across both GPUs
- **FP8 KV Cache** - reduces memory overhead
- **Custom Jinja template** - stable tool calling with reasoning
- **FlashInfer backend** - optimized attention kernels

### For AWQ Quantization (Alternative)

```bash
./start_vllm_AWQ_Claude_TP.sh
```

See "Quantization Comparison" below for tradeoffs.

---

## The 5 Problems Solved

### Problem 1: Version Compatibility

**Issue**: vLLM 0.19.0 partially supports Transformers 5, but defaults to 4.49. Qwen3.5-27B requires Transformers 5.3+ for new RoPE implementation.

**Solution**:
```bash
uv pip install -U transformers  # Upgrade to 5.5+
```

---

### Problem 2: Mixed GPU Precision Drift

**The Core Issue**: Tensor Parallelism (TP mode) splits matrix multiplication across GPUs. Different compute capabilities (SM80 vs SM89) produce small precision differences in Marlin kernels that accumulate via all-reduce operations.

**Why It Matters**:
- RTX 4090 (SM89): Native FP8 W8A8 support
- RTX 3090 (SM80): Falls back to W8A16
- Result: Precision mismatch → error accumulation → "garbage in, garbage out"

**Initial Workaround (PP Mode)**:
Pipeline Parallelism avoids mixed-precision by splitting layers sequentially, but doubles KV cache VRAM usage, reducing context from 220k → 100k tokens.

**Final Solution (FP8 with NCCL Tuning)**:
```bash
export NCCL_P2P_DISABLE=1    # Disable P2P to avoid precision mismatch
export NCCL_IB_DISABLE=1     # Force PCIe communication
export NCCL_ALGO=Ring        # Stable ring algorithm
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
```

Combined with FP8 KV cache (`--kv-cache-dtype fp8`), this achieves **219k context length** with stable outputs.

---

### Problem 3: Tool Calling Instability

**Issue**: Official Qwen3.5-27B has distillation instability - generates tool calls mid-thought without closing `</thinking>` tag, causing premature stops.

**Solution**: Custom Jinja template (`qwen3.5-enhanced.jinja`) with:
- M2.5-style interleaved thinking (hides historical reasoning, keeps current)
- XML tool call format that avoids `<stop>` token issues
- Proper `</thinking>` tag handling

**Alternative**: Use distilled models like `QuantTrio/Qwopus3.5-27B-v3-AWQ` or `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` with Hermes parser.

---

### Problem 4: Quantization Tradeoffs

**FP8 Quantization**:
- **Pros**: Minimal accuracy loss vs FP16, native support on RTX 4090
- **Cons**: RTX 3090 falls back to W8A16, potential precision drift
- **Best for**: 48GB VRAM setups needing maximum context length

**AWQ Quantization**:
- **Pros**: Uniform 0-1 range, consistent across different GPU architectures
- **Cons**: Higher precision requirements, slightly more VRAM usage
- **Best for**: Mixed GPU setups prioritizing stability over context length

**Recommendation**: Start with FP8 (this repo's default). If you encounter instability, switch to AWQ.

---

### Problem 5: Context Length vs VRAM

**VRAM Breakdown (48GB Total)**:

| Component | FP8 (TP Mode) | AWQ (PP Mode) |
|-----------|---------------|---------------|
| Model Weights | ~16-18 GB | ~16-18 GB |
| KV Cache (219k context) | ~18-20 GB | N/A |
| KV Cache (100k context) | N/A | ~18-22 GB |
| CUDA Graphs | ~2 GB | ~2 GB |
| System + Headroom | ~2 GB | ~2 GB |
| **Total** | **~38-42 GB** | **~38-44 GB** |

**Key Insight**: TP mode with FP8 KV cache achieves 219k context. PP mode with AWQ is limited to ~100k due to doubled KV cache allocation.

---

## Complete Working Configuration

### Environment Variables

```bash
# CUDA PATH SETTINGS
export CUDA_HOME=/usr
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Safe, Speed-Focused Env Vars
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export OMP_NUM_THREADS=4

# NCCL tuning for mixed GPU topology
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring

# FP8 configuration
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1
```

### vLLM Serve Command

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --served-model-name vllm/Qwen3.5-27B \
  --chat-template qwen3.5-enhanced.jinja \
  --attention-backend FLASHINFER \
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
```

---

## Verification

Test the setup with:

```bash
# Test basic generation
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# Test tool calling
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
  }'
```

---

## Troubleshooting

### Issue: "Out of Memory" errors
- Reduce `--gpu-memory-utilization` to 0.85
- Reduce `--max-model-len` to 131072
- Reduce `--max-num-seqs` to 2

### Issue: Unstable tool calling
- Verify `qwen3.5-enhanced.jinja` is in the working directory
- Check `--tool-call-parser qwen3_xml` is set
- Consider switching to distilled model with Hermes parser

### Issue: Precision drift in long conversations
- Verify all NCCL environment variables are set
- Try `export NCCL_NVLS_DISABLE=1`
- Consider switching to AWQ quantization

---

## References

- [vLLM Issue #34437](https://github.com/vllm-project/vllm/issues/34437) - Mixed GPU TP mode instability
- [Reddit Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1sdhvc5/qwen_35_tool_calling_fixes_for_agentic_use_whats/) - Tool calling fixes
- [Qwen3 Issue #1831](https://github.com/QwenLM/Qwen3/issues/1831) - Official model issues
- [NVIDIA Forums](https://forums.developer.nvidia.com/t/success-with-quanttrio-qwen3-5-27b-claude-4-6-opus-reasoning-distilled-v2-awq/365416) - AWQ success reports

---

## License

MIT
