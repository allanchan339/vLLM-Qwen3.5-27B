# Running Qwen3.5-27B on Mixed GPU Setup: From Pain to Success

## Success Story: Knowledge Platform

**This configuration now powers a production knowledge graph extraction system** - see [qwen_own_project](https://github.com/allanchan339/qwen_own_project) for the complete working example.

### Test Setup & Results

The configuration was validated through a comprehensive stability test documented in the [Knowledge Platform README](https://github.com/allanchan339/qwen_own_project/blob/main/README.md):

- **Test Duration**: 1h 9m continuous agentic session
- **Token Usage**: 138.2k tokens
- **Workload**: Full-stack application development (FastAPI backend + React frontend)
- **Tool Calling**: Stable throughout session with XML parser
- **Reasoning**: M2.5-style interleaved thinking maintained coherence
- **Result**: Successfully built a production-ready knowledge graph platform

**Key Achievement**: The LLM autonomously built the entire platform in 18 minutes of uninterrupted work, demonstrating stable long-context reasoning and reliable tool calling on mixed GPU hardware.

See [qwen_own_project README](https://github.com/allanchan339/qwen_own_project/blob/main/README.md) for complete test methodology and results.

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

## Root Cause: Jinja Template Instability (CRITICAL)

**This is the PRIMARY issue that must be fixed first.**

### The Problem

The official `qwen3.5_official.jinja` template has edge cases that cause instability in Qwen3.5-27B but NOT in larger models (122B+):

1. **Tool calling mid-thought**: Model generates `</think>` tags without properly closing `<think>` blocks
2. **Premature stops**: XML tool calls trigger `<stop>` tokens incorrectly
3. **Reasoning leakage**: Historical thinking blocks not properly hidden from context

**Why 27B vs 122B+?**
- Smaller models have less robust instruction following
- Edge cases in template logic that larger models handle gracefully
- Distillation artifacts from training data

### The Solution: Custom Template

Use `qwen3.5-enhanced.jinja` which implements M2.5-style interleaved thinking:

```bash
--chat-template qwen3.5-enhanced.jinja
```

**Key improvements**:
- Proper `</thinking>` tag handling before tool calls
- Historical reasoning hidden, current reasoning preserved
- XML format that avoids `<stop>` token issues
- Robust edge case handling for smaller models

**CRITICAL**: vLLM does NOT auto-detect templates. You MUST manually specify:
```bash
vllm serve ... --chat-template qwen3.5-enhanced.jinja ...
```

Without this flag, the model will use the default template and exhibit instability regardless of other optimizations.

---

## Other Problems Solved

### Problem 1: Version Compatibility

**Issue**: vLLM 0.19.0 partially supports Transformers 5, but defaults to 4.49. Qwen3.5-27B requires Transformers 5.3+ for new RoPE implementation.

**Solution**:
```bash
uv pip install -U transformers  # Upgrade to 5.5+
```

---

### Problem 2: Mixed GPU Precision Drift (Secondary Issue)

**Note**: This was initially suspected as the root cause, but the Jinja template is PRIMARY. NCCL tuning + FP8 Marlin force is still recommended for optimal stability.

**The Issue**: Tensor Parallelism (TP mode) splits matrix multiplication across GPUs. Different compute capabilities (SM80 vs SM89) use different FP8 implementations.

**Why It Matters**:
- RTX 4090 (SM89): Native FP8 W8A8 tensor cores
- RTX 3090 (SM80): No native FP8, falls back to W8A16
- **Problem**: Different precision → mismatched results → error accumulation

**The Fix: Force Consistent FP8 Behavior**
```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1  # Critical: Force 4090 to use W8A16 (match 3090)
```

**Why this works**:
- Without this flag: 4090 uses native W8A8, 3090 uses W8A16 → precision drift
- With this flag: Both GPUs use Marlin W8A16 → consistent results

**Additional NCCL Tuning** (provides stability margin):
```bash
export NCCL_P2P_DISABLE=1    # Disable P2P communication
export NCCL_IB_DISABLE=1     # Force PCIe
export NCCL_ALGO=Ring        # Stable algorithm
```

**Important**: `VLLM_TEST_FORCE_FP8_MARLIN=1` is critical for mixed GPU setups. NCCL tuning provides additional stability margin.

---

### Problem 3: Quantization Tradeoffs

**FP8 Quantization**:
- **Pros**: Minimal accuracy loss vs FP16, native support on RTX 4090
- **Cons**: RTX 3090 falls back to W8A16, potential precision drift
- **Best for**: 48GB VRAM setups needing maximum context length

**AWQ Quantization**:
- **Pros**: Uniform 0-1 range, consistent across different GPU architectures
- **Cons**: Higher precision requirements, slightly more VRAM usage
- **Best for**: Mixed GPU setups prioritizing stability over context length

**⚠️ WARNING: Qwopus3.5 Series is a TRAP for Long Context**

Models like `QuantTrio/Qwopus3.5-27B-v3-AWQ` are SFT-distilled from Claude 4.6 Opus, which:
- **Shifted tool calling format**: From `qwen3_xml` → `hermes` (JSON-based)
- **Appears stable initially**: Works fine for first ~65K tokens
- **Fails in long context**: After 65K+ tokens, output **mixes XML and JSON formats**
- **Root cause**: SFT doesn't maintain format consistency as well as base model fine-tuning

**Why this happens**: SFT (Supervised Fine-Tuning) changes the model's output distribution to match Claude's Hermes format, but doesn't fully align the underlying token probabilities. In long contexts (>65K tokens), the model drifts between its original Qwen XML format and the SFT'd JSON format.

**Recommendation**: For long-context agentic work (>65K tokens), use official `Qwen/Qwen3.5-27B-FP8` with custom Jinja template instead of SFT-distilled variants.

---

### Problem 4: Context Length vs VRAM

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

# FP8 configuration (CRITICAL for mixed GPU)
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_TEST_FORCE_FP8_MARLIN=1  # Force 4090 to use W8A16 (match 3090, avoid precision drift)
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
- **Do not use Qwopus3.5 series** for long-context work (format drift after 65K tokens)

### Issue: Precision drift in long conversations
- Verify all NCCL environment variables are set
- Check `VLLM_TEST_FORCE_FP8_MARLIN=1` is set
- Try `export NCCL_NVLS_DISABLE=1`

---

## References

- [vLLM Issue #34437](https://github.com/vllm-project/vllm/issues/34437) - Mixed GPU TP mode instability
- [Reddit Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1sdhvc5/qwen_35_tool_calling_fixes_for_agentic_use_whats/) - Tool calling fixes
- [Qwen3 Issue #1831](https://github.com/QwenLM/Qwen3/issues/1831) - Official model issues
- [NVIDIA Forums](https://forums.developer.nvidia.com/t/success-with-quanttrio-qwen3-5-27b-claude-4-6-opus-reasoning-distilled-v2-awq/365416) - AWQ success reports (note: long-context issues not discussed)

---

## License

MIT
