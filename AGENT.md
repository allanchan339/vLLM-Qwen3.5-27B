# Agent Guidelines: vLLM + Qwen3.5-27B on Mixed GPUs

**Last Updated**: April 2026 - **Verified**: Knowledge Platform (138.2k tokens, 1h 9m stable session)

---

## ⚠️ ROOT CAUSE: Jinja Template (MUST FIX FIRST)

**Without this fix, ALL other optimizations are useless.**

### The Critical Issue

Qwen3.5-27B is unstable with the official template (`qwen3.5_official.jinja`) due to:
- Tool calls generated mid-thought without closing `</thinking>` tags
- Edge cases that 122B+ models handle gracefully but 27B fails on
- Distillation artifacts causing premature `<stop>` tokens

**vLLM does NOT auto-detect templates. You MUST manually specify:**

```bash
--chat-template qwen3.5-enhanced.jinja
```

### Why the Custom Template Works

`qwen3.5-enhanced.jinja` implements M2.5-style interleaved thinking:
- ✅ Proper `</thinking>` tag handling before tool calls
- ✅ Historical reasoning hidden, current reasoning preserved
- ✅ XML format that avoids `<stop>` token issues
- ✅ Robust edge case handling for smaller models (27B)

**Without this flag, the model will fail regardless of NCCL tuning, quantization, or GPU setup.**

---

## Other Critical Requirements

### 1. Version Compatibility
- **vLLM 0.19.0 requires transformers 5.5** (not 4.49) for Qwen3.5-27B RoPE
- After installing vLLM, run:
  ```bash
  uv pip install -U transformers
  ```
- **Verification**: Run `transformers.__version__` - must be `>= 5.5`

---

### 2. GPU Parallelism: TP Mode Works (with Jinja template)

**TP Mode is RECOMMENDED** when using the correct Jinja template:

```bash
--tensor-parallel-size 2
--kv-cache-dtype fp8
```

**CRITICAL: Force Consistent FP8 Behavior**
```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1  # Force 4090 to use W8A16 (match 3090)
```

**Why this is critical**:
- RTX 4090 (SM89): Has native FP8 W8A8 tensor cores
- RTX 3090 (SM80): No native FP8, uses W8A16
- **Without this flag**: Precision mismatch → error accumulation
- **With this flag**: Both GPUs use Marlin W8A16 → consistent results

**Additional NCCL Tuning** (provides stability margin):
```bash
export NCCL_P2P_DISABLE=1    # Disable P2P communication
export NCCL_IB_DISABLE=1     # Force PCIe
export NCCL_ALGO=Ring        # Stable algorithm
```

**Benefits**:
- ✅ 219k context length
- ✅ Lower VRAM usage
- ✅ Faster inference
- ✅ Proven stable (Knowledge Platform)

**Note**: The Jinja template is PRIMARY. `VLLM_TEST_FORCE_FP8_MARLIN=1` is SECONDARY but critical for mixed GPU. NCCL tuning is optional optimization.

---

### 3. Model & Quantization Selection

#### For Maximum Context (219k)
**Recommended**: `Qwen/Qwen3.5-27B-FP8`
- Use with TP mode + FP8 KV cache
- Custom Jinja template for stable tool calling
- Proven stable in production

#### ⚠️ AVOID for Long Context: Qwopus3.5 Series
**Trap**: `QuantTrio/Qwopus3.5-27B-v3-AWQ` and similar SFT-distilled models

**Why it's a trap**:
- SFT from Claude 4.6 Opus shifted tool calling from `qwen3_xml` → `hermes` (JSON)
- **Appears stable**: Works fine for first ~65K tokens
- **Fails in long context**: After 65K+ tokens, output **mixes XML and JSON formats**
- **Root cause**: SFT doesn't maintain format consistency as well as base model fine-tuning

**Recommendation**: For long-context agentic work (>65K tokens), use official `Qwen/Qwen3.5-27B-FP8` with custom Jinja template.

---

### 4. Tool Calling Configuration

#### Official Qwen3.5-27B-FP8
```bash
--tool-call-parser qwen3_xml
--chat-template qwen3.5-enhanced.jinja
--reasoning-parser qwen3
```

**Why it works now**:
- Custom Jinja template handles `</thinking>` tag properly
- XML format avoids `<stop>` token issues
- M2.5-style interleaved thinking prevents premature stops

#### Distilled Models (Qwopus series) - NOT RECOMMENDED for Long Context
```bash
--tool-call-parser hermes
--chat-template qwen3.5-barubary-attuned-chat-template_hermes.jinja
```

**⚠️ Warning**: Only use for short contexts (<65K tokens)

**Why**:
- SFT from Claude shifted format from XML → JSON
- After 65K+ tokens: Output mixes XML and JSON (unstable)
- 17/17 tool calling accuracy only holds for short sessions

**For long-context work**: Use official Qwen3.5-27B-FP8 with custom Jinja template instead.

---

## Quick Start Checklist

### For FP8 + TP (Recommended)
- [ ] `transformers>=5.5` installed
- [ ] Using `Qwen/Qwen3.5-27B-FP8`
- [ ] `--tensor-parallel-size 2` (TP mode)
- [ ] `--kv-cache-dtype fp8`
- [ ] `--tool-call-parser qwen3_xml`
- [ ] `--chat-template qwen3.5-enhanced.jinja`
- [ ] NCCL env vars set (`P2P_DISABLE=1`, `ALGO=Ring`)
- [ ] `--max-model-len 219520`

### For AWQ + TP (Alternative)
- [ ] Using `QuantTrio/Qwopus3.5-27B-v3-AWQ`
- [ ] `--tensor-parallel-size 2`
- [ ] `--tool-call-parser hermes`
- [ ] `--max-model-len 219520`

---

## Environment Variables (Critical)

```bash
# CUDA setup
export CUDA_HOME=/usr
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# NCCL tuning for mixed GPU
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_CUMEM_ENABLE=0

# vLLM optimizations
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# CRITICAL for mixed GPU: Force 4090 to use W8A16 (match 3090)
export VLLM_TEST_FORCE_FP8_MARLIN=1

# Thread control
export OMP_NUM_THREADS=4
```

---

## Troubleshooting

### Issue: Precision drift in long conversations
**Solution**: Verify all NCCL env vars are set. Try adding:
```bash
export NCCL_NVLS_DISABLE=1
```

### Issue: Tool calling stops prematurely
**Solution**: 
1. Verify `qwen3.5-enhanced.jinja` is in working directory
2. Check `--tool-call-parser qwen3_xml` is set
3. Consider switching to distilled model (Qwopus)

### Issue: OOM errors
**Solution**:
- Reduce `--gpu-memory-utilization` to 0.85
- Reduce `--max-model-len` to 131072
- Reduce `--max-num-seqs` to 2

---

## References
- [Knowledge Platform Project](https://github.com/allanchan339/qwen_own_project) - Working example
- [vLLM Issue #34437](https://github.com/vllm-project/vllm/issues/34437) - Mixed GPU discussion
- [Start Script](start_vllm_FP8.sh) - Complete working configuration
