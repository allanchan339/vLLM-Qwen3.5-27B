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

**Optional NCCL Tuning** (provides additional stability margin):
```bash
export NCCL_P2P_DISABLE=1    # Recommended for mixed GPU
export NCCL_IB_DISABLE=1     # Force PCIe
export NCCL_ALGO=Ring        # Stable algorithm
```

**Benefits**:
- ✅ 219k context length
- ✅ Lower VRAM usage
- ✅ Faster inference
- ✅ Proven stable (Knowledge Platform)

**Note**: The Jinja template is the PRIMARY fix. NCCL tuning is secondary optimization.

---

### 3. Model & Quantization Selection

#### For Maximum Context (219k)
**Recommended**: `Qwen/Qwen3.5-27B-FP8`
- Use with TP mode + FP8 KV cache
- Custom Jinja template for stable tool calling
- Proven stable in production

#### For Maximum Stability
**Alternative**: `QuantTrio/Qwopus3.5-27B-v3-AWQ`
- AWQ (INT4) uniform quantization
- Distilled from Claude 4.6 Opus
- Better tool calling out-of-the-box
- Use with Hermes parser

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

#### Distilled Models (Qwopus series)
```bash
--tool-call-parser hermes
--chat-template qwen3.5-barubary-attuned-chat-template_hermes.jinja
```

**Benefits**:
- Trained on Claude's Hermes format
- 17/17 tool calling accuracy reported
- No distillation artifacts

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
