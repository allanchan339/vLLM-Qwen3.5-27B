# Qwen 3.5 27B/35BA3B Tool Calling Issues: Why It Breaks & How I Fixed It

**TL;DR**: Everyone's talking about Qwen 3.5's reasoning quality and slow TTFT, but nobody's discussing the tool calling issues that actually break agentic workflows. Here's what broke my setup and how I fixed it after weeks of debugging.

---

## The Story So Far

I've been running Qwen 3.5-27B on a mixed GPU setup (RTX 4090 + 3090) for about a month now. The reasoning capabilities are genuinely impressive, and yeah, the TTFT is slow, but honestly? **The tool calling issues are what actually matter if you're doing agentic work.**

After countless hours of debugging, failed runs, and reading through vLLM source code, I finally have a stable setup. I wanted to share what I learned because the official docs don't quite cover the real-world edge cases.

---

## Issue #1: The Jinja Template Problem (CRITICAL - This Broke Everything)

### What Happened

I started with the official `qwen3.5_official.jinja` template. Everything looked fine for the first few tool calls. Then suddenly:

- Tool calls appeared **mid-thought** (closing `</think>` without having opened `<think>` )
- Random **premature stops** in the middle of XML tool calls, e.g "Let me do that for you:" and stopped suddenly without finishing the tool call
- Historical thinking blocks **leaking into the context** and confusing the model

I thought it was my fault. I thought it was vLLM. I even thought my mixed GPU setup was the culprit.

**Turns out, it's the template.**

### Why It Happens

Here's the thing: the official template has edge cases that **122B+ models handle gracefully but 27B/35B don't**. Smaller models have less robust instruction following, and those edge cases cause silent failures.

### The Fix

I ended up using a custom **M2.5-style interleaved thinking template** (`qwen3.5-enhanced.jinja`) that:

- Properly closes `</thinking>` **before** tool calls (not after)
- Hides historical reasoning from context but keeps current reasoning visible
- Uses XML formatting that doesn't accidentally trigger `<stop>` tokens
- Handles the edge cases that smaller models struggle with

**The most important part**: vLLM does NOT auto-detect templates. You have to manually specify:

```bash
--chat-template qwen3.5-enhanced.jinja
```

Without this flag, your model uses the default template and will exhibit instability **no matter what other optimizations you try**.

---

## Issue #2: The Tool Call Parser (Deviation from Official Docs)

### The Official Recommendation

The [Qwen3.5-27B-FP8 HuggingFace page](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) says to use:

```bash
--tool-call-parser qwen3_coder
```

### What Actually Works

**`qwen3_coder` breaks on complex tool calls.** I spent way too much time on this one.

After digging into vLLM's source code, here's what I found:

| Parser | How It Works | Special Characters | Nested JSON | Malformed XML |
|--------|--------------|-------------------|-------------|---------------|
| `qwen3_coder` | Regex string extraction | ❌ Breaks on `<`, `>`, `&` | ❌ Corrupts during streaming | ❌ Fails hard |
| `qwen3_xml` | C-based `xml.parsers.expat` | ✅ Auto-sanitizes | ✅ Deferred parsing | ✅ Auto-heals |

**Real example**: If your tool call contains code like `if (a < b)`, `qwen3_coder`'s regex parser breaks because `<` and `>` mess up the pattern matching. `qwen3_xml` handles it natively because it's an actual XML parser.

### The Fix

```bash
--tool-call-parser qwen3_xml
```

Yes, this goes against the official recommendation. But for **long-context agentic work** (think 50K+ tokens), `qwen3_xml` is just more stable. The C-based parser is robust, auto-heals malformed XML, and doesn't try to parse nested JSON mid-stream.

---

## Issue #3: Mixed GPU Precision Drift

### The Problem

I'm running on a mixed GPU setup (RTX 4090 + 3090). Tensor Parallelism splits matrix multiplication across both GPUs, but they have different compute capabilities:

- **RTX 4090 (SM89)**: Has native FP8 W8A8 tensor cores
- **RTX 3090 (SM80)**: No native FP8, falls back to W8A16

**Result**: Different precision levels → mismatched intermediate results → error accumulation in long conversations.

I noticed conversations would drift after 30-40K tokens. Tool calls would become inconsistent. Reasoning quality would degrade.

### The Fix

```bash
export VLLM_TEST_FORCE_FP8_MARLIN=1
```

This forces the 4090 to use W8A16 (matching the 3090) instead of its native W8A8. Both GPUs now use the same precision, eliminating drift.

**Additional NCCL tuning** (helps with stability on mixed topologies):

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
```

---

## Issue #4: Model Choice Matters (Avoid SFT-Distilled Variants for Long Context)

### The Qwopus3.5 Trap

Models like `QuantTrio/Qwopus3.5-27B-v3-AWQ` are SFT-distilled from Claude 4.6 Opus. They look great initially:

- **First ~65K tokens**: Works perfectly, tool calling is stable
- **After 65K+ tokens**: Output starts **mixing XML and JSON formats**

**Why?** SFT (Supervised Fine-Tuning) shifted the tool calling format from `qwen3_xml` to `hermes` (JSON-based) to align with the output format of Claude, but it doesn't fully align the underlying token probabilities. In long contexts, the model drifts between its original Qwen XML format and the SFT'd JSON format.

**I lost hours debugging this** before realizing the model itself was the problem.

### What to Use Instead

**For 48GB VRAM** (best quality):
```bash
Qwen/Qwen3.5-27B-FP8
```
- Near-lossless accuracy
- Full 219K context support
- Stable tool calling with custom template

**For <48GB VRAM** (accept some accuracy loss):
```bash
Intel/Qwen3.5-27B-int4-AutoRound
```
- Saves ~4GB VRAM
- Still stable with custom template
- Higher perplexity than FP8 (INT4 isn't lossless)

---

## My Working Configuration (Production-Tested)

After all this, here's what actually works (independent repo example + 3 days of actual production use):

```bash
# Environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export VLLM_TEST_FORCE_FP8_MARLIN=1

# vLLM serve command
vllm serve Qwen/Qwen3.5-27B-FP8 \
  --served-model-name Qwen3.5-27B \
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
  --host 0.0.0.0 \
  --port 8000 \
  --language-model-only
```

---

## Real-World Test Results

I validated this setup with a **1h 9m continuous agentic session**:

- ✅ **138.2K tokens** generated
- ✅ **Stable tool calling** throughout (no format drift)
- ✅ **M2.5-style interleaved thinking** maintained coherence
- ✅ **Built a production-ready knowledge graph platform** (FastAPI + React) in 18 minutes of uninterrupted work

The model autonomously built the entire platform without tool calling failures. That's the kind of stability you need for actual agentic workflows.

---

## Key Takeaways

1. **Jinja template matters more than you think** - The custom template is CRITICAL for 27B/35B models. Official template has edge cases that break smaller models.

2. **Don't trust official parser recommendations blindly** - `qwen3_xml` beats `qwen3_coder` for agentic work. The C-based XML parser is just more robust.

3. **Mixed GPU needs precision alignment** - `VLLM_TEST_FORCE_FP8_MARLIN=1` is non-optional if you have different GPU generations.

4. **Avoid SFT-distilled models for long context** - Format drift after 65K tokens is real. Use official Qwen FP8 instead.

5. **FP8 quantization is near-lossless** - Don't sacrifice accuracy unless VRAM absolutely forces you to.

---

## Resources

- **My working setup**: [GitHub - vLLM Qwen 3.5 Config](https://github.com/allanchan339/vLLM-Qwen3.5-27B)
- **Concrete test example**: [qwen_own_project](https://github.com/allanchan339/qwen_own_project) - 1h 9m continuous agentic session with the model
- **Custom Jinja template**: `qwen3.5-enhanced.jinja` (in the repo)
- **Original discussion**: [Tool calling fixes thread](https://www.reddit.com/r/LocalLLaMA/comments/1sdhvc5/qwen_35_tool_calling_fixes_for_agentic_use_whats/)

---

**Edit**: Added real test results from production deployment. The setup's been running stable for weeks now.

**Edit 2**: Clarified that the `qwen3_xml` recommendation is based on vLLM source code analysis, not just empirical testing. The C-based XML parser is fundamentally more robust than regex extraction.

**Edit 3**: Added note about SFT-distilled models - this one cost me the most debugging time. Avoid Qwopus3.5 series for long-context work.

---

*If you're running Qwen 3.5-27B/35B for agentic tasks and experiencing silent tool calling failures, check your Jinja template first. It's almost certainly the culprit. The official template just doesn't handle edge cases that smaller models encounter.*

Happy deploying! 🚀
