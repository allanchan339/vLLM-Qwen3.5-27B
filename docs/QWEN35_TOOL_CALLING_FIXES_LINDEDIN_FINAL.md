# Qwen 3.5 27B/35B Tool Calling Issues: Why It Breaks & How I Fixed It 🚀

**TL;DR**: Everyone is discussing Qwen 3.5's reasoning and TTFT, but few have covered the tool calling issues that break agentic workflows. After weeks of debugging, I've resolved the key problems. Here's what I learned. 👇

---

## The Journey So Far 🛠️

I've been running Qwen 3.5-27B on a mixed GPU setup (RTX 4090 + 3090). While the reasoning is impressive, **stable tool calling is the real make-or-break factor for AI agents.**

After extensive debugging and deep dives into vLLM source code, I've built a stable production setup. The official docs don't cover these real-world edge cases—so I'm sharing them to help avoid similar issues.

---

## 1. The Jinja Template Problem (CRITICAL) ⚠️

### The Problem
The official `qwen3.5_official.jinja` template works... until it doesn't. On 27B/35B models, I observed:
- Tool calls appearing mid-thought.
- Premature stops during XML generation.
- Context leakage that confused the model.

### The Insight
Smaller models aren't as "forgiving" as 122B+ variants. They need precise formatting.

### The Solution
I developed a custom **M2.5-style interleaved thinking template** that:
✅ Closes reasoning blocks *before* tool calls.
✅ Keeps the context clean.
✅ Avoids accidental `<stop>` triggers.

**Pro-tip**: vLLM won't auto-detect templates. You MUST manually specify:
`--chat-template qwen3.5-enhanced.jinja`

---

## 2. Parser Choice: XML vs. Regex ⚔️

### The Issue
Official docs suggest `qwen3_coder`. **It doesn't work reliably for complex tool calls.** It breaks on code containing special characters (like `if (a < b)`) because regex can't handle `<`, `>`, `&` inside tool call arguments.

### The Fix
Switch to:
`--tool-call-parser qwen3_xml`

The C-based `expat` parser is robust. It auto-sanitizes, handles nested JSON, and even "auto-heals" malformed XML. It's the better choice for long-context (50K+ token) work.

---

## 3. Mixed GPU Precision Drift ⚖️

Running a 4090 (SM89) with a 3090 (SM80)? You'll see precision drift after ~30K tokens, leading to inconsistent reasoning.

**The Fix**: Force alignment.
`export VLLM_TEST_FORCE_FP8_MARLIN=1`

This ensures both GPUs operate with consistent precision, eliminating drift and maintaining agent stability.

---

## 4. The "Qwen3.5-Distilled-Claude" Trap: Avoid SFT-Distilled Models for Long Context 🪤

Distilled models like `Qwopus3.5` look promising initially, but after 65K tokens, they start mixing formats (XML and JSON). The SFT process shifts token probabilities without full alignment, causing format drift in long contexts.

Stick to the **Official Qwen/Qwen3.5-27B-FP8** for near-lossless accuracy and consistent stability.

---

## Validation Results 📈

Tested this setup in a **70-minute continuous agentic session**:
✅ **138.2K tokens** generated.
✅ **Zero** format drift.
✅ **Built a full-stack Knowledge Graph platform** (FastAPI + React) in just 18 minutes of autonomous work.

This is the stability required for production Agentic AI.

---

## Key Takeaways:
1️⃣ **Templates matter more than expected.** Don't rely on defaults.
2️⃣ **XML > Regex** for robust tool calling.
3️⃣ **Align precision** on mixed hardware.
4️⃣ **Avoid SFT-distilled models** for long-context work.
5️⃣ **FP8 quantization** is near-lossless.

**Full config & custom templates**: https://github.com/allanchan339/vLLM-Qwen3.5-27B

Let's keep building. 🚀

#LLM #vLLM #Qwen #MLEngineering #AgenticAI #LocalLLaMA #AIEngineering #LLMOps #GenerativeAI #TechCommunity
