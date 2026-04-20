# Qwen 3.6-35B-A3B: Reddit Asked, So I Tested If the 3.5 Tool Calling Fixes Carry Over

**TL;DR**: Following up on the [Qwen 3.5 thread](https://www.reddit.com/r/vLLM/comments/1skks8n/) — after everyone kept asking about 3.6, I set it up using the same `qwen3_xml` + `enhanced.jinja` fixes and ran real agentic tests. Here's the honest result: my config is still the most stable, but compared to Qwen3.5-27B, Qwen3.6-35B-A3B is notably more loopy and has a higher chance of malformed tool calls interrupting an agentic process. 

---

## The Short Story

After spending weeks ironing out Qwen 3.5-27B/35B for agentic use — same fixes, same template, same GPU tuning — people on Reddit kept asking about Qwen 3.6.

So I set it up and ran real agentic tests. Gave the model full ownership of the folder, and asked it to build a full-stack project with frontend and backend, with a prompt of $10k token budget. Wanted to see how it holds up in practice.

My config (enhanced.jinja + qwen3_xml) is still the most stable option. But compared to Qwen3.5-27B, Qwen3.6-35B-A3B has two new problems:

1. **More looping** — the model gets stuck in reasoning loops more often
2. **Malformed tool calls interrupting agentic flow** — higher chance of breaking mid-task, even with the same config that works perfectly on 3.5

---

## What Carried Over (Still Works)

### `qwen3_xml` parser

Registry-based parser handles complex tool arguments without corruption. Official docs still say `qwen3_coder`. I still say no.

### `qwen3.5-enhanced.jinja` template

The interleaved thinking template works on 3.6. Proper `</thinking>` tag handling, clean tool call formatting. 

### Precision drift on mixed GPUs

RTX 4090 (SM89) wants W8A8, RTX 3090 (SM80) falls back to W8A16. `VLLM_TEST_FORCE_FP8_MARLIN=1` still forces both to match. Without it, conversations drift.

### NCCL tuning

Same setup: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_ALGO=Ring`. Same reason: mixed topology stability.

---

## Real Agentic Test: Three Runs

I gave each trail the same prompt: full ownership of the folder, build a full-stack project with frontend and backend, $10k token budget.

### Run 1: `enhanced.jinja` + `qwen3_xml` **(my config)**

This is the one that lasted the longest.

| Prompt | Accumulated Tokens |
|--------|-------------------|
| Project setup | 13.9k |
| "Did you check if this is bug free? This is your own project." | 135.1K |
| DCP sweep auto-triggered | 107.0K |
| "Fix it then" | 110.0K |
| **Model died** - improper tool calling | 111.1K |

This config survived to ~111K tokens before dying from improper tool calling. The DCP sweep at 135K dropped it to 107K, but it kept going. For context, the 3.5 model with the same setup routinely goes 130K+ without any interruption.

### Run 2: `official.jinja` + `qwen3_coder`

**Died in 6m 32s** — improper tool calling.

### Run 3: `official.jinja` + `qwen3_xml`

**Died in <1m 16s** — malformed tool calls inside the thinking box.

### Comparison Summary

| Config | Survival | Failure Mode |
|--------|----------|-------------|
| `enhanced.jinja` + `qwen3_xml` | ~111K tokens | Improper tool calling (died) |
| `official.jinja` + `qwen3_coder` | 6m 32s | Improper tool calling |
| `official.jinja` + `qwen3_xml` | ~1m 16s | Malformed tool calls in thinking box |

For comparison, the same test on Qwen3.5-27B with `enhanced.jinja` + `qwen3_xml` reliably runs 130K+ tokens without interruption. 3.6 has a noticeably higher failure rate even with the best config.

---

## New Problems Specific to Qwen3.6-35B-A3B

### 1. More Loopy

The model gets stuck in reasoning loops more often. It'll loop through the same analysis step multiple times, consuming tokens, before eventually moving forward. This isn't a template issue — it's a model behavior change. On 3.5 this happened occasionally. On 3.6 it's frequent enough to meaningfully impact long sessions.

### 2. Malformed Tool Calls Interrupt Agentic Flow

Even with `enhanced.jinja` + `qwen3_xml` (the config that works perfectly on 3.5), 3.6 has a higher chance of generating malformed tool calls that break the agentic process. The tool calling format still uses XML and is technically correct — but the frequency is higher and the damage is worse: an interrupted session that can't recover.

On 3.5, a malformed tool call is a rare edge case. On 3.6, it's a regular occurrence that will eventually kill a long-running agentic session, no matter which config you use.

### 3. Context Sweep Interaction

When the DCP sweep triggered at 135K tokens (dropping session to 107K), the model recovered and kept going. But the context manipulation seems to interact poorly with the looping behavior — once swamped, it takes longer to regain coherence. This wasn't an issue with 3.5.

---

## The Fix (Partial)

**OpenCode 1.4.18** helps. The older version had tool calling issues that made things worse. Upgrading to 1.4.18 resolved some of the malformed tool call problems.

But here's the honest part: **upgrading the client doesn't solve the looping or the inherently higher failure rate on 3.6**. The issues above are in the model, not the tooling.

One config tweak helped: **context length from 219,520 down to 200,000**. The 3.6 model is larger, so at 219k context it starts forcing offloading, which kills token throughput. At 200K, everything stays in GPU RAM and speed is maintained.

---

## My Config

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_CUMEM_ENABLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_SLEEP_WHEN_IDLE=1

rm -rf ~/.cache/flashinfer

vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen3.6-35B-A3B \
  --chat-template qwen3.5-enhanced.jinja \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 200000 \
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
```

---

## Bottom Line

**My config (enhanced.jinja + qwen3_xml + OpenCode 1.4.18) is still the best you can do on Qwen3.6.** But it's worth being honest: Qwen3.6-35B-A3B is more loopy and has a higher failure rate for agentic tool calling compared to Qwen3.5-27B. The fixes carry over from 3.5, but the model itself is less reliable for long agentic work.

**I've decided to stick with Qwen3.5-27B-FP8.** For agentic obedience — following instructions, executing tool calls cleanly, not looping — the 27B model outperforms the 3.6 35B-A3B. 3.6 has better reasoning, but it pays for it with looping and tool call failures that kill long sessions. Reliability over raw intelligence for agentic work.
