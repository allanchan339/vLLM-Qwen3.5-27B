# Deprecated Files

This folder contains scripts and templates that are no longer recommended for use.

## Why Deprecated?

### Jinja Templates

#### `qwen3.5_official.jinja`
- **Issue**: Has edge cases that break Qwen3.5-27B (but not 122B+)
- **Symptoms**: Tool calls mid-thought, premature stops, reasoning leakage
- **Use instead**: `../qwen3.5-enhanced.jinja`

#### `qwen3.5-barubary-attuned-chat-template.jinja`
- **Issue**: Older template, superseded by enhanced version
- **Use instead**: `../qwen3.5-enhanced.jinja`

#### `qwen3.5-barubary-attuned-chat-template_hermes.jinja`
- **Issue**: Hermes format for SFT-distilled models (Qwopus3.5 series)
- **Problem**: Qwopus3.5 has format drift after 65K tokens
- **Use instead**: `../qwen3.5-enhanced.jinja` with official Qwen3.5-27B-FP8

### Start Scripts

#### Qwopus3.5 Series (SFT-Distilled from Claude)
- `start_vllm_AWQ_Claude.sh`
- `start_vllm_AWQ_Claude_TP.sh`
- `start_vllm_AWQ_Claude_single.sh`
- `start_vllm_FP8_Claude.sh`

**Why deprecated**: These use Qwopus3.5 models which are SFT-distilled from Claude 4.6 Opus
- **Issue**: SFT shifted tool calling format from `qwen3_xml` → `hermes` (JSON)
- **Symptom**: Works for first ~65K tokens, then **mixes XML and JSON formats**
- **Root cause**: SFT doesn't maintain format consistency like base model fine-tuning
- **Use instead**: `../start_vllm_FP8.sh` (official Qwen3.5-27B-FP8)

#### Other Quantization Methods
- `start_vllm_GPTQ_Deckard.sh` - 40B model, too large for 48GB VRAM
- `start_vllm_Polar_Claude.sh` - PolarQuant not officially supported by vLLM

**Use instead**: `../start_vllm_FP8.sh` or `../start_vllm_autoround.sh`

## Recommended Setup

### For 48GB VRAM (Best Quality)
```bash
../start_vllm_FP8.sh
```
- Model: `Qwen/Qwen3.5-27B-FP8`
- Template: `../qwen3.5-enhanced.jinja`
- Context: 219k tokens
- Accuracy: Near-lossless

### For Limited VRAM (<48GB)
```bash
../start_vllm_autoround.sh
```
- Model: `Intel/Qwen3.5-27B-int4-AutoRound`
- Template: `../qwen3.5-enhanced.jinja`
- Context: 219k tokens
- Tradeoff: Higher perplexity than FP8 (INT4 accuracy loss)

## See Also

- Main README: `../README.md`
- Agent Guidelines: `../AGENT.md`
