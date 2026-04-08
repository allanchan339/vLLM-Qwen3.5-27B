# The Pain of using vLLM with Qwen3.5-27B

## Background  
### Mixed GPU setup
- 1x RTX 4090 (SM89) - 24GB VRAM
- 1x RTX 3090 (SM80) - 24GB VRAM
- Intel I9-13900K - 32GB RAM

### Environment
- windows 11
- 591.86 nvidia driver
- wsl2 ubuntu 22.04

### Goal
1. Run Qwen3.5-27B in this setup.
2. Tool calling is stable.
3. Reasoning is stable.
4. Context Length is as long as possible.
5. Model is stable over long context length.

# Pain 0: Versioning Issue
The vllm v0.19.0 is partial support for transformers 5. Due to some bugs, the vLLM is still using transformers 4.49 

However, new model e.g. Qwen3.5-27B require new RoPE implementation, making transformers 5.3 or above is a must. 

You have to manually install transformers 5.3 or above, after installed vllm v0.19.0. e.g. 
```
uv pip install -U transformers
```

# Pain 1: Quantization vs Accuracy
When having 48GB VRAM in total, FP8 quantization is a good choice as the accuracy loss is negligible, comparing to FP16 official model.

However, this model is >= 27GB in size, it must be run in multi-GPU setup.

## FP8 quantization
The FP8 quantization is quite stable for this model. However, only the RTX 4090 is supported with native FP8 W8A8 matrix multiplication. For RTX 3090, it will fallback to W8A16 calculation, this difference will accumlate across layers, causing the model to generate incorrect results and non-consistent results. 

## AWQ quantization
The AWQ quantization is always having higher plexitiy than FP8 quantization, as FP8 or NVFP4 is not uniform in the range of 0-1. AWQ quantization is uniform in the range of 0-1, making the model to generate more consistent results. The outlier issue is also worse the AWQ quantization.

Polarquant or paraquant is a better choice. However, it is not officially supported by vLLM, making TP or PP mode is not possible. 

# Pain 2: Mixed GPU Setup
## TP mode
TP mode is handy for multi-GPU setup, it is the most popular mode for multi-GPU setup. It will automatically split the model into half (matrix multiplication) for each GPU. It save the VRAM usage for each GPU with a little overhead. 

Originally this mode is designed for single series of GPU, where the matrix multiplication between two GPUs is the same.

However, for mixed GPU setup (SM80 + SM89), the matrix multiplication between two GPUs is a problem, as marlin (the matrix multiplication kernel) may provide small difference in the result. Next, two results will be mixed together (all-reduce) to get the final result. The error will accumulate over the context length and layer propagation, causing the model to generate incorrect results. (garbage in, garbage out)

## PP mode
As said in https://github.com/vllm-project/vllm/issues/34437, for mixed GPU setup, TP mode is not recommended.

Therefore, PP mode is the only choice for mixed GPU setup. It will split the model into first half and second half for each GPU. It will not have the matrix multiplication issue. However, it will consume double the VRAM usage on kv cache. It is quite a big overhead as the current vLLM is not optimized for PP mode, making a large context length is consumed (instead of card A keep first half, card B keep second half). 

Eventually it reduce the context length from 220k to 100k to make sure the model can run. 

# Pain 3: Tool calling (qwen3-coder, qwen3-xml vs Hermes)

Tool calling parser is a critical component for tool calling. It will parse the tool call result from the model response. With two common formats: JSON and XML.

The tool calling is not stable for tool calling parser, it will cause the model unable to generate correct tool call result. Especially when offical model tend to call tool calling in the middle of thinking block (without </thinking> tag). It is actually a disiliation error from teacher model, causing the model to generate incorrect tool call result.

The solution 1 is to use qwen3-xml instead of qwen3-coder to ensure the tool calling is always xml format that avoid failed tool calling as <stop>, causing the model to stop generating result when it is not supposed to. However, this fix is not perfect, as the model may still generate incorrect tool call result.

The solution 2 is to replace the chat template to ensure the tool calling is always xml format that avoid failed tool calling as <stop>, causing the model to stop generating result when it is not supposed to. However, this fix is not perfect either, as the model may still generate incorrect tool call result.

# Pain 4: Small model stability over long context length
The solution 3 is more complicated, it rely on post-training to fix this issue. Luckily, we can find huggingface model that is distilled on Claude 4.6 Opus Reasoning model, by learning teacher model's CoT and reasoning result. Model can be more stable over long context length, reasoning, and tool calling. Model can be found in https://huggingface.co/Jackrong/Qwopus3.5-27B-v3

# Conclusion
1. Choose distilled model Qwopus3.5-27B-v3 instead of official model Qwen3.5-27B. This model is more stable over long context length, reasoning, and tool calling.

2. Use hermes as tool calling parser instead of qwen3-coder or qwen3-xml. After distillation, the model should align more to teacher model's behavior (which use hermes as tool calling parser).

3. Use AWQ quantization or other INT style quantization instead of FP8 quantization. This will make the model to generate more consistent results in mixed GPU setup. (Both GPU has native support on INT4 or INT8 calculation). Also, it save the VRAM usage for each GPU.

4. Use PP mode instead of TP mode in mixed GPU setup. This is to ensure the model accuracy is not affected by the matrix multiplication issue.

5. Use vllm v0.19.0 with transformers v5.5.