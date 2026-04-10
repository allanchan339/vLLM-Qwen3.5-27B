from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def chat(message, max_tokens=None):
    """与 Qwen3.5 对话，自动处理 reasoning"""
    kwargs = {
        "model": "vllm/Qwen3.5-27B",
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.6,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    result = {
        "content": msg.content,
        "reasoning": getattr(msg, "reasoning", None),
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage,
    }
    return result


# 测试 1：不限制长度（默认最大）
print("=== Test 1: No max_tokens limit ===")
r1 = chat("Hello! Introduce yourself.")
print(f"Answer: {r1['content']}")
print(f"Reasoning chars: {len(r1['reasoning']) if r1['reasoning'] else 0}")
print(f"Total tokens: {r1['usage'].total_tokens}")

# 测试 2：数学推理
print("\n=== Test 2: Math reasoning ===")
r2 = chat("Calculate 125 * 8 step by step.")
print(f"Answer: {r2['content']}")
if r2["reasoning"]:
    print(f"Reasoning preview: {r2['reasoning'][:300]}...")

# 测试 3：Tool calling (with error handling)
print("\n=== Test 3: Tool calling ===")
try:
    response = client.chat.completions.create(
        model="vllm/Qwen3.5-27B",
        messages=[{"role": "user", "content": "What is 25 squared?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
    )
    tool_calls = response.choices[0].message.tool_calls
    print(f"Tool calls: {tool_calls}")
    if tool_calls:
        print(f"Function: {tool_calls[0].function.name}")
        print(f"Arguments: {tool_calls[0].function.arguments}")
except Exception as e:
    print(f"Tool calling failed (parser/format mismatch): {type(e).__name__}: {e}")
    print(
        "Note: Ensure vLLM is launched with --enable-auto-tool-choice and correct --tool-call-parser"
    )
