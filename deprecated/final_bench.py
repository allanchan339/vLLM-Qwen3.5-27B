import time
import asyncio
from openai import AsyncOpenAI

# --------------------------
# CONFIGURATION
# --------------------------
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "vllm/Qwen3.5-27B"
NUM_PARALLEL_REQUESTS = 4  # How many requests to run at once
INPUT_PROMPT_LEN = 1024  # Approx input length
OUTPUT_TOKENS = 256  # Max tokens to generate

# Create a dummy prompt of roughly INPUT_PROMPT_LEN words
DUMMY_PROMPT = "Write a detailed technical analysis about quantum computing. " * (
    INPUT_PROMPT_LEN // 10
)

# Initialize client
client = AsyncOpenAI(base_url=BASE_URL, api_key="dummy_key")


# --------------------------
# BENCHMARK LOGIC
# --------------------------
async def run_single_request(request_id: int):
    try:
        start_time = time.perf_counter()

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": DUMMY_PROMPT}],
            max_tokens=OUTPUT_TOKENS,
            temperature=0.0,
        )

        end_time = time.perf_counter()

        return {
            "id": request_id,
            "success": True,
            "latency": end_time - start_time,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except Exception as e:
        return {"id": request_id, "success": False, "error": str(e)}


async def main():
    print(f"🚀 Starting Benchmark...")
    print(f"• Concurrent Requests: {NUM_PARALLEL_REQUESTS}")
    print(f"• Target Output:       {OUTPUT_TOKENS} tokens/req")
    print(f"• Model:               {MODEL_NAME}")
    print("-" * 50)

    start_time = time.perf_counter()

    # Run all requests in parallel
    tasks = [run_single_request(i) for i in range(NUM_PARALLEL_REQUESTS)]
    results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start_time

    # --------------------------
    # CALCULATE STATS
    # --------------------------
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print("❌ All requests failed!")
        if failed:
            print(f"First error: {failed[0]['error']}")
        return

    total_output_tokens = sum(r["output_tokens"] for r in successful)
    avg_latency = sum(r["latency"] for r in successful) / len(successful)

    # Sort latencies for percentiles
    latencies = sorted([r["latency"] for r in successful])
    p50_latency = latencies[int(len(latencies) * 0.5)]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    p99_latency = latencies[int(len(latencies) * 0.99)]

    # --------------------------
    # PRINT RESULTS
    # --------------------------
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Duration:        {total_duration:.2f}s")
    print(f"Successful:            {len(successful)}/{NUM_PARALLEL_REQUESTS}")
    print(f"Throughput:            {total_output_tokens / total_duration:.2f} tokens/s")
    print(f"Requests Per Second:   {len(successful) / total_duration:.2f} RPS")
    print("-" * 50)
    print(f"Avg Latency:           {avg_latency:.2f}s")
    print(f"P50 Latency:           {p50_latency:.2f}s")
    print(f"P95 Latency:           {p95_latency:.2f}s")
    print(f"P99 Latency:           {p99_latency:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
