"""Check actual token count in benchmark requests."""

import sys

sys.path.insert(0, "/Users/ramezes/job/sgl-project/sgl-jax/python")

import numpy as np

from sgl_jax.srt.hf_transformers_utils import get_tokenizer

# Initialize tokenizer
tokenizer_path = "/models/xai-grok-2/tokenizer.tok.json"
tokenizer = get_tokenizer(tokenizer_path)

print("=" * 60)
print("Analyzing Actual Token Counts")
print("=" * 60)

# Benchmark parameters
num_prompts = 16
random_input_len = 8192
random_output_len = 1024
random_range_ratio = 1.0

# Generate random input lengths (same logic as bench_serving.py)
input_lens = np.random.randint(
    max(int(random_input_len * random_range_ratio), 1),
    random_input_len + 1,
    size=num_prompts,
)
output_lens = np.random.randint(
    int(random_output_len * random_range_ratio),
    random_output_len + 1,
    size=num_prompts,
)

print(f"\nGenerated input lengths (first 8):")
for i in range(min(8, len(input_lens))):
    print(
        f"  Request {i}: input={input_lens[i]}, output={output_lens[i]}, total={input_lens[i] + output_lens[i]}"
    )

total_input_tokens = sum(input_lens)
total_output_tokens = sum(output_lens)
total_tokens = total_input_tokens + total_output_tokens

print(f"\nTotal across all {num_prompts} requests:")
print(f"  Total input tokens: {total_input_tokens:,}")
print(f"  Total output tokens: {total_output_tokens:,}")
print(f"  Grand total: {total_tokens:,}")

# With page_size=128 alignment
page_size = 128
print(f"\n{'=' * 60}")
print(f"With PAGE_SIZE={page_size} alignment:")
print(f"{'=' * 60}")

aligned_tokens_per_req = []
for i in range(min(8, len(input_lens))):
    req_total = input_lens[i] + output_lens[i]
    pages_needed = (req_total + page_size - 1) // page_size
    aligned_tokens = pages_needed * page_size
    aligned_tokens_per_req.append(aligned_tokens)
    print(f"  Request {i}:")
    print(f"    Actual tokens: {req_total}")
    print(f"    Pages needed: {pages_needed}")
    print(f"    Aligned tokens: {aligned_tokens} (+{aligned_tokens - req_total} overhead)")

# For 8 concurrent requests
print(f"\n{'=' * 60}")
print(f"Concurrent execution (8 requests):")
print(f"{'=' * 60}")
concurrent_aligned = sum(aligned_tokens_per_req)
print(f"  Total aligned tokens for 8 concurrent requests: {concurrent_aligned:,}")

# Server reported value
server_reported = 286848
print(f"\nServer reported: {server_reported:,} tokens")
print(f"Ratio: {server_reported / concurrent_aligned:.2f}x")
print(f"Difference: {server_reported - concurrent_aligned:,} tokens")

# Check if this might be context_len related
avg_per_req = server_reported / 8
print(f"\nAverage tokens per request (server): {avg_per_req:,.0f}")
print(f"This suggests each request might be using ~{avg_per_req:,.0f} tokens")

# Possible page-aligned context_len
possible_context_lens = [32768, 36864, 40960, 49152]
print(f"\nChecking common page-aligned context lengths:")
for ctx_len in possible_context_lens:
    pages = ctx_len // page_size
    actual = pages * page_size
    total_8_req = actual * 8
    print(
        f"  context_len={ctx_len:,} -> {pages} pages -> {actual:,} tokens/req -> {total_8_req:,} total"
    )
    if abs(total_8_req - server_reported) < 10000:
        print(f"    ^^^ MATCH! This might be it!")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)
print("1. Check if requests are pre-allocating a fixed context length")
print("2. Verify the actual input token count in server logs")
print("3. Look for prefill logs showing '#new-token' count")
