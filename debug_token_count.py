"""Debug script to understand token counting in sgl-jax server."""

# Based on the benchmark parameters:
# --num-prompts 16
# --random-input 8192
# --random-output 1024
# --max-concurrency 8

# Expected token count calculation:
num_prompts = 16
input_len = 8192
output_len = 1024
max_concurrency = 8

print("=" * 60)
print("Expected Token Count Analysis")
print("=" * 60)

# Calculation 1: If counting only currently running requests
running_reqs = 8  # From log: #running-req: 8
tokens_per_req_input = input_len
tokens_per_req_total = input_len + output_len  # Assuming all generated

expected_tokens_input_only = running_reqs * tokens_per_req_input
expected_tokens_full = running_reqs * tokens_per_req_total

print(f"\nScenario 1: Only counting input tokens")
print(f"  Running requests: {running_reqs}")
print(f"  Input tokens per request: {tokens_per_req_input}")
print(f"  Expected total: {expected_tokens_input_only:,} tokens")

print(f"\nScenario 2: Counting input + output tokens (if all generated)")
print(f"  Running requests: {running_reqs}")
print(f"  Total tokens per request: {tokens_per_req_total}")
print(f"  Expected total: {expected_tokens_full:,} tokens")

# Actual server report
actual_tokens = 286848

print(f"\nActual server report: {actual_tokens:,} tokens")
print(f"\nRatio analysis:")
print(f"  Actual / Expected (input only): {actual_tokens / expected_tokens_input_only:.2f}x")
print(f"  Actual / Expected (input + output): {actual_tokens / expected_tokens_full:.2f}x")

# Average tokens per request based on actual count
avg_tokens_per_req = actual_tokens / running_reqs
print(f"\nAverage tokens per running request: {avg_tokens_per_req:,.0f}")

# Check if this matches context_len or some other value
print(f"\nPossible explanations:")
print(f"1. If each request pre-allocated max_context_len space:")
print(f"   - Implied context_len: ~{avg_tokens_per_req:,.0f} tokens per request")

print(f"\n2. If using paged allocation with large page_size:")
print(f"   - Each request might be allocated in page-sized chunks")

print(f"\n3. If there's a multiplier in the token counting:")
print(f"   - Check if token count includes multiple layers or other factors")

# Grok-2 model info (from HuggingFace)
grok_num_layers = 64
grok_num_kv_heads = 8
grok_head_dim = 128

print(f"\nGrok-2 Model Configuration:")
print(f"  num_hidden_layers: {grok_num_layers}")
print(f"  num_key_value_heads: {grok_num_kv_heads}")
print(f"  head_dim: {grok_head_dim}")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)
print("Check the following in your server logs/code:")
print("1. What is the max_context_len for each request?")
print("2. What is the page_size setting?")
print("3. Are requests pre-allocating their maximum length?")
print("4. Check scheduler.py:_get_token_info() for the actual calculation")
