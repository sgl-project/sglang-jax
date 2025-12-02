"""Verify KV head replication factor in TP environment."""

# Grok-2 configuration
num_kv_heads = 8
tp_size = 32
num_hidden_layers = 64
head_dim = 128

print("=" * 80)
print("KV HEAD REPLICATION ANALYSIS")
print("=" * 80)

print(f"\nModel Configuration:")
print(f"  num_kv_heads: {num_kv_heads}")
print(f"  tp_size: {tp_size}")
print(f"  num_hidden_layers: {num_hidden_layers}")
print(f"  head_dim: {head_dim}")

# Calculate KV heads per device (from get_num_kv_heads_by_tp logic)
if tp_size >= num_kv_heads:
    kv_heads_per_device = 1
    replication_factor = tp_size // num_kv_heads
    print(f"\nKV Head Distribution:")
    print(f"  ✓ tp_size >= num_kv_heads: KV heads will be REPLICATED")
    print(f"  kv_heads_per_device: {kv_heads_per_device}")
    print(f"  replication_factor: {replication_factor}x")
    print(
        f"  → Each of the {num_kv_heads} unique KV heads is replicated {replication_factor} times"
    )
else:
    kv_heads_per_device = (num_kv_heads + tp_size - 1) // tp_size
    print(f"\nKV Head Distribution:")
    print(f"  kv_heads_per_device: {kv_heads_per_device}")
    print(f"  No replication")

# Expected vs actual token counts
expected_tokens_per_req = 8192 + 1024  # input + output
concurrent_reqs = 8
expected_total = expected_tokens_per_req * concurrent_reqs

server_reported = 286848
actual_per_req = server_reported / concurrent_reqs

print(f"\n{'=' * 80}")
print("TOKEN COUNT ANALYSIS")
print("=" * 80)

print(f"\nExpected (from benchmark):")
print(f"  Tokens per request: {expected_tokens_per_req:,}")
print(f"  Concurrent requests: {concurrent_reqs}")
print(f"  Total tokens: {expected_total:,}")

print(f"\nServer reported:")
print(f"  Total tokens: {server_reported:,}")
print(f"  Tokens per request: {actual_per_req:,.0f}")

ratio = server_reported / expected_total
print(f"\nRatio (server / expected): {ratio:.2f}x")

if tp_size >= num_kv_heads:
    print(f"\n{'=' * 80}")
    print("HYPOTHESIS: KV CACHE REPLICATION ISSUE")
    print("=" * 80)

    print(f"\nIf each TP rank tracks its own token count:")
    print(f"  Each rank stores: {actual_per_req:,.0f} tokens per request")
    print(f"  Replication factor: {replication_factor}x")
    print(f"  Expected ratio should be: {replication_factor}x")

    if abs(ratio - replication_factor) < 0.5:
        print(
            f"\n  ⚠️  MATCH! The {ratio:.2f}x ratio matches the {replication_factor}x replication factor!"
        )
        print(f"  ⚠️  This suggests that #token is counting ALL replicated KV caches,")
        print(f"  ⚠️  not just the unique tokens!")

        print(f"\n{'=' * 80}")
        print("ROOT CAUSE IDENTIFIED")
        print("=" * 80)
        print(
            f"""
When tp_size ({tp_size}) > num_kv_heads ({num_kv_heads}):
1. Each KV head is replicated across {replication_factor} TP ranks
2. Each rank allocates its own KV cache for the replicated head
3. The #token metric counts tokens across ALL {tp_size} ranks
4. This results in counting each unique token {replication_factor} times!

SOLUTION:
The token count calculation should divide by the replication factor,
or only count unique KV heads, not replicated ones.
"""
        )
    else:
        print(f"\n  ✗ Ratio {ratio:.2f}x doesn't match replication factor {replication_factor}x")
        print(f"  Need to investigate further...")
else:
    print(f"\nNo KV head replication in this configuration.")

print("=" * 80)
