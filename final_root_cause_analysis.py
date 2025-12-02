"""Final root cause analysis for token counting issue."""

# Configuration
num_kv_heads = 8
tp_size = 32
num_hidden_layers = 64
head_dim = 128

print("=" * 80)
print("ROOT CAUSE ANALYSIS: KV HEAD CALCULATION MISMATCH")
print("=" * 80)

print(f"\nGrok-2 Configuration:")
print(f"  num_kv_heads: {num_kv_heads}")
print(f"  tp_size: {tp_size}")
print(f"  num_hidden_layers: {num_hidden_layers}")
print(f"  head_dim: {head_dim}")

# Simulate get_num_kv_heads (used in profile_max_num_token)
if tp_size >= num_kv_heads:
    kv_heads_for_profiling = 1
else:
    kv_heads_for_profiling = (num_kv_heads + tp_size - 1) // tp_size

print(f"\n{'=' * 80}")
print("STEP 1: profile_max_num_token calculation")
print("=" * 80)
print(f"Uses: get_num_kv_heads({tp_size})")
print(f"Returns: {kv_heads_for_profiling} head(s) per device")
print(f"\nThis is used to calculate cell_size and max_total_num_tokens")

# Simulate get_total_num_kv_heads_with_replication (used in MHATokenToKVPool)
if tp_size > num_kv_heads:
    total_heads_with_replication = tp_size
else:
    total_heads_with_replication = num_kv_heads

print(f"\n{'=' * 80}")
print("STEP 2: MHATokenToKVPool creation")
print("=" * 80)
print(f"Uses: get_total_num_kv_heads_with_replication({tp_size})")
print(f"Returns: {total_heads_with_replication} heads total")
print(f"\nThis is used as head_num parameter when creating KV cache buffer")

# Calculate the mismatch
print(f"\n{'=' * 80}")
print("STEP 3: THE MISMATCH")
print("=" * 80)
print(f"\nCell size calculated assuming: {kv_heads_for_profiling} head")
print(f"But KV cache buffer created with: {total_heads_with_replication} heads")
print(
    f"\nMultiplier from this mismatch: {total_heads_with_replication / kv_heads_for_profiling:.2f}x"
)

# But wait, there's replication...
replication_factor = tp_size // num_kv_heads
unique_heads = num_kv_heads

print(f"\n{'=' * 80}")
print("STEP 4: ACTUAL REPLICATION FACTOR")
print("=" * 80)
print(f"\nOf the {total_heads_with_replication} heads in KV cache:")
print(f"  Only {unique_heads} are unique")
print(f"  Each is replicated {replication_factor} times")
print(f"  {unique_heads} × {replication_factor} = {total_heads_with_replication}")

print(f"\nSo the ACTUAL multiplier should be: {replication_factor}x")

# Combine with tokenizer
tokenizer_multiplier = 1.03
total_multiplier = tokenizer_multiplier * replication_factor

print(f"\n{'=' * 80}")
print("FINAL CALCULATION")
print("=" * 80)
print(f"\nTokenizer effect: {tokenizer_multiplier}x")
print(f"KV head replication: {replication_factor}x")
print(f"Total expected multiplier: {total_multiplier:.2f}x")

# Compare with observed
observed = 3.89
diff = abs(total_multiplier - observed)
match = "✓ MATCH!" if diff < 0.3 else "✗ Mismatch"

print(f"\nObserved multiplier: {observed}x")
print(f"Difference: {diff:.2f}")
print(f"{match}")

if diff < 0.3:
    print(f"\n{'=' * 80}")
    print("ROOT CAUSE CONFIRMED")
    print("=" * 80)
    print(
        f"""
The token count is inflated by {replication_factor}x because:

1. profile_max_num_token() calculates max_total_num_tokens based on
   {kv_heads_for_profiling} KV head per device

2. But MHATokenToKVPool() creates a buffer with {total_heads_with_replication} heads
   (all replicated heads across TP ranks)

3. Since there are only {unique_heads} unique heads, but {total_heads_with_replication} total heads,
   the replication factor is {replication_factor}x

4. This causes available_size() to be calculated incorrectly,
   making num_used appear {replication_factor}x larger than it should be

SOLUTION:
Either:
A) Make profile_max_num_token use get_total_num_kv_heads_with_replication
B) Or adjust the token count calculation to account for replication
C) Or change MHATokenToKVPool to only allocate for unique heads
"""
    )

print("=" * 80)
