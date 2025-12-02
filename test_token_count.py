"""Test script to verify token counts between bench_serving and server.

Usage:
    python test_token_count.py

This script simulates the token generation logic from bench_serving.py
and shows what token counts should be expected on the server side.
"""

import os
import sys

import numpy as np

# Add python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def sample_random_lengths(input_len, output_len, num_prompts, range_ratio, seed=1):
    """
    Same logic as sample_random_requests() in bench_serving.py
    """
    np.random.seed(seed)

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    return input_lens, output_lens


def generate_actual_tokens_with_tokenizer(tokenizer, target_len):
    """
    Generate actual token sequence using tokenizer, matching bench_serving logic.
    """
    # Generate random token IDs (same as bench_serving when not using dataset)
    offset = np.random.randint(0, tokenizer.vocab_size)
    token_ids = [(offset + j) % tokenizer.vocab_size for j in range(target_len)]

    # Decode to text and re-encode (this matches adjust_prompt_decode_to_target_len)
    text = tokenizer.decode(token_ids)
    actual_token_ids = tokenizer.encode(text)

    return text, actual_token_ids, len(actual_token_ids)


def calculate_page_aligned_tokens(token_count, page_size):
    """Calculate page-aligned token count."""
    if page_size == 1:
        return token_count
    pages_needed = (token_count + page_size - 1) // page_size
    return pages_needed * page_size


def main():
    # Benchmark parameters from your command:
    # --num-prompts 16 --random-input 8192 --random-output 1024 --random-range-ratio 1
    num_prompts = 16
    random_input_len = 8192
    random_output_len = 1024
    random_range_ratio = 1.0
    max_concurrency = 8
    seed = 1  # Default seed in bench_serving
    tokenizer_path = "/models/xai-grok-2/tokenizer.tok.json"

    # Server parameter
    page_size = 128  # From your server command: --page-size=128

    print("=" * 80)
    print("TOKEN COUNT VERIFICATION TEST (WITH TOKENIZER)")
    print("=" * 80)
    print(f"\nBenchmark Parameters:")
    print(f"  --num-prompts: {num_prompts}")
    print(f"  --random-input: {random_input_len}")
    print(f"  --random-output: {random_output_len}")
    print(f"  --random-range-ratio: {random_range_ratio}")
    print(f"  --max-concurrency: {max_concurrency}")
    print(f"  --seed: {seed}")
    print(f"  --tokenizer: {tokenizer_path}")
    print(f"\nServer Parameters:")
    print(f"  --page-size: {page_size}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    try:
        from sgl_jax.srt.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(tokenizer_path)
        print(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print(f"  Continuing with theoretical token counts...")
        tokenizer = None

    # Generate token lengths using same logic as bench_serving
    np.random.seed(seed)
    input_lens, output_lens = sample_random_lengths(
        random_input_len, random_output_len, num_prompts, random_range_ratio, seed
    )

    print(f"\n{'=' * 80}")
    print("STEP 1: THEORETICAL TOKEN COUNTS")
    print("=" * 80)
    print(f"(These are the target lengths bench_serving will try to generate)")
    print()
    print(
        f"{'Req':>4} {'Input':>8} {'Output':>8} {'Total':>8} {'Pages':>8} {'Aligned':>10} {'Overhead':>10}"
    )
    print("-" * 80)

    total_input = 0
    total_output = 0
    total_aligned = 0

    for i in range(num_prompts):
        inp = input_lens[i]
        out = output_lens[i]
        total = inp + out
        aligned = calculate_page_aligned_tokens(total, page_size)
        pages = aligned // page_size
        overhead = aligned - total

        total_input += inp
        total_output += out
        total_aligned += aligned

        print(f"{i:4d} {inp:8d} {out:8d} {total:8d} {pages:8d} {aligned:10d} {overhead:10d}")

    print("-" * 80)
    print(
        f"{'SUM':>4} {total_input:8d} {total_output:8d} {total_input+total_output:8d} "
        f"{total_aligned//page_size:8d} {total_aligned:10d} {total_aligned-(total_input+total_output):10d}"
    )

    # Test actual tokenization
    if tokenizer is not None:
        print(f"\n{'=' * 80}")
        print("STEP 2: ACTUAL TOKENIZATION TEST")
        print("=" * 80)
        print(f"(Generate actual token sequences using tokenizer)")
        print()
        print(f"Testing first {min(3, num_prompts)} requests to verify tokenizer behavior:")
        print()

        actual_token_counts = []
        mismatches = []

        for i in range(min(3, num_prompts)):
            target_len = input_lens[i]
            try:
                text, _actual_tokens, actual_len = generate_actual_tokens_with_tokenizer(
                    tokenizer, target_len
                )
                actual_token_counts.append(actual_len)

                status = "✓" if actual_len == target_len else "✗"
                diff = actual_len - target_len

                print(f"  Request {i}:")
                print(f"    Target length: {target_len:,}")
                print(f"    Actual length: {actual_len:,} {status}")
                if diff != 0:
                    print(f"    Difference: {diff:+,} tokens")
                    mismatches.append((i, target_len, actual_len, diff))
                print(f"    Text preview: {text[:100]}...")
                print()

            except Exception as e:
                print(f"  Request {i}: ERROR - {e}")
                print()

        if mismatches:
            print(f"⚠️  WARNING: Token count mismatches detected!")
            print(
                f"   {len(mismatches)} out of {min(3, num_prompts)} requests have mismatched token counts"
            )
            print()
            print(f"   Possible causes:")
            print(f"   1. Tokenizer adds special tokens (BOS, EOS, etc.)")
            print(f"   2. Decode-encode cycle changes token count")
            print(f"   3. Invalid token IDs that get filtered/replaced")
            print()

            # Calculate potential multiplier
            if mismatches:
                avg_ratio = sum(actual / target for _, target, actual, _ in mismatches) / len(
                    mismatches
                )
                print(f"   Average ratio (actual/target): {avg_ratio:.2f}x")
                if abs(avg_ratio - 3.89) < 0.5:
                    print(f"   ⚠️  This {avg_ratio:.2f}x ratio is close to the observed 3.89x!")
                    print(f"   ⚠️  The tokenizer behavior might be the root cause!")
        else:
            print(f"✓ All test requests generated correct token counts")

    print(f"\n{'=' * 80}")
    print("STEP 3: CONCURRENT EXECUTION ANALYSIS")
    print("=" * 80)

    # Simulate decode batch with first 8 requests
    concurrent_input = sum(input_lens[:max_concurrency])
    concurrent_output = sum(output_lens[:max_concurrency])
    concurrent_total = concurrent_input + concurrent_output
    concurrent_aligned = sum(
        calculate_page_aligned_tokens(input_lens[i] + output_lens[i], page_size)
        for i in range(max_concurrency)
    )

    print(f"\nFirst {max_concurrency} requests (during decode):")
    print(f"  Total input tokens:  {concurrent_input:,}")
    print(f"  Total output tokens: {concurrent_output:,}")
    print(f"  Total tokens:        {concurrent_total:,}")
    print(f"  Page-aligned tokens: {concurrent_aligned:,}")
    print(f"  Alignment overhead:  {concurrent_aligned - concurrent_total:,}")

    # Expected values during prefill
    print(f"\n{'=' * 80}")
    print("EXPECTED SERVER LOGS")
    print("=" * 80)

    print(f"\nDuring PREFILL (first 8 requests):")
    print(f"  Expected #new-token:   {concurrent_input:,}")
    print(f"  Expected #token (KV):  {concurrent_aligned:,}  (page-aligned)")

    print(f"\nDuring DECODE (first 8 requests after full generation):")
    # Assuming all output tokens have been generated
    print(f"  Expected #running-req: {max_concurrency}")
    print(f"  Expected #token (KV):  {concurrent_aligned:,}  (page-aligned)")

    # Compare with actual server report
    server_reported = 286848
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH SERVER")
    print("=" * 80)
    print(f"\nServer reported:  {server_reported:,} tokens")
    print(f"Expected aligned: {concurrent_aligned:,} tokens")
    print(f"Difference:       {server_reported - concurrent_aligned:,} tokens")
    print(f"Ratio:            {server_reported / concurrent_aligned:.2f}x")

    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)

    if abs(server_reported - concurrent_aligned) < 1000:
        print("✓ MATCH! Token counts are consistent.")
    else:
        print("✗ MISMATCH! Investigating possible causes...")

        # Check if it's a different phase
        avg_per_req = server_reported / max_concurrency
        print(f"\nAverage tokens per request (server): {avg_per_req:,.0f}")

        # Check if requests allocated more than expected
        print(f"\nPossible explanations:")
        print(f"1. Each request might have allocated more tokens than input+output")
        print(
            f"   Current average: {(concurrent_input + concurrent_output) / max_concurrency:.0f} tokens/req"
        )
        print(f"   Server shows:    {avg_per_req:.0f} tokens/req")
        print(
            f"   Difference:      {avg_per_req - (concurrent_input + concurrent_output) / max_concurrency:.0f} tokens/req"
        )

        # Check common context lengths
        print(f"\n2. If using fixed context length allocation:")
        for ctx_len in [16384, 32768, 36864, 40960]:
            aligned_ctx = calculate_page_aligned_tokens(ctx_len, page_size)
            total_8 = aligned_ctx * 8
            print(f"   context_len={ctx_len:,} -> {total_8:,} total", end="")
            if abs(total_8 - server_reported) < 1000:
                print(f"  ← POSSIBLE MATCH!")
            else:
                print()

        # Check if it's actual input lengths
        print(f"\n3. If actual input lengths are different from expected:")
        implied_avg_input = (server_reported / max_concurrency / page_size) * page_size
        print(f"   Implied avg length per request: ~{implied_avg_input:.0f} tokens")

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(
        """
1. Check server prefill logs for '#new-token' to verify actual input sizes
2. Check if there's a fixed context_len allocation happening
3. Verify random seed is consistent between this test and bench_serving
4. Check if there are additional tokens (e.g., special tokens) being added
"""
    )

    # Save for comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY FOR COMPARISON")
    print("=" * 80)
    print(f"Expected KV cache tokens (8 concurrent): {concurrent_aligned:,}")
    print(f"Server reported tokens:                  {server_reported:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
