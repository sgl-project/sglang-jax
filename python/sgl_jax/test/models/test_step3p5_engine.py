"""Plan 4 engine self-consistency skeletons (TPU-only).

These tests require the full serving runtime (ModelRunner / KV pool / RadixCache /
scheduler) on a real TPU with a real or synthetic Step3p5 checkpoint. Each test
carries a detailed docstring specifying the exact assertion so it can be fleshed
out on TPU without re-reading the design document.

All tests skip on CPU with a clear message (no trivial assertions, no fake passes).

Run on TPU from python/ directory::

    python -m pytest sgl_jax/test/models/test_step3p5_engine.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import unittest

import jax

# TPU guard: engine tests require real KV pool + RadixCache which use TPU kernels.
_IS_TPU = jax.devices()[0].platform == "tpu"


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestPrefillDecodeConsistency(unittest.TestCase):
    """prefill==decode: full-prefill logits match incremental-decode logits per position.

    SPEC:
      Setup: load Step3p5ForCausalLM with flash attention on a real/synthetic checkpoint.
             Run full-sequence prefill of [t0, t1, ..., t_{N-1}] → logits_prefill[i].
             Run incremental decode starting from empty context, feeding one token at a
             time, recording logits_decode[i] at each step.
      Assert: logits_prefill[i] ≈ logits_decode[i] for all i, atol=1e-2 (bf16 floor).
              This validates KV cache correctness: cached K/V written during prefill
              must produce identical attention context during decode step i.
    """

    def test_prefill_equals_decode(self):
        self.skipTest(
            "Plan 4: needs real Step3p5 checkpoint + ModelRunner on TPU. "
            "Assert: prefill logits[i] ≈ decode logits[i] ∀i (atol=1e-2, bf16). "
            "See class docstring for full spec."
        )


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestRadixCacheHitVsMiss(unittest.TestCase):
    """cache_hit==miss: RadixCache prefix-hit output matches recompute from scratch.

    SPEC:
      Setup: Prefill a long prompt P of length L. Then run two queries that share
             prefix P but have different suffixes S1, S2.
             Query A: cold start (no cached prefix) — full prefill of P+S1.
             Query B: warm start (RadixCache hit on P) — prefix served from cache, only
                      S2 is newly computed.
             For a fresh query identical to A (P+S1) with RadixCache hit:
      Assert: logits(cache_hit, P+S1) ≈ logits(cache_miss, P+S1), atol=1e-2.
              Validates that RadixCache KV retrieval is bit-consistent with recompute.
    """

    def test_cache_hit_equals_miss(self):
        self.skipTest(
            "Plan 4: needs RadixCache + real checkpoint on TPU. "
            "Assert: prefix-hit logits ≈ recomputed logits (atol=1e-2, bf16). "
            "See class docstring for full spec."
        )


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestChunkedVsFullPrefill(unittest.TestCase):
    """chunked==full: chunked prefill output matches single-pass prefill.

    SPEC:
      Setup: Given input sequence of length N, run:
             (A) single-pass prefill (chunk_size >= N) → logits_full[0..N-1].
             (B) chunked prefill with chunk_size=C < N (two or more chunks) → logits_chunked.
      Assert: logits_full ≈ logits_chunked, atol=1e-2 (bf16 floor).
              Validates that chunked KV accumulation is consistent with monolithic prefill.
              Particularly important for SWA layers where the window boundary may fall
              inside a chunk boundary.
    """

    def test_chunked_equals_full_prefill(self):
        self.skipTest(
            "Plan 4: needs chunked-prefill scheduler + real checkpoint on TPU. "
            "Assert: chunked logits ≈ full logits (atol=1e-2, bf16). "
            "See class docstring for full spec."
        )


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestSWAKVPoolVsFullMHA(unittest.TestCase):
    """Q6 golden: SWAKVPool output == full MHA output with --disable-hybrid-swa-memory.

    SPEC (strongest SWA gate):
      Setup: Run Step3p5ForCausalLM with default config (SWAKVPool for sliding layers,
             full MHATokenToKVPool for full-attention layers).
             Run again with --disable-hybrid-swa-memory (all layers use full MHATokenToKVPool).
      Assert: logits(SWAKVPool) ≈ logits(full MHA), atol=1e-2 (bf16 floor).
              This is the definitive SWA correctness gate: if SWAKVPool's sliding
              eviction logic produces the same logits as the full pool (which retains
              all tokens), the eviction boundary is correct end-to-end.
              Covers both the RadixAttention sliding_window_size boundary AND the
              SWAKVPool eviction logic simultaneously.
    """

    def test_swa_pool_equals_full_mha(self):
        self.skipTest(
            "Plan 4: needs --disable-hybrid-swa-memory flag + real checkpoint on TPU. "
            "Assert: SWAKVPool logits ≈ full-MHA logits (atol=1e-2, bf16). "
            "This is the Q6 golden gate for SWA correctness. "
            "See class docstring for full spec."
        )


if __name__ == "__main__":
    unittest.main()
