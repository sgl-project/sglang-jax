"""Plan 4 SWA diagnostic (TPU-only): bisect flash-vs-naive at the RAW attention level.

Task A's flash==naive failed (88.7% mismatch). The flash kernel's sliding-window
convention was verified to MATCH naive in source (ragged_paged_attention_v3.py:499,510:
causal q>=k AND q-k<W == naive predicate), so this is NOT a window off-by-one.

This test isolates the ATTENTION compute (before gate/o_proj/MLP) for:
  (a) layer 0 — full attention (no window)
  (b) layer 1 — sliding attention (window=16, T=24)
and prints per-query-position rel-err so we can bisect:
  - BOTH (a) and (b) mismatch  -> KV-cache / harness setup is broken (flash reads wrong K/V).
  - only (b) mismatches        -> window-specific (paged SWA path / swa_page_indices).
  - all positions wrong        -> KV read wrong; only window-boundary positions wrong -> window.

Run on TPU from python/ (paste the printed per-position tables back)::

    python -m pytest sgl_jax/test/models/test_step3p5_swa_diagnostic.py -x -v -s -o "pythonpath=."
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Reuse the proven harness helpers from the flash==naive test.
from sgl_jax.test.models.test_step3p5_flash_vs_naive import (
    _HEAD_DIM,
    _NUM_TOKENS,
    _attach_flash_backend,
    _build_checkpoint,
    _load_weights,
    _make_config,
    _make_forward_batch,
    _make_kv_pool,
)

_IS_TPU = jax.devices()[0].platform == "tpu"


def _per_position_report(name, flash_out, naive_out):
    """Print per-query-position max-abs-diff and rel-err; return (n_bad, max_rel)."""
    f = np.asarray(flash_out, dtype=np.float64).reshape(flash_out.shape[0], -1)
    n = np.asarray(naive_out, dtype=np.float64).reshape(naive_out.shape[0], -1)
    print(f"\n=== {name}: raw attention output, per query position ===")
    print(" pos | max_abs_diff | rel_err  | verdict")
    n_bad = 0
    max_rel = 0.0
    for t in range(f.shape[0]):
        d = np.max(np.abs(f[t] - n[t]))
        denom = np.max(np.abs(n[t])) + 1e-9
        rel = d / denom
        max_rel = max(max_rel, rel)
        bad = rel > 1e-2
        n_bad += int(bad)
        print(f" {t:>3} | {d:>11.4e} | {rel:>7.4f} | {'BAD' if bad else 'ok'}")
    print(f" -> {n_bad}/{f.shape[0]} positions bad, max_rel={max_rel:.4f}")
    return n_bad, max_rel


@unittest.skipUnless(_IS_TPU, "flash path requires RadixAttention (TPU kernel) — skip on CPU")
class TestSWADiagnostic(unittest.TestCase):
    """Compare flash RadixAttention vs naive at the raw-attention level (no gate/o_proj/MLP)."""

    @classmethod
    def setUpClass(cls):
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        cls._cfg = _make_config()
        cls._mesh = create_device_mesh(
            ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
        )
        jax.sharding.set_mesh(cls._mesh)
        cls._weights = _build_checkpoint(cls._cfg)

    def _attn_module(self, layer_id, attn_impl="flash"):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(
                self._cfg, mesh=self._mesh, dtype=jnp.float32, attn_impl=attn_impl
            )
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model.model.layers[layer_id].self_attn

    def _flash_vs_naive_attn(self, layer_id):
        """Run identical post-rope q/k/v through flash and naive; return (flash_out, naive_out)."""
        attn = self._attn_module(layer_id)
        rng = np.random.default_rng(layer_id + 1)
        hidden = jnp.asarray(rng.standard_normal((_NUM_TOKENS, self._cfg.hidden_size)), jnp.float32)
        positions = jnp.arange(_NUM_TOKENS, dtype=jnp.int32)

        with jax.set_mesh(self._mesh):
            # Replicate Step3p5Attention pre-attention steps (shared by both paths).
            q, _ = attn.q_proj(hidden)
            k, _ = attn.k_proj(hidden)
            v, _ = attn.v_proj(hidden)
            q = q.reshape(-1, attn.num_heads, _HEAD_DIM)
            k = k.reshape(-1, attn.num_kv_heads, _HEAD_DIM)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q, k = attn.rotary_emb(positions, q, k)
            v = v.reshape(-1, attn.num_kv_heads, _HEAD_DIM)

            # naive (CPU-validated oracle): direct eager attention.
            naive_out = attn._naive_attention(q, k, v)

            # flash: RadixAttention with backend + KV pool wired.
            kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
            fb = _make_forward_batch(_NUM_TOKENS)
            _attach_flash_backend(fb, self._cfg, self._mesh, kv_pool.page_size)
            flash_out, _ = attn.attn(q, k, v, fb, kv_pool)

        return np.asarray(flash_out), np.asarray(naive_out)

    def test_full_layer0_attn(self):
        """Layer 0 (full attention, no window) — if this mismatches, it's KV setup not window."""
        flash_out, naive_out = self._flash_vs_naive_attn(0)
        n_bad, max_rel = _per_position_report("layer0 FULL-attn", flash_out, naive_out)
        self.assertEqual(
            n_bad, 0, f"full-attn flash != naive ({n_bad} positions). KV-cache setup, not window."
        )

    def test_sliding_layer1_attn(self):
        """Layer 1 (sliding, W=16, T=24) — window-specific check (kernel conv == naive, verified)."""
        flash_out, naive_out = self._flash_vs_naive_attn(1)
        n_bad, max_rel = _per_position_report("layer1 SLIDING-attn", flash_out, naive_out)
        self.assertEqual(n_bad, 0, f"sliding flash != naive ({n_bad} positions).")

    def test_attn_module_call_layer1(self):
        """Full Step3p5Attention.__call__ (raw attn → gate → o_proj) flash vs naive, layer 1.

        Bisects what's AFTER raw attn: raw attn already matches (test above), so if THIS
        mismatches the divergence is in the head-wise gate or o_proj path; if it matches,
        the divergence is downstream in the decoder layer (residual/MLP) or the
        flash_vs_naive harness itself.
        """
        flash_attn = self._attn_module(1, "flash")
        naive_attn = self._attn_module(1, "naive")
        rng = np.random.default_rng(99)
        hidden = jnp.asarray(rng.standard_normal((_NUM_TOKENS, self._cfg.hidden_size)), jnp.float32)
        positions = jnp.arange(_NUM_TOKENS, dtype=jnp.int32)

        with jax.set_mesh(self._mesh):
            naive_out, _ = naive_attn(positions, hidden, _make_forward_batch(_NUM_TOKENS), None)
            kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
            fb = _make_forward_batch(_NUM_TOKENS)
            _attach_flash_backend(fb, self._cfg, self._mesh, kv_pool.page_size)
            flash_out, _ = flash_attn(positions, hidden, fb, kv_pool)

        n_bad, max_rel = _per_position_report(
            "layer1 full __call__ (gate+o_proj)", np.asarray(flash_out), np.asarray(naive_out)
        )
        self.assertEqual(n_bad, 0, f"full attn-module __call__ flash != naive ({n_bad} positions).")

    def _model(self, attn_impl):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            m = Step3p5ForCausalLM(
                self._cfg, mesh=self._mesh, dtype=jnp.float32, attn_impl=attn_impl
            )
        _load_weights(m, self._weights, self._cfg, self._mesh)
        return m

    def _run_full(self, model, kv_pool):
        """Full ForCausalLM forward; return the raw result tuple (incl layers_topk_ids)."""
        from sgl_jax.srt.layers.logits_processor import LogitsMetadata
        from sgl_jax.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        fb = _make_forward_batch(_NUM_TOKENS)
        if kv_pool is not None:
            _attach_flash_backend(fb, model.config, self._mesh, kv_pool.page_size)
        lm = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            logits_indices=jnp.array([_NUM_TOKENS - 1], dtype=jnp.int32),
            extend_seq_lens=jnp.array([_NUM_TOKENS], dtype=jnp.int32),
            extend_seq_lens_cpu=[_NUM_TOKENS],
        )

        class _MP:
            token_to_kv_pool = kv_pool

        with jax.set_mesh(self._mesh):
            return model(fb, _MP(), lm)

    def test_decoder_layer1_isolated(self):
        """Full DecoderLayer 1 (sliding + DENSE) isolated: attn-module + residual + dense MLP.

        Layer 1 has NO MoE (moe_layers_enum=2,3,4), so a mismatch here is residual / dense-MLP
        / harness — NOT a top-k flip. Isolates the dense-layer 88.7% from the 2-layer
        sequential run in flash_vs_naive (whether layer-0 propagation or this layer's wrapper).
        """
        fm = self._model("flash")
        nm = self._model("naive")
        rng = np.random.default_rng(123)
        hidden = jnp.asarray(rng.standard_normal((_NUM_TOKENS, self._cfg.hidden_size)), jnp.float32)
        pos = jnp.arange(_NUM_TOKENS, dtype=jnp.int32)
        with jax.set_mesh(self._mesh):
            nh, _, _, _ = nm.model.layers[1](
                pos, hidden, _make_forward_batch(_NUM_TOKENS), None, None
            )
            kv = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
            fb = _make_forward_batch(_NUM_TOKENS)
            _attach_flash_backend(fb, self._cfg, self._mesh, kv.page_size)
            fh, _, _, _ = fm.model.layers[1](pos, hidden, fb, kv, None)
        n_bad, _ = _per_position_report(
            "layer1 full DecoderLayer (dense)", np.asarray(fh), np.asarray(nh)
        )
        self.assertEqual(n_bad, 0, f"dense DecoderLayer 1 flash != naive ({n_bad} positions).")

    def test_moe_topk_selection(self):
        """Compare flash vs naive MoE expert SELECTION (topk_ids) at MoE layers 2,3,4.

        Decisive for the logits 88.7%: if selections DIFFER, the continuous attention diff
        (abs 0.5-1.0) flipped borderline top-k (notes ⑩ continuous→discrete) — flash/naive
        both numerically correct but route to different experts. If selections MATCH, top-k
        flip is refuted and the divergence is elsewhere.
        """
        fm = self._model("flash")
        nm = self._model("naive")
        kv = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
        n_res = self._run_full(nm, None)
        f_res = self._run_full(fm, kv)
        n_topk, f_topk = n_res[3], f_res[3]

        print("\n=== MoE topk_ids flash vs naive (per layer) ===")
        total_flipped = 0
        for layer in (2, 3, 4):
            nt = np.asarray(n_topk[layer])
            ft = np.asarray(f_topk[layer])
            # per-token set difference (sorted, since order within top-k is irrelevant)
            diff_tokens = int(np.sum(np.any(np.sort(nt, -1) != np.sort(ft, -1), axis=-1)))
            total_flipped += diff_tokens
            print(f" layer {layer}: {diff_tokens}/{nt.shape[0]} tokens select different experts")
        print(f" -> total {total_flipped} flipped-selection tokens across MoE layers")
        # Not an assertion on correctness: this is diagnostic. If >0, top-k flip confirmed;
        # the fix is decision-tolerant comparison (notes P6), not a real model bug.
        print(" (>0 => top-k flip is the logits-divergence mechanism; 0 => look elsewhere)")


if __name__ == "__main__":
    unittest.main()
