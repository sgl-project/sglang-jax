"""Offline oracle: a single GDN forward == chained 128-token split (the extra-buffer
boundary-split schedule), pure JAX on CPU -- the TPU-free check that splitting a
prefill does not change model numerics. Pool-table dtypes mirror production
(conv bf16, recurrent fp32): the carry dtype is set purely by the initial tables."""

from __future__ import annotations

import os
import unittest

# Pure-JAX kernels run eager on one CPU device. Pin CPU before JAX initializes
# (mirrors test_recurrent_track_scatter's preamble) but DO NOT set a module
# mesh — these ops need none and a module mesh would pollute sibling files.
if os.environ.get("USE_DEVICE_TYPE") == "cpu" or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gdn.gated_delta import (
    jax_causal_conv1d_prefill,
    ragged_gated_delta_rule_ref,
)

# Small CPU shapes (mirror test_gdn_attention). conv_dim = 2*n_kq*d_k + n_v*d_v.
N_KQ, N_V, D_K, D_V, K = 4, 8, 32, 32, 4
CONV_DIM = 2 * N_KQ * D_K + N_V * D_V  # 512
NUM_BLOCKS = 4  # slot 0 = dummy, requests use slot 1
T_MAX = 384

# slot the single + split runs both target (slot 0 is the dummy).
SLOT = 1

# (length, chunk sizes) — model GPQA prompts crossing 128 boundaries. Two cases
# cover the distinct splits: one clean boundary, and multi-boundary with a
# partial final chunk. A multi-boundary all-clean case (e.g. 384) adds nothing:
# the chunk-to-chunk carry is identical to the partial case minus the tail.
PATTERNS = [
    (256, [128, 128]),  # one boundary
    (300, [128, 128, 44]),  # two boundaries + partial final chunk (realistic)
]


def _make_inputs(seed: int = 0):
    """Generate the shared layer weights + input tokens ONCE (float32).

    Same rationale as test_gdn_attention's ``_scaled_randn``: scale 0.1 for the
    activations/gates keeps the recurrent state bounded so bf16 noise stays
    inside the global atol; scale 1.0 for weight / A_log / dt_bias.
    """
    rng = np.random.default_rng(seed)

    def scaled(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "mixed_raw": scaled((T_MAX, CONV_DIM), 0.1),
        "b": scaled((T_MAX, N_V), 0.1),
        "a": scaled((T_MAX, N_V), 0.1),
        "weight": scaled((CONV_DIM, K), 1.0),
        "A_log": scaled((N_V,), 1.0),
        "dt_bias": scaled((N_V,), 1.0),
    }


def _run_chunks(inputs, chunk_sizes, *, conv_dtype, rec_dtype, io_dtype):
    """Run the GDN forward as a chain of chunks, carrying the FULL pool tables.

    The pool-table dtypes (conv_dtype / rec_dtype) drive the production
    round-trip automatically (kernels cast on write-back; ragged reads the
    recurrent table back as float32). io_dtype casts mixed_raw + weight.

    Returns float32 numpy ``(output, final_rec, final_conv)``.
    """
    weight = jnp.asarray(inputs["weight"], dtype=io_dtype)
    A_log = jnp.asarray(inputs["A_log"], dtype=jnp.float32)
    dt_bias = jnp.asarray(inputs["dt_bias"], dtype=jnp.float32)
    mixed_raw = jnp.asarray(inputs["mixed_raw"], dtype=io_dtype)
    b = jnp.asarray(inputs["b"], dtype=jnp.float32)
    a = jnp.asarray(inputs["a"], dtype=jnp.float32)

    conv_table = jnp.zeros((NUM_BLOCKS, CONV_DIM, K - 1), dtype=conv_dtype)
    rec_table = jnp.zeros((NUM_BLOCKS, N_V, D_K, D_V), dtype=rec_dtype)

    state_idx = jnp.array([SLOT], dtype=jnp.int32)
    outs = []
    start = 0
    for ci, csz in enumerate(chunk_sizes):
        end = start + csz
        T = end - start
        cu = jnp.array([0, T], dtype=jnp.int32)
        # First chunk is a brand-new prefill (no prior state); later chunks
        # continue from the carried tables.
        has_init = jnp.array([ci != 0], dtype=bool)

        y, conv_table = jax_causal_conv1d_prefill(
            mixed_raw[start:end].T,
            weight,
            None,
            cu,
            conv_table,
            state_idx,
            has_init,
            "silu",
        )
        rec_table, out = ragged_gated_delta_rule_ref(
            y.T,
            b[start:end],
            a[start:end],
            rec_table,
            A_log,
            dt_bias,
            cu,
            state_idx,
            has_init,
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
        )
        outs.append(out)
        start = end

    output = jnp.concatenate(outs, axis=0)
    final_conv = conv_table[SLOT]
    final_rec = rec_table[SLOT]
    return (
        np.asarray(output, dtype=np.float32),
        np.asarray(final_rec, dtype=np.float32),
        np.asarray(final_conv, dtype=np.float32),
    )


def _metrics(single: np.ndarray, split: np.ndarray):
    diff = np.abs(single - split)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = float(np.linalg.norm(single.reshape(-1))) + 1e-12
    rel = float(np.linalg.norm((single - split).reshape(-1)) / denom)
    return max_abs, mean_abs, rel


class _SplitEquivalenceBase(unittest.TestCase):
    """Compare a single forward over N tokens vs the chained 128-token split."""

    # Subclasses set the variant name + the three dtypes + the bit-exact flag.
    VARIANT = ""
    CONV_DTYPE = jnp.float32
    REC_DTYPE = jnp.float32
    IO_DTYPE = jnp.float32
    BITEXACT_REC = False

    RTOL = 2e-2
    ATOL = 1e-2

    @classmethod
    def setUpClass(cls):
        # unittest/pytest collect any TestCase subclass regardless of the
        # leading underscore, so skip the bare base (it would otherwise run a
        # third, mislabeled all-fp32 variant). Mirrors test_retract_decode.py.
        if cls is _SplitEquivalenceBase:
            raise unittest.SkipTest("abstract base; variants run via subclasses")

    def setUp(self):
        self.inputs = _make_inputs(0)

    def _check_pattern(self, length, chunk_sizes):
        kw = dict(
            conv_dtype=self.CONV_DTYPE,
            rec_dtype=self.REC_DTYPE,
            io_dtype=self.IO_DTYPE,
        )
        single = _run_chunks(self.inputs, [length], **kw)
        split = _run_chunks(self.inputs, chunk_sizes, **kw)

        names = ("output", "final_rec", "final_conv")
        print(
            f"\n[{self.VARIANT}] length={length} split={chunk_sizes}",
            flush=True,
        )
        for name, s, sp in zip(names, single, split):
            max_abs, mean_abs, rel = _metrics(s, sp)
            print(
                f"  {name:<10s} max_abs={max_abs:.3e} " f"mean_abs={mean_abs:.3e} rel={rel:.3e}",
                flush=True,
            )

        # Established kernel-equivalence bar (same as test_gdn_attention).
        for name, s, sp in zip(names, single, split):
            np.testing.assert_allclose(
                sp, s, rtol=self.RTOL, atol=self.ATOL, err_msg=f"{name} mismatch"
            )

        # fp32-only: with no bf16 and an fp32 sequential carry, the split must
        # reproduce the single forward bit-exactly for the recurrent final
        # state. Any diff there is a real split-mechanism bug. (conv/output may
        # differ by ULPs from XLA op reordering — covered by assert_allclose.)
        if self.BITEXACT_REC:
            np.testing.assert_array_equal(
                split[1], single[1], err_msg="final_rec not bit-exact (fp32 split)"
            )

    def test_len256_one_boundary(self):
        self._check_pattern(256, [128, 128])

    def test_len300_partial_final(self):
        self._check_pattern(300, [128, 128, 44])


class TestSplitEquivalenceProd(_SplitEquivalenceBase):
    """Faithful production: conv pool bfloat16, recurrent pool float32."""

    VARIANT = "V_prod"
    CONV_DTYPE = jnp.bfloat16
    REC_DTYPE = jnp.float32
    IO_DTYPE = jnp.bfloat16
    BITEXACT_REC = False


class TestSplitEquivalenceFp32(_SplitEquivalenceBase):
    """Isolates the split mechanism: no bf16 anywhere (all float32)."""

    VARIANT = "V_fp32"
    CONV_DTYPE = jnp.float32
    REC_DTYPE = jnp.float32
    IO_DTYPE = jnp.float32
    BITEXACT_REC = True


if __name__ == "__main__":
    unittest.main()
