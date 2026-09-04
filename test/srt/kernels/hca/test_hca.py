"""Whole-operator HCA tests against the independent numpy reference.

One stateful HCA step is compared end to end with ``oracle.py`` across the
three request shapes the backend dispatches: uniform prefill, decode, and
ragged EXTEND. This is the only numerical coverage the kernels need — a
composite that agrees with dense fp32 math cannot have a wrong compressor,
cache write, page-table resolution, or attention stage hiding inside it.

fp32 dense math and bf16 tiled math cannot agree bit for bit, so the gate is a
tolerance -- ``rtol=2e-2, atol=1e-2``, the same one every bf16 attention kernel
in this repo uses (flash, paged, MLA, GDN, KDA). These shapes clear it using
18-32% of the budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.layers.attention.hca_backend import HCABackend
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from . import oracle
from .common import HCATestFactory

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="production HCA requires TPU"
)
HCA_TEST = HCATestFactory()

HIDDEN, HEADS, HEAD_DIM, RATIO = 4096, 64, 512, 128
SOFTMAX_SCALE = HEAD_DIM**-0.5
# Inputs are scaled so both softmaxes run in their peaked regime (attention
# ~29x above uniform, compressor gating ~9x). With near-uniform inputs both
# degenerate to averaging and the comparison stops being sensitive to the score
# scale at all -- a 50% softmax-scale error stayed inside the gate. Sensitivity
# was then confirmed by injecting defects into the reference: a one-token window
# shift, a dropped attention sink, a skipped compressor RoPE, a +-10% softmax
# scale, a missing compressed record and a neutralised RMSNorm weight overrun
# the tolerance by 2.7x to 123x.
RTOL, ATOL = 2e-2, 1e-2


def _bf16_master(array):
    """Pre-round so the kernels and the reference start from identical values."""
    return np.asarray(array, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _weights(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rope_rows = 8192
    angle = np.arange(rope_rows * 32, dtype=np.float32).reshape(rope_rows, 32) * 1e-4
    return {
        "wkv": _bf16_master(rng.standard_normal((HEAD_DIM, HIDDEN)) * 0.05),
        "wgate": _bf16_master(rng.standard_normal((HEAD_DIM, HIDDEN)) * 0.05),
        "ape": np.asarray(rng.standard_normal((RATIO, HEAD_DIM)), np.float32),
        "norm": _bf16_master(rng.standard_normal((HEAD_DIM,)) * 0.02 + 1.0),
        "cos": np.cos(angle).astype(np.float32),
        "sin": np.sin(angle).astype(np.float32),
        "sink": np.asarray(np.linspace(-0.5, 0.25, HEADS), np.float32),
    }


def _stream(batch: int, length: int, seed: int) -> dict:
    """Per-request tensors indexed by absolute position, shared with the oracle."""
    rng = np.random.default_rng(seed)
    return {
        "hidden": _bf16_master(rng.standard_normal((batch, length, HIDDEN)) * 0.05),
        "q": _bf16_master(rng.standard_normal((batch, length, HEADS, HEAD_DIM))),
        "kv": _bf16_master(rng.standard_normal((batch, length, HEAD_DIM))),
    }


class _Driver:
    """Hold one runtime and run successive stateful steps against it."""

    def __init__(self, batch: int, max_context_len: int, weights: dict):
        self.batch = batch
        self.weights = weights
        (
            self.mesh,
            self.kv_pool,
            self.state_pool,
            self.request_pool,
            self.allocator,
        ) = HCA_TEST.runtime(requests=batch, max_context_len=max_context_len)
        with jax.set_mesh(self.mesh):
            self.req_indices = np.asarray(
                self.allocator.alloc([HCA_TEST.request() for _ in range(batch)]),
                np.int32,
            )
            self.backend = HCABackend(mesh=self.mesh, page_size=128)
            self.backend.allocator = self.allocator
            self.device_weights = {
                "wkv": self._put(weights["wkv"], P(None, None), jax.numpy.bfloat16),
                "wgate": self._put(weights["wgate"], P(None, None), jax.numpy.bfloat16),
                "ape": self._put(weights["ape"], P(None, None), jax.numpy.float32),
                "norm": self._put(weights["norm"], P(None), jax.numpy.bfloat16),
                "cos": self._put(weights["cos"], P(None, None), jax.numpy.float32),
                "sin": self._put(weights["sin"], P(None, None), jax.numpy.float32),
                "sink": self._put(weights["sink"], P("tensor"), jax.numpy.float32),
            }

    def _put(self, value, spec, dtype):
        return jax.device_put(
            jax.numpy.asarray(value, dtype), NamedSharding(self.mesh, spec)
        )

    def step(self, stream, mode, plan, seq_lens, q_lens, prefix_lens):
        """Run one step for ``plan`` = [(request, positions), ...]."""
        hidden = np.concatenate([stream["hidden"][r][list(p)] for r, p in plan])
        q = np.concatenate([stream["q"][r][list(p)] for r, p in plan])
        new_kv = np.concatenate([stream["kv"][r][list(p)] for r, p in plan])
        positions = np.concatenate([np.asarray(p, np.int32) for _, p in plan])
        seq_lens = np.asarray(seq_lens, np.int32)

        with jax.set_mesh(self.mesh):
            self.allocator.ensure_compressed_capacity(self.req_indices, seq_lens)
            self.backend.forward_metadata = self.backend.get_forward_metadata(
                SimpleNamespace(
                    forward_mode=mode,
                    req_pool_indices=self.req_indices,
                    seq_lens=seq_lens,
                    positions=positions,
                    extend_seq_lens=None
                    if q_lens is None
                    else np.asarray(q_lens, np.int32),
                    extend_prefix_lens=(
                        None
                        if prefix_lens is None
                        else np.asarray(prefix_lens, np.int32)
                    ),
                    recurrent_indices=self.request_pool.get_linear_recurrent_indices(
                        self.req_indices
                    ),
                )
            )
            output, update = self.backend(
                self._put(q, P("data", "tensor", None), jax.numpy.bfloat16),
                self._put(new_kv, P("data", None), jax.numpy.bfloat16),
                self._put(new_kv, P("data", None), jax.numpy.bfloat16),
                SimpleNamespace(layer_id=0, scaling=SOFTMAX_SCALE),
                SimpleNamespace(
                    forward_mode=mode,
                    positions=jax.device_put(
                        positions, NamedSharding(self.mesh, P("data"))
                    ),
                ),
                self.kv_pool,
                recurrent_state_pool=self.state_pool,
                compressor_input=self._put(hidden, P("data", None), jax.numpy.bfloat16),
                wkv=self.device_weights["wkv"],
                wgate=self.device_weights["wgate"],
                ape=self.device_weights["ape"],
                norm_weight=self.device_weights["norm"],
                cos=self.device_weights["cos"],
                sin=self.device_weights["sin"],
                attention_sink=self.device_weights["sink"],
            )
            jax.block_until_ready((output, update))
            MemoryPools(
                token_to_kv_pool=self.kv_pool, recurrent_state_pool=self.state_pool
            ).replace_all(self.backend.pack_pool_updates([update]))
        return np.asarray(output, np.float32).reshape(-1, HEADS, HEAD_DIM)


def _check(actual, stream, plan, weights):
    expected = np.concatenate(
        [
            oracle.request_outputs(stream, r, list(p), weights, SOFTMAX_SCALE)
            for r, p in plan
        ]
    )
    metrics = oracle.compare(actual, expected)
    assert metrics["nan"] == 0 and metrics["inf"] == 0, metrics
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch,sequence", [(1, 384), (3, 256), (2, 1024)])
def test_hca_uniform_prefill_matches_reference(batch, sequence):
    """Zero-prefix equal-length EXTEND: the uniform fast path.

    The 1024-token case is load-bearing: compressed records are only ~1.5% of
    the attended keys at S=256, which dilutes a broken record (wrong RoPE, wrong
    RMSNorm) below the gate. At S=1024 they are ~6% and the same defect shows up
    100x over baseline.
    """
    weights = _weights(20260901)
    stream = _stream(batch, sequence, seed=11 + batch)
    driver = _Driver(batch, sequence + RATIO, weights)
    plan = [(r, range(sequence)) for r in range(batch)]
    output = driver.step(
        stream,
        ForwardMode.EXTEND,
        plan,
        [sequence] * batch,
        [sequence] * batch,
        [0] * batch,
    )
    _check(output, stream, plan, weights)


def test_hca_decode_chain_matches_reference():
    """Prefill, then successive single-token decodes off the persisted state."""
    batch, prefill, steps = 2, 260, 4
    weights = _weights(20260902)
    stream = _stream(batch, prefill + steps, seed=23)
    driver = _Driver(batch, prefill + steps + RATIO, weights)

    plan = [(r, range(prefill)) for r in range(batch)]
    driver.step(
        stream,
        ForwardMode.EXTEND,
        plan,
        [prefill] * batch,
        [prefill] * batch,
        [0] * batch,
    )
    for step in range(steps):
        position = prefill + step
        plan = [(r, [position]) for r in range(batch)]
        output = driver.step(
            stream, ForwardMode.DECODE, plan, [position + 1] * batch, None, None
        )
        _check(output, stream, plan, weights)


def test_hca_ragged_extend_matches_reference():
    """Mixed q_len and nonzero prefixes, continued across two chunks."""
    weights = _weights(20260903)
    first = {"q_lens": [1, 131, 100], "prefix_lens": [0, 0, 0]}
    second = {"q_lens": [1, 130, 64], "prefix_lens": [1, 131, 100]}
    batch = len(first["q_lens"])
    longest = max(p + q for p, q in zip(second["prefix_lens"], second["q_lens"]))
    stream = _stream(batch, longest, seed=37)
    driver = _Driver(batch, longest + RATIO, weights)

    for chunk in (first, second):
        plan = [
            (r, range(p, p + q))
            for r, (p, q) in enumerate(zip(chunk["prefix_lens"], chunk["q_lens"]))
        ]
        seq_lens = [p + q for p, q in zip(chunk["prefix_lens"], chunk["q_lens"])]
        output = driver.step(
            stream,
            ForwardMode.EXTEND,
            plan,
            seq_lens,
            chunk["q_lens"],
            chunk["prefix_lens"],
        )
        _check(output, stream, plan, weights)
