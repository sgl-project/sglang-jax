"""FlashAttention derives RPA v3's decode_only / skip_decode from forward_mode.

decode_only drops the prefill and mixed pallas_calls and is only safe when every
sequence is a plain decode; skip_decode drops the decode pallas_call and is only
safe when the batch has no decode sequences. The kernel is stubbed to record the
keyword arguments the backend passes.
"""

import os

# MUST precede any jax import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention import flashattention_backend as fab
from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    FlashAttentionMetadata,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, NUM_TOKENS = 4, 2, 128, 16, 2


@pytest.fixture
def rpa_kwargs(monkeypatch):
    captured = {}

    def _stub(queries, keys, values, kv_cache, *args, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return queries, kv_cache

    monkeypatch.setattr(fab, "ragged_paged_attention_v3", _stub)
    return captured


def _call_backend(monkeypatch, forward_mode, spec_info=None):
    mesh = jax.make_mesh(
        (1, 1),
        ("data", "tensor"),
        devices=jax.devices()[:1],
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )

    def put(x, *spec):
        return jax.device_put(x, NamedSharding(mesh, P(*spec)))

    with jax.set_mesh(mesh):
        backend = FlashAttention(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size=PAGE_SIZE, mesh=mesh)
        metadata = FlashAttentionMetadata()
        metadata.seq_lens = put(np.array([5, 9], np.int32), "data")
        metadata.cu_q_lens = put(np.array([0, 1, 2], np.int32), "data")
        metadata.cu_kv_lens = put(np.array([0, PAGE_SIZE, 2 * PAGE_SIZE], np.int32), "data")
        metadata.page_indices = put(np.array([1, 2], np.int32), "data")
        metadata.distribution = put(np.array([2, 2, 2], np.int32), "data")
        backend.forward_metadata = metadata
        kv_cache = put(
            jnp.zeros((4, PAGE_SIZE, NUM_KV_HEADS, 2, HEAD_DIM), jnp.bfloat16),
            "data",
            None,
            "tensor",
            None,
            None,
        )
        monkeypatch.setattr(backend, "_get_fused_kv_cache", lambda *_: kv_cache)
        layer = RadixAttention(NUM_HEADS, HEAD_DIM, HEAD_DIM**-0.5, NUM_KV_HEADS, layer_id=0)
        q = put(jnp.zeros((NUM_TOKENS, NUM_HEADS * HEAD_DIM), jnp.bfloat16), "data", "tensor")
        kv = put(jnp.zeros((NUM_TOKENS, NUM_KV_HEADS * HEAD_DIM), jnp.bfloat16), "data", "tensor")
        forward_batch = SimpleNamespace(forward_mode=forward_mode, spec_info=spec_info)
        backend(q, kv, kv, layer, forward_batch, token_to_kv_pool=object())


@pytest.mark.parametrize(
    "forward_mode,spec_info,decode_only,skip_decode",
    [
        (ForwardMode.DECODE, None, True, False),
        # Speculative draft decode can carry multi-token segments: keep every stage.
        (ForwardMode.DECODE, SimpleNamespace(), False, False),
        (ForwardMode.EXTEND, None, False, True),
        (ForwardMode.MIXED, None, False, True),
        (ForwardMode.TARGET_VERIFY, None, False, True),
        (ForwardMode.DRAFT_EXTEND, None, False, True),
    ],
    ids=["decode", "spec-decode", "extend", "mixed", "target-verify", "draft-extend"],
)
def test_stage_flags_follow_forward_mode(
    monkeypatch, rpa_kwargs, forward_mode, spec_info, decode_only, skip_decode
):
    _call_backend(monkeypatch, forward_mode, spec_info)
    assert rpa_kwargs["decode_only"] is decode_only
    assert rpa_kwargs["skip_decode"] is skip_decode
    assert not (rpa_kwargs["decode_only"] and rpa_kwargs["skip_decode"])
    for flag in ("predicate_cache_dma", "elide_single_tile_loops", "compact_span_mask"):
        assert rpa_kwargs[flag] is True, flag
