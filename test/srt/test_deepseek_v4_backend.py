"""Unified B entry point against real C pools; no serving/runtime wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend
from sgl_jax.srt.layers.attention.dsv4.execution import CompressorWeights, IndexerInputs
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def resources(page_size=128, dp=1, tp=1):
    mesh = Mesh(np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp), ("data", "tensor"))
    spec = DeepseekV4CacheSpec((0, 4, 128), head_dim=128, index_head_dim=128)
    kv = DeepseekV4TokenToKVPool(8 * page_size * dp, 8 * page_size * dp, page_size, spec, mesh, dp)
    state = DeepseekV4CompressStatePool(2, spec, mesh, dp)
    req_pool = ReqToTokenPool(2, 512)
    allocator = DeepseekV4TokenToKVPoolAllocator(kv)
    backend = DeepseekV4AttentionBackend(
        mesh=mesh,
        page_size=page_size,
        max_context_len=512,
        config=SimpleNamespace(hidden_size=16, num_attention_heads=2, head_dim=128),
    )
    backend.bind_resources(req_pool, allocator)
    return backend, req_pool, allocator, kv, state


def append(req_pool, allocator, slot, start, end, rank=0):
    last = int(req_pool.req_to_token[slot, start - 1]) if start else -1
    locations = allocator.alloc_extend([start], [end], [last], end - start, rank)
    assert locations is not None
    req_pool.write((slot, slice(start, end)), locations)
    return locations


def batch(req_pool, slot, start, end, *, mode=ForwardMode.EXTEND, padding=2):
    n = end - start
    return SimpleNamespace(
        forward_mode=mode,
        seq_lens=np.array([end, 0], np.int32),
        req_pool_indices=np.array([slot, req_pool.size], np.int32),
        positions=np.r_[np.arange(start, end), np.zeros(padding)].astype(np.int32),
        cache_loc=np.zeros(((end + 127) // 128 * 128,), np.int32),
        out_cache_loc=np.r_[req_pool.req_to_token[slot, start:end], np.full(padding, -1)].astype(
            np.int32
        ),
        extend_seq_lens=np.array([n, 0], np.int32),
        extend_prefix_lens=np.array([start, 0], np.int32),
    )


def test_metadata_is_host_only_and_derived_from_c_ownership():
    backend, req, allocator, _, _ = resources()
    append(req, allocator, 0, 0, 129)
    b = batch(req, 0, 0, 129)
    # B has no submission path, including when preparing HCA's kernel metadata.
    backend.use_pallas_hca = True
    with (
        patch.object(jax, "device_put", side_effect=AssertionError("R owns device transfer")),
        patch(
            "sgl_jax.srt.layers.attention.dsv4.hca.get_hca_kernel_schedule",
            side_effect=lambda _, **kw: get_hca_kernel_schedule(
                "TPU7x", **(kw | {"local_heads": 8})
            ),
        ),
    ):
        md = backend.get_forward_metadata(b, request_pool=req, allocator=allocator)
    assert isinstance(md.packed, np.ndarray) and md.packed.dtype == np.int32
    attention, tables = md.resolve()
    np.testing.assert_array_equal(attention.c4.visible_entries_after, [32, 0])
    np.testing.assert_array_equal(attention.c128.visible_entries_after, [1, 0])
    np.testing.assert_array_equal(attention.state_init_mask, [True, False])
    np.testing.assert_array_equal(attention.valid_token_mask, [True] * 129 + [False] * 2)
    assert int(tables[1].compressed_rows[0]) == int(req.req_to_token[0, 0]) // 4
    assert np.all(np.asarray(attention.swa_write_loc)[129:] == -1)
    assert md.hca_metadata().kernel is not None


@pytest.mark.parametrize("failure", ["unbound", "positions", "writes", "released", "prefix"])
def test_metadata_rejects_inconsistent_resource_inputs(failure):
    backend, req, allocator, _, _ = resources()
    append(req, allocator, 0, 0, 5)
    b = batch(req, 0, 0, 5)
    if failure == "unbound":
        backend.resources_bound = False
    elif failure == "positions":
        b.positions[0] = 1
    elif failure == "writes":
        b.out_cache_loc[0] += 1
    elif failure == "released":
        allocator.free_swa(req.req_to_token[0, :5])
    else:
        b.extend_prefix_lens[0] = 1
    with pytest.raises(RuntimeError if failure == "unbound" else ValueError):
        backend.get_forward_metadata(b, request_pool=req, allocator=allocator)


def test_decode_page_tables_and_state_continuation():
    backend, req, allocator, _, _ = resources()
    append(req, allocator, 1, 0, 130)
    md = backend.get_forward_metadata(
        batch(req, 1, 129, 130, mode=ForwardMode.DECODE), request_pool=req, allocator=allocator
    )
    attention, tables = md.resolve()
    np.testing.assert_array_equal(attention.state_init_mask, [False, False])
    np.testing.assert_array_equal(attention.query_positions[:1], [129])
    first_page = int(req.req_to_token[1, 0]) // 128
    np.testing.assert_array_equal(tables[1].decode_page_indices[0], [first_page, 0, 0, 0])
    assert int(tables[1].decode_page_segment_counts[0, 0]) == 1
    assert np.all(np.asarray(tables[1].decode_window_rows[1:]) == 0)


def test_mixed_extend_metadata_separates_reordered_requests():
    backend, req, allocator, _, _ = resources()
    append(req, allocator, 0, 0, 4)
    append(req, allocator, 1, 0, 129)
    b = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=np.array([129, 4, 0], np.int32),
        req_pool_indices=np.array([1, 0, 2], np.int32),
        positions=np.array([128, 0, 1, 2, 3, 0, 0, 0], np.int32),
        out_cache_loc=np.r_[req.req_to_token[1, 128], req.req_to_token[0, :4], [-1] * 3].astype(
            np.int32
        ),
        extend_seq_lens=np.array([1, 4, 0], np.int32),
        extend_prefix_lens=np.array([128, 0, 0], np.int32),
    )
    md = backend.get_forward_metadata(b, request_pool=req, allocator=allocator)
    attention, tables = md.resolve()
    np.testing.assert_array_equal(attention.query_request_ids, [0, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(attention.state_init_mask, [False, True, False])
    np.testing.assert_array_equal(attention.c4.boundary_token_indices[:1], [4])
    np.testing.assert_array_equal(attention.c4.boundary_state_slots[:1], [0])
    np.testing.assert_array_equal(attention.c4.visible_entries_after, [32, 1, 0])
    np.testing.assert_array_equal(np.asarray(tables[1].compressed_request_ids)[:33], [0] * 32 + [1])


def weights(rng, ratio, dim=128, hidden=16):
    width = dim * (2 if ratio == 4 else 1)
    return CompressorWeights(
        jnp.asarray(rng.normal(size=(width, hidden)) * 0.02, jnp.bfloat16),
        jnp.asarray(rng.normal(size=(width, hidden)) * 0.02, jnp.bfloat16),
        jnp.asarray(rng.normal(size=(ratio, width)) * 0.1, jnp.float32),
        jnp.ones(dim, jnp.float32),
        jnp.asarray(np.tile(np.r_[np.ones(32), np.zeros(32)], (512, 1)), jnp.float32),
    )


def run_chunks(geometry, layer_id, boundaries, *, decode_last=False):
    backend, req, allocator, kv, state = geometry
    ratio = kv.spec.compress_ratios[layer_id]
    rng = np.random.default_rng(42)
    tokens = boundaries[-1]
    q = jnp.asarray(rng.normal(size=(tokens, 2, 128)) * 0.1, jnp.bfloat16)
    new_kv = jnp.asarray(rng.normal(size=(tokens, 128)) * 0.1, jnp.bfloat16)
    hidden = jnp.asarray(rng.normal(size=(tokens, 16)) * 0.1, jnp.bfloat16)
    compressor = weights(rng, ratio) if ratio else None
    index_compressor = weights(rng, 4) if ratio == 4 else None
    index_q = jnp.asarray(rng.normal(size=(tokens, 2, 128)), jnp.bfloat16)
    index_weights = jnp.ones((tokens, 2), jnp.float32)
    layer = SimpleNamespace(layer_id=layer_id, scaling=128**-0.5)
    outputs = []
    start = 0
    for end in boundaries:
        append(req, allocator, 0, start, end)
        mode = ForwardMode.DECODE if decode_last and end == tokens else ForwardMode.EXTEND
        b = batch(req, 0, start, end, mode=mode)
        md = backend.get_forward_metadata(b, request_pool=req, allocator=allocator)
        # The test supplies metadata directly and applies replacement payloads
        # between calls, simulating the interface that R will bind separately.
        backend.forward_metadata = md
        n = end - start
        inputs = lambda array, start=start, end=end: jnp.pad(
            array[start:end], ((0, 2),) + ((0, 0),) * (array.ndim - 1)
        )
        indexer = (
            IndexerInputs(inputs(index_q), inputs(index_weights), index_compressor)
            if ratio == 4
            else None
        )
        before = {
            "kv": [np.asarray(a).copy() for a in jax.tree.leaves(kv)],
            "state": [np.asarray(a).copy() for a in jax.tree.leaves(state)],
        }
        with jax.set_mesh(backend.mesh):
            output, updates = backend(
                inputs(q),
                inputs(new_kv),
                inputs(new_kv),
                layer,
                None,
                kv,
                compressor_state_pool=state,
                compressor_input=inputs(hidden),
                compressor=compressor,
                indexer=indexer,
                attention_sink=jnp.zeros(2, jnp.float32),
                index_topk=512,
            )
        np.testing.assert_array_equal(np.asarray(output)[n:], 0)
        assert np.isfinite(np.asarray(output)).all()
        for owner, key in ((kv, "kv"), (state, "state")):
            for old, current in zip(before[key], jax.tree.leaves(owner), strict=True):
                np.testing.assert_array_equal(old, np.asarray(current))
        payload = backend.pack_pool_updates({layer_id: updates}, kv, state)
        assert set(payload["token_to_kv_pool"]) == set(kv.buffers)
        assert set(payload["compressor_state_pool"]) == set(state.buffers)
        for owner, key in ((kv, "token_to_kv_pool"), (state, "compressor_state_pool")):
            # Untouched layers and padding slots remain exact in a complete payload.
            for family, arrays in payload[key].items():
                if layer_id not in owner.layer_to_buffer[family]:
                    for old, new in zip(owner.buffers[family], arrays, strict=True):
                        assert old is new
                for array in arrays:
                    if owner is state:
                        np.testing.assert_array_equal(
                            np.asarray(array)[1:], np.asarray(owner.buffers[family][0])[1:]
                        )
                    else:
                        np.testing.assert_array_equal(
                            np.asarray(array)[: backend.page_size if family == "swa" else 1], 0
                        )
            owner.replace_buffer(payload[key])
        outputs.append(np.asarray(output)[:n])
        start = end
    return np.concatenate(outputs), kv, state


@pytest.mark.parametrize("layer_id", [0, 1, 2], ids=["swa", "csa", "hca"])
def test_prefill_equals_non_aligned_chunks_and_decode(layer_id):
    whole, whole_kv, whole_state = run_chunks(resources(), layer_id, [131])
    split, split_kv, split_state = run_chunks(
        resources(), layer_id, [3, 127, 130, 131], decode_last=True
    )
    np.testing.assert_allclose(split, whole, rtol=2e-2, atol=2e-3)
    for owner_a, owner_b in ((whole_kv, split_kv), (whole_state, split_state)):
        for a, b in zip(jax.tree.leaves(owner_a), jax.tree.leaves(owner_b), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-2, atol=2e-3)
    assert whole_state.get_buffer("c128", 2).shape == (3, 128, 2, 128)


@pytest.mark.parametrize("layer_id", [1, 2], ids=["csa", "hca"])
def test_zero_prefix_resets_recycled_state_without_touching_other_slots(layer_id):
    expected, _, expected_state = run_chunks(resources(), layer_id, [4])
    dirty = resources()
    state = dirty[-1]
    state.buffers = {
        family: tuple(array.at[0:2].set(7.0) for array in arrays)
        for family, arrays in state.buffers.items()
    }
    actual, _, actual_state = run_chunks(dirty, layer_id, [4])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    families = [f"c{state.spec.compress_ratios[layer_id]}"] + (["indexer"] if layer_id == 1 else [])
    for family in families:
        np.testing.assert_array_equal(
            np.asarray(actual_state.get_buffer(family, layer_id))[0],
            np.asarray(expected_state.get_buffer(family, layer_id))[0],
        )
        np.testing.assert_array_equal(np.asarray(actual_state.get_buffer(family, layer_id))[1], 7.0)


def test_metadata_keeps_dp_rank_local_addresses_and_slots():
    if len(jax.devices()) < 2:
        pytest.skip("requires two logical devices")
    backend, req, allocator, _, _ = resources(dp=2)
    for slot, rank in ((0, 0), (1, 1)):
        append(req, allocator, slot, 0, 4, rank)
    b = batch(req, 0, 0, 4)
    other = batch(req, 1, 0, 4)
    for name in vars(b):
        if name != "forward_mode":
            setattr(b, name, np.concatenate((getattr(b, name), getattr(other, name))))
    md = backend.get_forward_metadata(b, request_pool=req, allocator=allocator)
    attention, _ = md.resolve()
    np.testing.assert_array_equal(attention.request_slots, [0, 2, 1, 2])
    np.testing.assert_array_equal(attention.cu_q_lens, [0, 4, 4, 0, 4, 4])
    np.testing.assert_array_equal(
        attention.history_write_loc[:4], attention.history_write_loc[6:10]
    )


def test_pack_pool_updates_rejects_a_family_on_the_wrong_layer():
    backend, _, _, kv, state = resources()
    with pytest.raises(ValueError, match="no 'compressed' resource"):
        backend.pack_pool_updates({0: {"compressed": kv.get_compressed_buffer(1)}}, kv, state)
