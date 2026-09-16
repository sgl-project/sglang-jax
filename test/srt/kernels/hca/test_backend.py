"""Complete HCA backend tests using native SGLang cache contracts."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hca_backend import HCABackend
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from .common import HCATestFactory

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="production HCA requires TPU"
)
HCA_TEST = HCATestFactory()


def _worker_batch(
    mode, request_pool, allocator, req_indices, positions, seq_lens, q_lens, prefix_lens
):
    """Mimic the scheduler: grow the compressed tier and fill the standard
    hybrid-recurrent batch fields before metadata construction."""
    req_indices = np.asarray(req_indices, np.int32)
    seq_lens = np.asarray(seq_lens, np.int32)
    allocator.ensure_compressed_capacity(req_indices, seq_lens)
    recurrent_indices = np.zeros_like(req_indices)
    active = seq_lens > 0
    recurrent_indices[active] = request_pool.get_linear_recurrent_indices(req_indices[active])
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=req_indices,
        seq_lens=seq_lens,
        positions=np.asarray(positions, np.int32),
        extend_seq_lens=(None if q_lens is None else np.asarray(q_lens, np.int32)),
        extend_prefix_lens=(None if prefix_lens is None else np.asarray(prefix_lens, np.int32)),
        recurrent_indices=recurrent_indices,
    )


def _run_backend(
    mode,
    positions,
    seq_lens,
    q_lens,
    prefix_lens,
    *,
    seed,
):
    batch_size = len(seq_lens)
    mesh, kv_pool, state_pool, request_pool, allocator = HCA_TEST.runtime(
        requests=batch_size, max_context_len=max(512, int(np.max(seq_lens)))
    )
    active_requests = np.asarray(seq_lens, np.int32) > 0
    requests = [HCA_TEST.request() for _ in range(int(active_requests.sum()))]
    with jax.set_mesh(mesh):
        req_indices = np.full((batch_size,), -1, np.int32)
        req_indices[active_requests] = np.asarray(allocator.alloc(requests), np.int32)
        worker_batch = _worker_batch(
            mode,
            request_pool,
            allocator,
            req_indices,
            positions,
            seq_lens,
            q_lens,
            prefix_lens,
        )
        backend = HCABackend(mesh=mesh, page_size=128)
        backend.allocator = allocator
        backend.forward_metadata = backend.get_forward_metadata(worker_batch)
        hidden, q, new_kv, wkv, wgate, ape, norm, cos, sin, sink = HCA_TEST.inputs(
            mesh, len(positions), seed=seed
        )
        device_positions = jax.device_put(
            np.asarray(positions, np.int32), NamedSharding(mesh, P("data"))
        )
        output, update = backend(
            q,
            new_kv,
            new_kv,
            SimpleNamespace(layer_id=0, scaling=512**-0.5),
            SimpleNamespace(forward_mode=mode, positions=device_positions),
            kv_pool,
            recurrent_state_pool=state_pool,
            compressor_input=hidden,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm_weight=norm,
            cos=cos,
            sin=sin,
            attention_sink=sink,
        )
        jax.block_until_ready((output, update))
        pools = MemoryPools(
            token_to_kv_pool=kv_pool,
            recurrent_state_pool=state_pool,
        )
        pools.replace_all(backend.pack_pool_updates([update]))
        state, window, compressed = update
        assert kv_pool.window_buffer[0] is window
        assert kv_pool.compressed_buffer[0] is compressed
        assert state_pool.state_buffers[0] is state
    assert output.shape == (len(positions), 64 * 512)
    assert np.isfinite(np.asarray(output, np.float32)).all()
    return output, update, backend.forward_metadata.kernel


def test_hca_decode_updates_state_and_both_cache_tiers():
    _, update, metadata = _run_backend(
        ForwardMode.DECODE,
        [127, 128],
        [128, 129],
        None,
        None,
        seed=20260814,
    )
    state, window, compressed = update
    assert np.any(np.asarray(state[1]) != 0)
    assert np.any(np.asarray(window) != 0)
    boundary = np.asarray(metadata.boundary_token_indices)
    tokens = int(metadata.valid_token_mask.shape[0])
    assert boundary[boundary < tokens].tolist() == [0]  # padded entries carry ``tokens``
    assert np.any(np.asarray(compressed) != 0)


def test_hca_decode_ignores_padded_request_rows():
    output, _, metadata = _run_backend(
        ForwardMode.DECODE,
        [127, 128, 0, 0],
        [128, 129, 0, 0],
        None,
        None,
        seed=20260816,
    )
    np.testing.assert_array_equal(np.diff(np.asarray(metadata.cu_q_lens)), [1, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(metadata.valid_token_mask), [True, True, False, False])
    np.testing.assert_array_equal(np.asarray(metadata.state_slots)[2:], [0, 0])
    np.testing.assert_array_equal(np.asarray(output)[2:], 0)


def test_hca_extend_ignores_padded_request_rows():
    _, _, metadata = _run_backend(
        ForwardMode.EXTEND,
        [128, 129, 128],
        [130, 129, 0, 0],
        [2, 1, 0, 0],
        [128, 128, 0, 0],
        seed=20260817,
    )
    np.testing.assert_array_equal(np.diff(np.asarray(metadata.cu_q_lens)), [2, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(metadata.state_slots), [1, 1, 2])


def test_hca_extend_token_padding_runs_ragged_path():
    positions = np.pad(np.tile(np.arange(130, dtype=np.int32), 2), (0, 512 - 260))
    output, _, metadata = _run_backend(
        ForwardMode.EXTEND,
        positions,
        [130, 130],
        [130, 130],
        [0, 0],
        seed=20260818,
    )
    np.testing.assert_array_equal(np.asarray(metadata.valid_token_mask)[:260], True)
    np.testing.assert_array_equal(np.asarray(metadata.valid_token_mask)[260:], False)
    np.testing.assert_array_equal(np.asarray(output)[260:], 0)


@pytest.mark.parametrize(
    "positions,seq_lens,q_lens,prefix_lens",
    [
        (list(range(128)), [128], [128], [0]),
        # 9 complete groups: exercises the partial trailing compression step
        # (prefill_entries_per_step does not divide the group count).
        (list(range(1152)), [1152], [1152], [0]),
        ([128, 129, 128], [130, 129], [2, 1], [128, 128]),
    ],
)
def test_hca_extend_uniform_and_ragged(positions, seq_lens, q_lens, prefix_lens):
    _run_backend(
        ForwardMode.EXTEND,
        positions,
        seq_lens,
        q_lens,
        prefix_lens,
        seed=20260815 + len(positions),
    )


def _metadata_signature(
    backend, request_pool, allocator, req_indices, mode, positions, q_lens, prefix_lens
):
    positions = np.asarray(positions, np.int32)
    if mode == ForwardMode.DECODE:
        # Decode contract: one query token per request, sitting at seq_len - 1.
        seq_lens = positions + 1
    else:
        seq_lens = np.asarray(prefix_lens, np.int32) + np.asarray(q_lens, np.int32)
    batch = _worker_batch(
        mode,
        request_pool,
        allocator,
        req_indices,
        positions,
        seq_lens,
        q_lens,
        prefix_lens,
    )
    metadata = backend.get_forward_metadata(batch)
    leaves, treedef = jax.tree_util.tree_flatten(metadata)
    return treedef, tuple((leaf.shape, leaf.dtype) for leaf in leaves)


def test_hca_forward_metadata_shapes_are_stable_across_drift():
    """Boundary-count, compressed-entry, and page-table drift must not change
    compiled metadata shapes; only the empty/non-empty boundary split may."""
    mesh, _, _, request_pool, allocator = HCA_TEST.runtime(requests=2, max_context_len=2048)
    requests = [HCA_TEST.request(), HCA_TEST.request()]
    with jax.set_mesh(mesh):
        req_indices = np.asarray(allocator.alloc(requests), np.int32)
        backend = HCABackend(mesh=mesh, page_size=128)
        backend.allocator = allocator

        no_boundary = [
            _metadata_signature(
                backend,
                request_pool,
                allocator,
                req_indices,
                ForwardMode.DECODE,
                p,
                None,
                None,
            )
            for p in ([1000, 1001], [1160, 1281])  # entry counts 7/7 vs 9/10
        ]
        with_boundary = [
            _metadata_signature(
                backend,
                request_pool,
                allocator,
                req_indices,
                ForwardMode.DECODE,
                p,
                None,
                None,
            )
            for p in ([1023, 1001], [1023, 895])  # 1 vs 2 boundary tokens
        ]
        assert no_boundary[0] == no_boundary[1]
        assert with_boundary[0] == with_boundary[1]

        extend = [
            _metadata_signature(
                backend,
                request_pool,
                allocator,
                req_indices,
                ForwardMode.EXTEND,
                np.concatenate([np.arange(p, p + q) for p, q in zip(prefixes, (130, 64))]),
                [130, 64],
                prefixes,
            )
            for prefixes in ([0, 0], [126, 120])  # 1 vs 3 boundary tokens
        ]
        assert extend[0] == extend[1]


def test_hca_uniform_fast_path_selection():
    """Zero-prefix equal-length prompts take the uniform path; ragged mixes do not."""
    mesh, _, _, request_pool, allocator = HCA_TEST.runtime(requests=2, max_context_len=512)
    requests = [HCA_TEST.request(), HCA_TEST.request()]
    with jax.set_mesh(mesh):
        req_indices = np.asarray(allocator.alloc(requests), np.int32)
        backend = HCABackend(mesh=mesh, page_size=128)
        backend.allocator = allocator
        uniform = _worker_batch(
            ForwardMode.EXTEND,
            request_pool,
            allocator,
            req_indices,
            np.tile(np.arange(128, dtype=np.int32), 2),
            [128, 128],
            [128, 128],
            [0, 0],
        )
        ragged = _worker_batch(
            ForwardMode.EXTEND,
            request_pool,
            allocator,
            req_indices,
            [128, 129, 128],
            [130, 129],
            [2, 1],
            [128, 128],
        )
        token_padded = _worker_batch(
            ForwardMode.EXTEND,
            request_pool,
            allocator,
            req_indices,
            np.pad(np.tile(np.arange(130, dtype=np.int32), 2), (0, 512 - 260)),
            [130, 130],
            [130, 130],
            [0, 0],
        )
        assert backend.get_forward_metadata(uniform).use_uniform_prefill_fast_path
        assert not backend.get_forward_metadata(ragged).use_uniform_prefill_fast_path
        assert not backend.get_forward_metadata(token_padded).use_uniform_prefill_fast_path
