"""CPU equivalence and ownership checks for speculative compact page metadata."""

import copy
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention import flashattention_backend as fa
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def fixture(counts=(2, 2), page=4):
    bs, context = 4, 64
    pages = np.random.default_rng(9).permutation(bs * context // page).reshape(bs, -1)
    pool = NS(req_to_token=(pages[..., None] * page + np.arange(page)).reshape(bs, context))
    lens = np.array([9, 14, 17, 22], np.int32)
    for rank, count in enumerate(counts):
        lens[rank * 2 + count : (rank + 1) * 2] = 0
    spec = EagleDraftInput(
        allocate_lens=np.where(lens > 0, lens + 6, 0),
        verified_id=np.arange(bs, dtype=np.int32),
        topk_p=np.ones((bs, 5), np.float32),
        topk_index=np.zeros((bs, 5), np.int32),
        hidden_states=np.zeros((bs, 16), np.float32),
    )
    batch = NS(
        real_bs=sum(counts),
        dp_size=2,
        per_dp_bs_size=2,
        seq_lens=lens,
        req_pool_indices=np.arange(bs, dtype=np.int32),
        spec_info_padded=spec,
        input_ids=np.zeros(bs, np.int32),
        positions=np.zeros(bs, np.int32),
        out_cache_loc=np.arange(bs * 6, dtype=np.int32),
        cache_loc=np.empty(0, np.int32),
        extend_prefix_lens=None,
        extend_seq_lens=None,
    )
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.server_args = NS(disable_overlap_schedule=False)
    worker.speculative_algorithm = SpeculativeAlgorithm.NEXTN
    worker.topk, worker.speculative_num_steps, worker.speculative_num_draft_tokens = (
        1,
        5,
        6,
    )
    worker.page_size, worker.hot_token_ids = page, None
    worker.precompile_bs_paddings = [bs]
    worker.precompile_cache_loc_paddings = [bs * context]
    worker.target_worker_ref = NS(get_memory_pool=lambda: (pool, None))
    return worker, batch, pool


@pytest.mark.parametrize("counts", [(2, 2), (0, 2), (1, 0), (0, 0)])
@pytest.mark.parametrize("page", [1, 4, 16])
def test_compact_matches_dense_and_survives_next_batch(counts, page):
    worker, batch, pool = fixture(counts, page)
    dense = copy.deepcopy(batch)
    worker.padding_for_decode(dense)
    worker.padding_for_decode(batch, compact_cache=True)
    expected = dense.cache_loc[::page] // page
    np.testing.assert_array_equal(batch.cache_loc_page_indices, expected)
    assert batch.cache_loc.size == 0
    assert not batch.cache_loc_page_indices.flags.writeable
    old = batch.cache_loc_page_indices
    pool.req_to_token += page * 256
    worker.padding_for_decode(batch, compact_cache=True)
    np.testing.assert_array_equal(old, expected)
    assert batch.cache_loc_page_indices is not old
    np.testing.assert_array_equal(batch.seq_lens, dense.seq_lens)
    np.testing.assert_array_equal(batch.out_cache_loc, dense.out_cache_loc)


@pytest.mark.parametrize("swa", [False, True])
def test_fa_metadata_equivalence_reuse_and_mapping_refresh(monkeypatch, swa):
    worker, batch, _ = fixture()
    dense = copy.deepcopy(batch)
    worker.padding_for_decode(dense)
    worker.padding_for_decode(batch, compact_cache=True)
    for b in (batch, dense):
        b.forward_mode = ForwardMode.TARGET_VERIFY
        b.spec_info_padded.custom_mask = None
    backend = NS(mesh=Mesh(np.array(jax.devices()[:1]), ("data",)), page_size=4)
    if swa:
        backend.swa_index_mapping = [np.arange(1024, dtype=np.int32) + 4] * 2
    uploads = []

    def upload(x, sharding):
        uploads.append(x)
        return np.array(x, copy=True)

    monkeypatch.setattr(fa, "device_array", upload)
    expected = fa.FlashAttention.get_eagle_base_metadata(backend, dense)
    first = fa.FlashAttention.get_eagle_base_metadata(backend, batch)
    np.testing.assert_array_equal(first.page_indices, expected.page_indices)
    np.testing.assert_array_equal(first.swa_page_indices, expected.swa_page_indices)
    uploads.clear()
    if swa:
        backend.swa_index_mapping = [np.arange(1024, dtype=np.int32) + 8] * 2
    second = fa.FlashAttention.get_eagle_base_metadata(backend, batch)
    assert second is not first
    assert second.page_indices is first.page_indices
    assert len(uploads) == int(swa)  # SWA is always refreshed.
    if swa:
        np.testing.assert_array_equal(second.swa_page_indices, first.swa_page_indices + 1)
    batch.cache_loc_page_indices = batch.cache_loc_page_indices.copy()
    third = fa.FlashAttention.get_eagle_base_metadata(backend, batch)
    assert third.page_indices is not first.page_indices  # Mutable source: no reuse.
    fourth = fa.FlashAttention.get_eagle_base_metadata(backend, batch)
    assert fourth.page_indices is not third.page_indices


@pytest.mark.parametrize("swa", [False, True])
@pytest.mark.parametrize(
    "mode,device_lengths",
    [
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DRAFT_EXTEND, True),
        (ForwardMode.DRAFT_EXTEND, False),
    ],
)
def test_forward_metadata_matches_dense(monkeypatch, swa, mode, device_lengths):
    worker, batch, _ = fixture()
    dense = copy.deepcopy(batch)
    worker.padding_for_decode(dense)
    worker.padding_for_decode(batch, compact_cache=True)
    for b in (batch, dense):
        b.forward_mode = mode
        b.spec_info_padded.custom_mask = None
        b.spec_info_padded.draft_token_num = 6
        b.spec_info_padded.device_seq_lens_for_draft_extend = device_lengths
        b.logits_indices_selector = np.arange(4, dtype=np.int32)
        b.extend_seq_lens = np.full(4, 6, dtype=np.int32)
    backend = NS(mesh=Mesh(np.array(jax.devices()[:1]), ("data",)), page_size=4)
    if swa:
        backend.swa_index_mapping = [np.arange(1024, dtype=np.int32) + 4] * 2

    def upload(x, sharding):
        if isinstance(x, tuple):
            return tuple(np.array(v, copy=True) for v in x)
        return np.array(x, copy=True)

    monkeypatch.setattr(fa, "device_array", upload)
    base = fa.FlashAttention.get_eagle_base_metadata(backend, batch)
    expected = fa.FlashAttention.get_eagle_forward_metadata(backend, dense)
    actual = fa.FlashAttention.get_eagle_forward_metadata(backend, batch)
    for field in (
        "page_indices",
        "swa_page_indices",
        "cu_q_lens",
        "cu_kv_lens",
        "seq_lens",
        "distribution",
        "custom_mask",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    if mode == ForwardMode.TARGET_VERIFY or device_lengths:
        assert actual.page_indices is base.page_indices
    else:
        assert actual.page_indices is not base.page_indices
