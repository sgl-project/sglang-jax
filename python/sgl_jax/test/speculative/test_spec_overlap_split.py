import inspect
import queue
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.scheduler import (
    Scheduler,
    SpecDraftExtendPhaseResult,
    SpecVerifyPhaseResult,
)
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


def test_spec_verify_phase_result_keeps_dp_padded_accept_layout():
    per_dp_bs = 4
    dp_size = 2
    stride = 4
    total_bs = per_dp_bs * dp_size
    accept_lens = np.array([4, 2, 0, 0, 3, 1, 0, 0], dtype=np.int32)
    next_token_ids = np.arange(total_bs * stride, dtype=np.int32)
    draft = EagleDraftInput(
        verified_id=np.arange(total_bs, dtype=np.int32),
        new_seq_lens=np.arange(total_bs, dtype=np.int32) + 100,
        allocate_lens=np.arange(total_bs, dtype=np.int32) + 128,
        hidden_states=np.zeros((total_bs, 8), dtype=np.float32),
    )

    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        accept_lens=accept_lens,
        allocate_lens=draft.allocate_lens,
        scheduler_next_draft_input=draft,
        draft_extend_state={"stride": stride},
        bid=7,
        cache_miss_count=0,
    )

    assert result.accept_lens.shape == (total_bs,)
    assert result.next_token_ids.shape == (total_bs * stride,)
    assert result.scheduler_next_draft_input.new_seq_lens.shape == (total_bs,)


def test_split_phase_entrypoints_import():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase

    assert callable(spec_decode_verify_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_verify_phase")


def test_split_phase_wrapper_entrypoints_import():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
    from sgl_jax.srt.speculative.draft_extend_fused import (
        spec_decode,
        spec_decode_draft_extend_phase,
        spec_decode_verify_phase,
    )

    assert callable(spec_decode)
    assert callable(spec_decode_verify_phase)
    assert callable(spec_decode_draft_extend_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_draft_extend_phase")


def test_spec_precompile_token_paddings_respect_user_paddings():
    from sgl_jax.srt.speculative.base_worker import (
        filter_spec_precompile_token_paddings,
    )

    server_args = SimpleNamespace(precompile_token_paddings=[256, 512], dp_size=4)

    assert filter_spec_precompile_token_paddings(
        server_args,
        [256, 512, 16384],
    ) == [256, 512]


def test_spec_precompile_token_paddings_keep_default_buckets_without_user_paddings():
    from sgl_jax.srt.speculative.base_worker import (
        filter_spec_precompile_token_paddings,
    )

    server_args = SimpleNamespace(precompile_token_paddings=None, dp_size=4)

    assert filter_spec_precompile_token_paddings(
        server_args,
        [256, 512, 16384],
    ) == [256, 512, 16384]


def test_device_array_preserve_device_reuses_matching_sharding(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    arr = jax.device_put(np.array([1, 2], dtype=np.int32), sharding)

    def fail_device_put(*args, **kwargs):
        raise AssertionError("device_put should not be called for matching sharding")

    monkeypatch.setattr(draft_extend_fused.jax, "device_put", fail_device_put)

    assert draft_extend_fused._device_array_preserve_device(arr, sharding) is arr


def test_cached_host_device_array_reuses_equal_host_value(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    owner = SimpleNamespace()
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    first_host = np.array([10, 30], dtype=np.int32)
    second_equal_host = np.array([10, 30], dtype=np.int32)
    converted = object()
    calls = []

    def fake_preserve(value, sharding_arg):
        calls.append((value, sharding_arg))
        return converted

    monkeypatch.setattr(
        draft_extend_fused,
        "_device_array_preserve_device",
        fake_preserve,
    )

    first = draft_extend_fused._cached_host_device_array_preserve_device(
        owner,
        "req_pool_indices",
        first_host,
        sharding,
    )
    second = draft_extend_fused._cached_host_device_array_preserve_device(
        owner,
        "req_pool_indices",
        second_equal_host,
        sharding,
    )

    assert first is converted
    assert second is converted
    assert len(calls) == 1


def test_fused_verify_zero_placeholders_are_cached_device_arrays():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    worker = SimpleNamespace(
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        _fused_verify_placeholder_cache={},
    )

    first = draft_extend_fused._get_fused_verify_zero_placeholders(worker, bs=4, n=3)
    second = draft_extend_fused._get_fused_verify_zero_placeholders(worker, bs=4, n=3)

    assert first.draft_token is second.draft_token
    assert first.positions is second.positions
    assert first.retrive_index is second.retrive_index
    assert first.draft_token.shape == (12,)
    assert first.retrive_index.shape == (4, 3)


def test_same_batch_chain_verify_prepare_skips_padding_cpu_materialization():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    def fail_padding_for_decode(_model_worker_batch):
        raise AssertionError("same-batch chained verify must not CPU-materialize padding")

    worker = SimpleNamespace(
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        topk=1,
        _fused_verify_placeholder_cache={},
        padding_for_decode=fail_padding_for_decode,
    )
    previous_verified_id = jnp.arange(8, dtype=jnp.int32)
    previous_token_list = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
    draft_input = EagleDraftInput(
        verified_id=previous_verified_id,
        topk_index=jnp.zeros((2, 3, 1), dtype=jnp.int32),
        new_seq_lens=jnp.array([12, 24], dtype=jnp.int32),
        allocate_lens=np.array([64, 64], dtype=np.int32),
    )
    draft_input.previous_token_list = previous_token_list
    batch = SimpleNamespace(
        skip_fused_verify_padding_for_decode=True,
        seq_lens=np.array([10, 20], dtype=np.int32),
        seq_lens_sum=30,
        spec_info_padded=draft_input,
    )

    verified_id, token_list = (
        draft_extend_fused._prepare_topk1_verify_placeholders_from_draft_state(
            worker,
            batch,
        )
    )

    assert verified_id is previous_verified_id
    assert token_list is previous_token_list
    assert isinstance(batch.spec_info_padded, EagleVerifyInput)
    assert batch.spec_info_padded.draft_token.shape == (8,)
    assert batch.target_verify_seq_lens_device is draft_input.new_seq_lens


def test_select_next_verified_id_for_verify_uses_accept_lens():
    import jax.numpy as jnp

    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    verified_tree = jnp.array([10, 11, 12, 20, 21, 22], dtype=jnp.int32)
    accept_lens = jnp.array([2, 3], dtype=jnp.int32)

    selected = draft_extend_fused._select_next_verified_id_for_verify(
        verified_tree,
        accept_lens,
    )

    np.testing.assert_array_equal(np.asarray(selected), np.array([11, 22], dtype=np.int32))


def test_target_verify_metadata_reuses_static_device_fields_within_page(monkeypatch):
    import sgl_jax.srt.layers.attention.flashattention_backend as flashattention_backend
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

    calls = []

    def fake_device_array(value, sharding=None):
        calls.append(value)
        if isinstance(value, tuple):
            return value
        return value

    monkeypatch.setattr(flashattention_backend, "device_array", fake_device_array)

    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        dp_size=1,
        per_dp_bs_size=2,
        seq_lens=np.array([100, 120], dtype=np.int32),
        cache_loc=np.arange(256, dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            custom_mask=None,
            draft_token_num=4,
        ),
    )
    attn = FlashAttention(
        num_attn_heads=1,
        num_kv_heads=1,
        head_dim=16,
        page_size=64,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
    )

    first = attn.get_eagle_forward_metadata(batch)
    batch.seq_lens = np.array([101, 121], dtype=np.int32)
    second = attn.get_eagle_forward_metadata(batch)

    assert calls[0][0].shape == first.cu_q_lens.shape
    assert isinstance(calls[1], np.ndarray)
    assert calls[1].shape == second.seq_lens.shape
    assert first.cu_q_lens is second.cu_q_lens
    assert first.cu_kv_lens is second.cu_kv_lens
    assert first.page_indices is second.page_indices
    assert first.distribution is second.distribution
    assert first.seq_lens is not second.seq_lens


def test_draft_extend_metadata_reuses_static_device_fields_within_page(monkeypatch):
    import sgl_jax.srt.layers.attention.flashattention_backend as flashattention_backend
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

    calls = []

    def fake_device_array(value, sharding=None):
        calls.append(value)
        if isinstance(value, tuple):
            return value
        return value

    monkeypatch.setattr(flashattention_backend, "device_array", fake_device_array)

    batch = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND,
        dp_size=1,
        per_dp_bs_size=2,
        seq_lens=np.array([100, 120], dtype=np.int32),
        extend_seq_lens=np.array([4, 4], dtype=np.int32),
        cache_loc=np.arange(256, dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            allocate_lens=np.array([128, 128], dtype=np.int32),
        ),
    )
    attn = FlashAttention(
        num_attn_heads=1,
        num_kv_heads=1,
        head_dim=16,
        page_size=64,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
    )

    first = attn.get_eagle_forward_metadata(batch)
    batch.seq_lens = np.array([101, 121], dtype=np.int32)
    second = attn.get_eagle_forward_metadata(batch)

    assert calls[0][0].shape == first.cu_q_lens.shape
    assert isinstance(calls[1], np.ndarray)
    assert calls[1].shape == second.seq_lens.shape
    assert first.cu_q_lens is second.cu_q_lens
    assert first.cu_kv_lens is second.cu_kv_lens
    assert first.page_indices is second.page_indices
    assert first.distribution is second.distribution
    assert first.seq_lens is not second.seq_lens


def test_spec_prepare_for_decode_uses_reserved_slots_without_allocator(monkeypatch):
    import sgl_jax.srt.speculative.eagle_util as eagle_util

    original_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4

    def fail_alloc(*_args, **_kwargs):
        raise AssertionError("allocator should not run when reserve frontier has slack")

    monkeypatch.setattr(eagle_util, "alloc_paged_token_slots_extend", fail_alloc)

    req_to_token = np.zeros((2, 32), dtype=np.int32)
    req_to_token[0, :] = np.arange(100, 132, dtype=np.int32)
    req_to_token[1, :] = np.arange(200, 232, dtype=np.int32)
    info = SimpleNamespace(
        seq_lens=np.array([10, 11], dtype=np.int32),
        req_pool_indices=np.array([0, 1], dtype=np.int32),
        out_cache_loc=None,
        seq_lens_sum=0,
    )
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 2,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[info],
    )
    spec_info = EagleDraftInput(
        allocate_lens=np.array([24, 24], dtype=np.int32),
        verify_write_lens=np.array([12, 13], dtype=np.int32),
    )

    try:
        spec_info.prepare_for_decode(schedule_batch)
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = original_alloc_len

    np.testing.assert_array_equal(spec_info.allocate_lens, np.array([24, 24], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.verify_write_lens, np.array([13, 14], dtype=np.int32))
    np.testing.assert_array_equal(info.out_cache_loc, np.array([112, 213], dtype=np.int32))


def test_spec_prepare_for_decode_reserves_chain_windows(monkeypatch):
    import sgl_jax.srt.speculative.eagle_util as eagle_util

    original_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4

    alloc_calls = []

    def fake_alloc(_tree_cache, old_reserve, reserve, _last_loc, ext, *, dp_rank):
        alloc_calls.append((old_reserve.copy(), reserve.copy(), int(ext), dp_rank))
        return np.arange(900, 900 + int(ext), dtype=np.int32)

    monkeypatch.setattr(eagle_util, "alloc_paged_token_slots_extend", fake_alloc)

    assigned = []

    def fake_assign(req_pool_indices, req_to_token_pool, old_reserve, reserve, alloc):
        assigned.append((req_pool_indices.copy(), old_reserve.copy(), reserve.copy(), alloc.copy()))
        offset = 0
        for req_pool_idx, old, new in zip(req_pool_indices, old_reserve, reserve, strict=True):
            length = int(new - old)
            if length > 0:
                req_to_token_pool.req_to_token[int(req_pool_idx), int(old) : int(new)] = alloc[
                    offset : offset + length
                ]
                offset += length

    monkeypatch.setattr(eagle_util, "assign_req_to_token_pool", fake_assign)

    req_to_token = np.zeros((2, 32), dtype=np.int32)
    req_to_token[0, :] = np.arange(100, 132, dtype=np.int32)
    req_to_token[1, :] = np.arange(200, 232, dtype=np.int32)
    info = SimpleNamespace(
        seq_lens=np.array([10, 11], dtype=np.int32),
        req_pool_indices=np.array([0, 1], dtype=np.int32),
        out_cache_loc=None,
        seq_lens_sum=0,
    )
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 2,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[info],
    )
    spec_info = EagleDraftInput(
        allocate_lens=np.array([16, 16], dtype=np.int32),
        verify_write_lens=np.array([12, 13], dtype=np.int32),
    )

    try:
        spec_info.prepare_for_decode(schedule_batch)
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = original_alloc_len

    assert alloc_calls
    np.testing.assert_array_equal(spec_info.allocate_lens, np.array([24, 24], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.verify_write_lens, np.array([13, 14], dtype=np.int32))
    np.testing.assert_array_equal(info.out_cache_loc, np.array([112, 213], dtype=np.int32))


def test_spec_peek_reserved_decode_out_cache_loc_has_no_allocator_or_mutation(monkeypatch):
    import sgl_jax.srt.speculative.eagle_util as eagle_util

    original_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4

    def fail_alloc(*_args, **_kwargs):
        raise AssertionError("peek must not allocate")

    monkeypatch.setattr(eagle_util, "alloc_paged_token_slots_extend", fail_alloc)

    req_to_token = np.zeros((2, 32), dtype=np.int32)
    req_to_token[0, :] = np.arange(100, 132, dtype=np.int32)
    req_to_token[1, :] = np.arange(200, 232, dtype=np.int32)
    info = SimpleNamespace(
        seq_lens=np.array([10, 11], dtype=np.int32),
        req_pool_indices=np.array([0, 1], dtype=np.int32),
        out_cache_loc=None,
        seq_lens_sum=0,
    )
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 2,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[info],
    )
    spec_info = EagleDraftInput(
        allocate_lens=np.array([16, 16], dtype=np.int32),
        verify_write_lens=np.array([12, 13], dtype=np.int32),
    )

    try:
        preview = spec_info.peek_reserved_decode_out_cache_loc(schedule_batch)
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = original_alloc_len

    assert preview is not None
    out_cache_chunks, new_write_lens = preview
    np.testing.assert_array_equal(out_cache_chunks[0], np.array([112, 213], dtype=np.int32))
    np.testing.assert_array_equal(new_write_lens, np.array([13, 14], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.verify_write_lens, np.array([12, 13], dtype=np.int32))
    assert info.out_cache_loc is None


def test_spec_peek_reserved_decode_out_cache_loc_rejects_missing_slack():
    original_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4
    info = SimpleNamespace(
        seq_lens=np.array([10], dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
    )
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 1,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
        req_to_token_pool=SimpleNamespace(req_to_token=np.arange(32, dtype=np.int32)[None, :]),
        reqs_info=[info],
    )
    spec_info = EagleDraftInput(
        allocate_lens=np.array([12], dtype=np.int32),
        verify_write_lens=np.array([12], dtype=np.int32),
    )

    try:
        assert spec_info.peek_reserved_decode_out_cache_loc(schedule_batch) is None
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = original_alloc_len


def test_verify_phase_entrypoint_preserves_padded_allocate_lens(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = BaseSpecWorker.__new__(BaseSpecWorker)
    padded_allocate_lens = np.array([10, 20, 0, 30], dtype=np.int32)
    selector = np.array([0, 1, 3], dtype=np.int32)
    model_worker_batch = SimpleNamespace(
        logits_indices_selector=selector,
        spec_info_padded=EagleDraftInput(allocate_lens=padded_allocate_lens),
    )
    captured = {}

    def fake_spec_decode_verify_phase(
        spec_worker,
        model_worker_batch_arg,
        padded_arg,
        compact_arg,
        *,
        predispatch_draft_extend=False,
    ):
        captured["spec_worker"] = spec_worker
        captured["model_worker_batch"] = model_worker_batch_arg
        captured["padded"] = padded_arg
        captured["compact"] = compact_arg
        captured["predispatch"] = predispatch_draft_extend
        return "verify-result"

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_verify_phase",
        fake_spec_decode_verify_phase,
    )

    assert worker.forward_batch_speculative_verify_phase(model_worker_batch) == "verify-result"
    assert captured["spec_worker"] is worker
    assert captured["model_worker_batch"] is model_worker_batch
    np.testing.assert_array_equal(captured["padded"], padded_allocate_lens)
    np.testing.assert_array_equal(captured["compact"], np.array([10, 20, 30], dtype=np.int32))
    assert captured["predispatch"] is False


def test_verify_phase_entrypoint_accepts_compact_allocate_lens_with_padded_selector(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = BaseSpecWorker.__new__(BaseSpecWorker)
    compact_allocate_lens = np.array([10, 20, 30], dtype=np.int32)
    selector = np.array([0, 2, 4], dtype=np.int32)
    model_worker_batch = SimpleNamespace(
        logits_indices_selector=selector,
        spec_info_padded=EagleDraftInput(allocate_lens=compact_allocate_lens),
    )
    captured = {}

    def fake_spec_decode_verify_phase_enqueue(
        spec_worker,
        model_worker_batch_arg,
        padded_arg,
        compact_arg,
    ):
        captured["spec_worker"] = spec_worker
        captured["model_worker_batch"] = model_worker_batch_arg
        captured["padded"] = padded_arg
        captured["compact"] = compact_arg
        return "verify-async"

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_verify_phase_enqueue",
        fake_spec_decode_verify_phase_enqueue,
    )

    assert (
        worker.forward_batch_speculative_verify_phase_enqueue(model_worker_batch) == "verify-async"
    )
    assert captured["spec_worker"] is worker
    assert captured["model_worker_batch"] is model_worker_batch
    np.testing.assert_array_equal(captured["padded"], compact_allocate_lens)
    np.testing.assert_array_equal(captured["compact"], compact_allocate_lens)


def test_verify_phase_enqueue_relays_original_verify_write_lens(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    verify_write_lens = np.array([16, 40], dtype=np.int32)
    batch = SimpleNamespace(
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            verified_id=np.array([10, 30], dtype=np.int32),
            topk_index=np.zeros((2, 3, 1), dtype=np.int32),
            allocate_lens=np.array([32, 64], dtype=np.int32),
            verify_write_lens=verify_write_lens,
        ),
        bid=7,
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
    )

    class FakeVerifyInput:
        def prepare_for_verify(self, model_worker_batch, page_size, target_worker):
            self.prepared = (model_worker_batch, page_size, target_worker)

    def fake_prepare(_draft_worker, model_worker_batch):
        model_worker_batch.spec_info_padded = FakeVerifyInput()
        return jnp.array([10, 30], dtype=jnp.int32), jnp.zeros((2, 4), dtype=jnp.int32)

    fake_forward_batch = SimpleNamespace(bid=None)
    monkeypatch.setattr(
        draft_extend_fused,
        "_prepare_topk1_verify_placeholders_from_draft_state",
        fake_prepare,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_forward_batch_init_new_preserve_device",
        lambda _batch, _target_mr: fake_forward_batch,
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "_logits_metadata_from_model_worker_batch_preserve_device",
        lambda _batch, _mesh, **_kwargs: "logits-metadata",
    )

    def fake_verify_jit(*_args, **_kwargs):
        return (
            jnp.zeros((2, 4), dtype=jnp.float32),
            jnp.arange(8, dtype=jnp.int32),
            jnp.array([14, 38], dtype=jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
            jnp.array([11, 33], dtype=jnp.int32),
            jnp.array([2, 2], dtype=jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
            "pool-updates",
            None,
            None,
        )

    draft_worker = SimpleNamespace(
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        _fused_greedy_verify_jit_fn=fake_verify_jit,
    )
    target_mr = SimpleNamespace(
        _model_def="model-def",
        _model_state_def="state-def",
        model_state_leaves=[],
        memory_pools=SimpleNamespace(replace_all=lambda _updates: None),
        attn_backend=SimpleNamespace(
            get_eagle_forward_metadata=lambda _batch: "forward-metadata",
            forward_metadata=None,
        ),
    )
    spec_worker = SimpleNamespace(
        draft_worker=draft_worker,
        target_worker=SimpleNamespace(model_runner=target_mr),
        page_size=64,
        mesh=draft_worker.mesh,
    )

    async_result = draft_extend_fused.spec_decode_verify_phase_enqueue(
        spec_worker,
        batch,
        np.array([32, 64], dtype=np.int32),
        np.array([32, 64], dtype=np.int32),
    )

    assert (
        async_result.draft_extend_state.batch_output.next_draft_input.verify_write_lens
        is verify_write_lens
    )


def test_verify_phase_entrypoint_can_request_phase_b_predispatch(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = BaseSpecWorker.__new__(BaseSpecWorker)
    model_worker_batch = SimpleNamespace(
        logits_indices_selector=np.array([0], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([10], dtype=np.int32),
        ),
    )
    captured = {}

    def fake_spec_decode_verify_phase(
        spec_worker,
        model_worker_batch_arg,
        padded_arg,
        compact_arg,
        *,
        predispatch_draft_extend=False,
    ):
        captured["predispatch"] = predispatch_draft_extend
        return "verify-result"

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_verify_phase",
        fake_spec_decode_verify_phase,
    )

    assert (
        worker.forward_batch_speculative_verify_phase(
            model_worker_batch,
            predispatch_draft_extend=True,
        )
        == "verify-result"
    )
    assert captured["predispatch"] is True


def test_publish_spec_verify_phase_updates_lengths_without_overwriting_spec_info():
    scheduler = Scheduler.__new__(Scheduler)
    rank0_spec = EagleDraftInput(
        allocate_lens=np.array([100, 200], dtype=np.int32),
        verified_id=np.array([1, 2], dtype=np.int32),
    )
    rank1_spec = EagleDraftInput(
        allocate_lens=np.array([300], dtype=np.int32),
        verified_id=np.array([3], dtype=np.int32),
    )
    batch = SimpleNamespace(
        dp_size=2,
        reqs_info=[
            SimpleNamespace(
                reqs=[object(), object()],
                seq_lens=np.array([10, 20], dtype=np.int32),
                spec_info=rank0_spec,
            ),
            SimpleNamespace(
                reqs=[object()],
                seq_lens=np.array([30], dtype=np.int32),
                spec_info=rank1_spec,
            ),
        ],
    )
    model_worker_batch = SimpleNamespace(
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=2,
    )
    scheduler_next_draft_input = EagleDraftInput(
        verified_id=np.array([11, 22, 33], dtype=np.int32),
        new_seq_lens=np.array([12, 24, 33], dtype=np.int32),
        allocate_lens=np.array([64, 64, 64], dtype=np.int32),
    )
    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=np.arange(8, dtype=np.int32),
        accept_lens=np.array([2, 4, 3, 0], dtype=np.int32),
        allocate_lens=np.array([64, 128, 256, 0], dtype=np.int32),
        scheduler_next_draft_input=scheduler_next_draft_input,
        draft_extend_state=None,
        bid=1,
        cache_miss_count=0,
    )

    scheduler._publish_spec_verify_phase_lengths_to_batch(
        batch,
        model_worker_batch,
        result,
    )

    np.testing.assert_array_equal(batch.reqs_info[0].seq_lens, np.array([12, 24]))
    np.testing.assert_array_equal(batch.reqs_info[1].seq_lens, np.array([33]))
    assert batch.reqs_info[0].spec_info is rank0_spec
    assert batch.reqs_info[1].spec_info is rank1_spec
    np.testing.assert_array_equal(rank0_spec.allocate_lens, np.array([64, 128]))
    np.testing.assert_array_equal(rank1_spec.allocate_lens, np.array([256]))
    np.testing.assert_array_equal(rank0_spec.verified_id, np.array([1, 2]))
    np.testing.assert_array_equal(rank1_spec.verified_id, np.array([3]))


def test_publish_spec_verify_phase_accepts_compact_allocate_lens_from_chain():
    scheduler = Scheduler.__new__(Scheduler)
    rank0_spec = EagleDraftInput(
        allocate_lens=np.array([100, 200], dtype=np.int32),
        verified_id=np.array([1, 2], dtype=np.int32),
    )
    rank1_spec = EagleDraftInput(
        allocate_lens=np.array([300], dtype=np.int32),
        verified_id=np.array([3], dtype=np.int32),
    )
    batch = SimpleNamespace(
        dp_size=2,
        reqs_info=[
            SimpleNamespace(
                reqs=[object(), object()],
                seq_lens=np.array([10, 20], dtype=np.int32),
                spec_info=rank0_spec,
            ),
            SimpleNamespace(
                reqs=[object()],
                seq_lens=np.array([30], dtype=np.int32),
                spec_info=rank1_spec,
            ),
        ],
    )
    model_worker_batch = SimpleNamespace(
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=2,
    )
    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=np.arange(8, dtype=np.int32),
        accept_lens=np.array([2, 4, 3, 0], dtype=np.int32),
        allocate_lens=np.array([64, 128, 256], dtype=np.int32),
        scheduler_next_draft_input=None,
        draft_extend_state=None,
        bid=1,
        cache_miss_count=0,
    )

    scheduler._publish_spec_verify_phase_lengths_to_batch(
        batch,
        model_worker_batch,
        result,
    )

    np.testing.assert_array_equal(batch.reqs_info[0].seq_lens, np.array([12, 24]))
    np.testing.assert_array_equal(batch.reqs_info[1].seq_lens, np.array([33]))
    np.testing.assert_array_equal(rank0_spec.allocate_lens, np.array([64, 128]))
    np.testing.assert_array_equal(rank1_spec.allocate_lens, np.array([256]))


def test_write_back_spec_draft_state_requires_complete_next_draft_input():
    scheduler = Scheduler.__new__(Scheduler)
    batch = SimpleNamespace(
        reqs_info=[
            SimpleNamespace(spec_info=None),
            SimpleNamespace(spec_info=None),
        ],
    )
    model_worker_batch = SimpleNamespace(real_bs=3, real_bs_per_dp=[2, 1])
    incomplete = EagleDraftInput(
        topk_index=None,
        topk_p=np.zeros((3, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((3, 8), dtype=np.float32),
        verified_id=np.arange(3, dtype=np.int32),
        allocate_lens=np.full((3,), 64, dtype=np.int32),
    )

    try:
        scheduler._write_back_spec_draft_state_to_batch(
            batch,
            model_worker_batch,
            incomplete,
        )
    except AssertionError as exc:
        assert "topk_index" in str(exc)
    else:
        raise AssertionError("expected incomplete draft state to be rejected")


def test_write_back_spec_draft_state_splits_complete_next_draft_input():
    scheduler = Scheduler.__new__(Scheduler)
    batch = SimpleNamespace(
        reqs_info=[
            SimpleNamespace(spec_info=None),
            SimpleNamespace(spec_info=None),
        ],
    )
    model_worker_batch = SimpleNamespace(real_bs=3, real_bs_per_dp=[2, 1])
    complete = EagleDraftInput(
        topk_index=np.arange(3, dtype=np.int32).reshape(3, 1, 1),
        topk_p=np.ones((3, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((3, 8), dtype=np.float32),
        verified_id=np.arange(10, 13, dtype=np.int32),
        allocate_lens=np.full((3,), 64, dtype=np.int32),
        new_seq_lens=np.array([12, 24, 33], dtype=np.int32),
    )

    scheduler._write_back_spec_draft_state_to_batch(
        batch,
        model_worker_batch,
        complete,
    )

    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.topk_index,
        np.array([[[0]], [[1]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.new_seq_lens,
        np.array([12, 24], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[1].spec_info.verified_id,
        np.array([12], dtype=np.int32),
    )


def test_write_back_spec_draft_state_by_req_pool_reorders_current_batch():
    scheduler = Scheduler.__new__(Scheduler)
    phase_b_state = EagleDraftInput(
        topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
        topk_p=np.ones((2, 1, 1), dtype=np.float32),
        hidden_states=np.array([[10.0, 0.0], [30.0, 0.0]], dtype=np.float32),
        verified_id=np.array([110, 130], dtype=np.int32),
        allocate_lens=np.array([64, 96], dtype=np.int32),
        new_seq_lens=np.array([14, 38], dtype=np.int32),
    )
    existing_new_req_state = EagleDraftInput(
        topk_index=np.array([[[40]]], dtype=np.int32),
        topk_p=np.ones((1, 1, 1), dtype=np.float32),
        hidden_states=np.array([[40.0, 0.0]], dtype=np.float32),
        verified_id=np.array([140], dtype=np.int32),
        allocate_lens=np.array([128], dtype=np.int32),
        new_seq_lens=np.array([48], dtype=np.int32),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        reqs_info=[
            SimpleNamespace(
                reqs=[object(), object()],
                req_pool_indices=np.array([30, 10], dtype=np.int32),
                spec_info=None,
            ),
            SimpleNamespace(
                reqs=[object()],
                req_pool_indices=np.array([40], dtype=np.int32),
                spec_info=existing_new_req_state,
            ),
        ],
    )
    draft_extend_result = SpecDraftExtendPhaseResult(
        next_draft_input=phase_b_state,
        req_pool_indices=np.array([10, 30], dtype=np.int32),
    )

    scheduler._write_back_spec_draft_state_by_req_pool_to_batch(
        batch,
        draft_extend_result,
    )

    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.topk_index,
        np.array([[[30]], [[10]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.verified_id,
        np.array([130, 110], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[1].spec_info.topk_index,
        np.array([[[40]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[1].spec_info.new_seq_lens,
        np.array([48], dtype=np.int32),
    )


def test_write_back_spec_draft_state_by_req_pool_skips_empty_rank_without_req_pool():
    scheduler = Scheduler.__new__(Scheduler)
    phase_b_state = EagleDraftInput(
        topk_index=np.array([[[10]]], dtype=np.int32),
        topk_p=np.ones((1, 1, 1), dtype=np.float32),
        hidden_states=np.array([[10.0, 0.0]], dtype=np.float32),
        verified_id=np.array([110], dtype=np.int32),
        allocate_lens=np.array([64], dtype=np.int32),
        new_seq_lens=np.array([14], dtype=np.int32),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        reqs_info=[
            SimpleNamespace(
                reqs=[object()],
                req_pool_indices=np.array([10], dtype=np.int32),
                spec_info=None,
            ),
            SimpleNamespace(reqs=[], req_pool_indices=None, spec_info=None),
        ],
    )

    scheduler._write_back_spec_draft_state_by_req_pool_to_batch(
        batch,
        SpecDraftExtendPhaseResult(
            next_draft_input=phase_b_state,
            req_pool_indices=np.array([10], dtype=np.int32),
        ),
    )

    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.verified_id,
        np.array([110], dtype=np.int32),
    )
    assert batch.reqs_info[1].spec_info is None


def test_worker_applies_pending_phase_b_state_to_next_decode_batch_by_req_pool():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.pending_spec_draft_extend_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
            topk_p=np.ones((2, 1, 1), dtype=np.float32),
            hidden_states=np.array([[10.0, 0.0], [30.0, 0.0]], dtype=np.float32),
            verified_id=np.array([110, 130], dtype=np.int32),
            allocate_lens=np.array([64, 96], dtype=np.int32),
            new_seq_lens=np.array([14, 38], dtype=np.int32),
        ),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
    )
    spec_info = EagleDraftInput(
        topk_index=np.array([[[-1]], [[-2]], [[-3]]], dtype=np.int32),
        topk_p=np.zeros((3, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((3, 2), dtype=np.float32),
        verified_id=np.array([-1, -2, -3], dtype=np.int32),
        allocate_lens=np.array([1, 2, 3], dtype=np.int32),
        new_seq_lens=np.array([4, 5, 6], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=np.array([30, 40, 10], dtype=np.int32),
        spec_info_padded=spec_info,
    )

    client._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    np.testing.assert_array_equal(
        spec_info.topk_index,
        np.array([[[30]], [[-2]], [[10]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spec_info.verified_id,
        np.array([130, -2, 110], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spec_info.hidden_states,
        np.array([[30.0, 0.0], [0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
    )
    assert client.pending_spec_draft_extend_result is None


def test_worker_uses_padded_phase_b_topk_when_req_pool_layout_matches():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    padded_topk = np.array([[[10]], [[30]]], dtype=np.int32)
    padded_verified_id = np.array([110, 130], dtype=np.int32)
    padded_token_list = np.array([[10], [30]], dtype=np.int32)
    padded_allocate_lens = np.array([64, 96], dtype=np.int32)
    padded_verify_write_lens = np.array([16, 40], dtype=np.int32)
    client.pending_spec_draft_extend_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(
            topk_index=None,
            topk_p=None,
            hidden_states=None,
            verified_id=np.array([110, 130], dtype=np.int32),
            allocate_lens=np.array([64, 96], dtype=np.int32),
            new_seq_lens=np.array([14, 38], dtype=np.int32),
        ),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=padded_topk,
            verified_id=padded_verified_id,
            allocate_lens=padded_allocate_lens,
            verify_write_lens=padded_verify_write_lens,
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
    )
    client.pending_spec_draft_extend_result.padded_next_draft_input.previous_token_list = (
        padded_token_list
    )
    spec_info = EagleDraftInput(
        topk_index=np.array([[[-1]], [[-2]]], dtype=np.int32),
        topk_p=np.zeros((2, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((2, 2), dtype=np.float32),
        verified_id=np.array([-1, -2], dtype=np.int32),
        allocate_lens=np.array([1, 2], dtype=np.int32),
        new_seq_lens=np.array([4, 5], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        spec_info_padded=spec_info,
    )

    client._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    assert spec_info.topk_index is padded_topk
    assert spec_info.verified_id is padded_verified_id
    assert spec_info.allocate_lens is padded_allocate_lens
    assert spec_info.verify_write_lens is padded_verify_write_lens
    assert spec_info.previous_token_list is padded_token_list
    np.testing.assert_array_equal(spec_info.verified_id, np.array([110, 130]))
    assert client.pending_spec_draft_extend_result is None


def test_worker_apply_pending_falls_back_to_padded_source_when_compact_index_missing():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.pending_spec_draft_extend_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(
            verified_id=np.arange(100, 130, dtype=np.int32),
            allocate_lens=np.arange(64, 94, dtype=np.int32),
            new_seq_lens=np.arange(10, 40, dtype=np.int32),
        ),
        req_pool_indices=np.arange(32, dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            verified_id=np.arange(200, 232, dtype=np.int32),
            allocate_lens=np.arange(300, 332, dtype=np.int32),
            new_seq_lens=np.arange(400, 432, dtype=np.int32),
        ),
        padded_req_pool_indices=np.arange(32, dtype=np.int32),
    )
    spec_info = EagleDraftInput(
        verified_id=np.array([-1], dtype=np.int32),
        allocate_lens=np.array([-1], dtype=np.int32),
        new_seq_lens=np.array([-1], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=np.array([30], dtype=np.int32),
        spec_info_padded=spec_info,
    )

    client._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    np.testing.assert_array_equal(spec_info.verified_id, np.array([230], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.allocate_lens, np.array([330], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.new_seq_lens, np.array([430], dtype=np.int32))
    assert client.pending_spec_draft_extend_result is None


def test_worker_builds_same_batch_chain_candidate_from_preview():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    padded_topk = np.array([[[10]], [[30]]], dtype=np.int32)
    padded_verified_id = np.array([110, 130], dtype=np.int32)
    new_seq_lens = np.array([14, 38], dtype=np.int32)
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=padded_topk,
            verified_id=padded_verified_id,
            new_seq_lens=new_seq_lens,
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=np.array([10, 30], dtype=np.int32),
        same_batch_chain_out_cache_loc_chunks=[
            np.array([100, 101], dtype=np.int32),
            np.array([200, 201], dtype=np.int32),
        ],
        same_batch_chain_verify_write_lens=np.array([16, 40], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([32, 64], dtype=np.int32),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(),
    )

    candidate = client._build_same_batch_spec_chain_candidate_batch(base_batch, pending)

    assert candidate is not base_batch
    assert candidate.req_pool_indices is base_batch.req_pool_indices
    np.testing.assert_array_equal(
        candidate.out_cache_loc,
        np.array([100, 101, 200, 201], dtype=np.int32),
    )
    assert candidate.spec_info_padded.topk_index is padded_topk
    assert candidate.spec_info_padded.verified_id is padded_verified_id
    assert candidate.spec_info_padded.new_seq_lens is new_seq_lens
    np.testing.assert_array_equal(candidate.seq_lens, new_seq_lens)
    assert candidate.seq_lens_sum == int(new_seq_lens.sum())
    np.testing.assert_array_equal(
        candidate.spec_info_padded.allocate_lens,
        np.array([32, 64], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        candidate.spec_info_padded.verify_write_lens,
        np.array([16, 40], dtype=np.int32),
    )
    assert candidate.allow_same_batch_spec_chain


def test_worker_recursively_builds_same_batch_chain_candidate_without_scheduler_preview(
    monkeypatch,
):
    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    req_to_token = np.stack([np.arange(i * 100, i * 100 + 100, dtype=np.int32) for i in range(32)])
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=req_to_token))
    )
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
            verified_id=np.array([110, 130], dtype=np.int32),
            new_seq_lens=np.array([14, 38], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=None,
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=None,
        same_batch_chain_allocate_lens=None,
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([32, 64], dtype=np.int32),
            verify_write_lens=np.array([12, 36], dtype=np.int32),
        ),
    )

    candidate = client._build_same_batch_spec_chain_candidate_batch(base_batch, pending)

    assert candidate is not None
    np.testing.assert_array_equal(
        candidate.out_cache_loc,
        np.concatenate([req_to_token[10, 12:17], req_to_token[30, 36:41]]),
    )
    np.testing.assert_array_equal(
        candidate.spec_info_padded.verify_write_lens,
        np.array([17, 41], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        candidate.seq_lens,
        np.array([14, 38], dtype=np.int32),
    )
    assert candidate.seq_lens_sum == 52
    assert candidate.allow_same_batch_spec_chain


def test_worker_recursively_builds_same_batch_chain_candidate_from_host_seq_lens_relay(
    monkeypatch,
):
    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    req_to_token = np.stack([np.arange(i * 100, i * 100 + 100, dtype=np.int32) for i in range(32)])
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=req_to_token))
    )
    device_new_seq_lens = jnp.array([14, 38], dtype=jnp.int32)
    host_new_seq_lens = np.array([14, 38], dtype=np.int32)
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
            verified_id=np.array([110, 130], dtype=np.int32),
            new_seq_lens=device_new_seq_lens,
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_new_seq_lens_host=host_new_seq_lens,
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=None,
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=None,
        same_batch_chain_allocate_lens=None,
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([32, 64], dtype=np.int32),
            verify_write_lens=np.array([12, 36], dtype=np.int32),
        ),
    )

    candidate = client._build_same_batch_spec_chain_candidate_batch(base_batch, pending)

    assert candidate is not None
    assert candidate.spec_info_padded.new_seq_lens is device_new_seq_lens
    np.testing.assert_array_equal(candidate.seq_lens, host_new_seq_lens)
    np.testing.assert_array_equal(
        candidate.out_cache_loc,
        np.concatenate([req_to_token[10, 12:17], req_to_token[30, 36:41]]),
    )
    np.testing.assert_array_equal(
        candidate.spec_info_padded.verify_write_lens,
        np.array([17, 41], dtype=np.int32),
    )


def test_worker_prebuild_chain_candidate_preserves_spec_adjust_coefficient(
    monkeypatch,
):
    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    req_to_token = np.stack([np.arange(i * 100, i * 100 + 100, dtype=np.int32) for i in range(32)])
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=req_to_token))
    )
    verify_result = SimpleNamespace(
        padded_new_seq_lens_host=np.array([14, 38], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=None,
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=None,
        same_batch_chain_allocate_lens=None,
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([32, 64], dtype=np.int32),
            verify_write_lens=np.array([12, 36], dtype=np.int32),
        ),
    )

    candidate = client._prebuild_same_batch_spec_chain_candidate_after_phase_a(
        base_batch,
        verify_result,
    )

    assert candidate is not None
    np.testing.assert_array_equal(
        candidate.out_cache_loc,
        np.concatenate([req_to_token[10, 12:17], req_to_token[30, 36:41]]),
    )
    np.testing.assert_array_equal(
        candidate.spec_info_padded.verify_write_lens,
        np.array([17, 41], dtype=np.int32),
    )


def test_worker_recursive_chain_pads_out_cache_loc_per_dp_rank(monkeypatch):
    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    req_to_token = np.stack([np.arange(i * 100, i * 100 + 100, dtype=np.int32) for i in range(32)])
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=req_to_token))
    )
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 11, 20, 21], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.zeros((4, 3, 1), dtype=np.int32),
            verified_id=np.array([110, 111, 120, 121], dtype=np.int32),
            new_seq_lens=jnp.array([14, 14, 24, 27], dtype=jnp.int32),
        ),
        padded_req_pool_indices=np.array([10, 11, 20, 21], dtype=np.int32),
        padded_new_seq_lens_host=np.array([14, 14, 24, 27], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=None,
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=None,
        same_batch_chain_allocate_lens=None,
        req_pool_indices=np.array([10, 11, 20, 21], dtype=np.int32),
        seq_lens=np.array([12, 13, 22, 25], dtype=np.int32),
        dp_size=2,
        per_dp_bs_size=2,
        speculative_num_draft_tokens=4,
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([32, 32, 64, 64], dtype=np.int32),
            verify_write_lens=np.array([12, 16, 22, 25], dtype=np.int32),
        ),
    )

    candidate = client._build_same_batch_spec_chain_candidate_batch(base_batch, pending)

    assert candidate is not None
    expected_rank0 = np.concatenate([req_to_token[10, 12:17], req_to_token[11, 16:17]])
    expected_rank1 = np.concatenate([req_to_token[20, 22:27], req_to_token[21, 25:30]])
    np.testing.assert_array_equal(
        candidate.out_cache_loc,
        np.concatenate(
            [
                np.pad(expected_rank0, (0, 2), constant_values=-1),
                expected_rank1[:8],
            ]
        ),
    )
    assert candidate.out_cache_loc.shape == (16,)


def test_worker_recursively_builds_chain_after_verify_replaces_spec_info(monkeypatch):
    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    req_to_token = np.stack([np.arange(i * 100, i * 100 + 100, dtype=np.int32) for i in range(32)])
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=req_to_token))
    )
    first_pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
            verified_id=np.array([110, 130], dtype=np.int32),
            new_seq_lens=jnp.array([14, 38], dtype=jnp.int32),
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_new_seq_lens_host=np.array([14, 38], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=None,
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=None,
        same_batch_chain_allocate_lens=None,
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        seq_lens=np.array([12, 36], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            allocate_lens=np.array([32, 64], dtype=np.int32),
            verify_write_lens=np.array([12, 36], dtype=np.int32),
        ),
    )
    first_candidate = client._build_same_batch_spec_chain_candidate_batch(
        base_batch,
        first_pending,
    )
    assert first_candidate is not None
    first_candidate.spec_info_padded = SimpleNamespace()

    second_pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[11]], [[31]]], dtype=np.int32),
            verified_id=np.array([111, 131], dtype=np.int32),
            new_seq_lens=jnp.array([16, 40], dtype=jnp.int32),
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_new_seq_lens_host=np.array([16, 40], dtype=np.int32),
    )

    second_candidate = client._build_same_batch_spec_chain_candidate_batch(
        first_candidate,
        second_pending,
    )

    assert second_candidate is not None
    np.testing.assert_array_equal(
        second_candidate.out_cache_loc,
        np.concatenate([req_to_token[10, 17:19], req_to_token[30, 41:43]]),
    )
    np.testing.assert_array_equal(
        second_candidate.spec_info_padded.verify_write_lens,
        np.array([19, 43], dtype=np.int32),
    )


def test_worker_same_batch_chain_candidate_rejects_layout_mismatch():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]]], dtype=np.int32),
            verified_id=np.array([110, 130], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10, 30], dtype=np.int32),
    )
    base_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=np.array([30, 10], dtype=np.int32),
        req_pool_indices=np.array([30, 10], dtype=np.int32),
    )

    assert client._build_same_batch_spec_chain_candidate_batch(base_batch, pending) is None


def test_worker_stashes_chained_verify_without_publishing_until_next_batch(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    async0 = SimpleNamespace(name="async0")
    async1 = SimpleNamespace(name="async1")
    verify0 = SimpleNamespace(name="verify0")
    verify1 = SimpleNamespace(name="verify1")
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]]], dtype=np.int32),
            verified_id=np.array([110], dtype=np.int32),
            new_seq_lens=np.array([14], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        lambda *_args, **_kwargs: pending,
    )

    class _SpecWorker:
        def __init__(self):
            self.enqueued = []

        def forward_batch_speculative_verify_phase_enqueue(self, model_worker_batch):
            self.enqueued.append(model_worker_batch)
            return async0 if len(self.enqueued) == 1 else async1

        def materialize_speculative_verify_phase(self, async_result):
            return verify0 if async_result is async0 else verify1

        def forward_batch_speculative_draft_extend_phase(self, *_args, **_kwargs):
            raise AssertionError("pending fast path should not materialize Phase B")

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.pending_same_batch_spec_chain_candidate = None
    client.input_queue = queue.Queue()
    client.output_queue = queue.Queue()
    base_batch = SimpleNamespace(
        bid=7,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=np.array([10], dtype=np.int32),
        same_batch_chain_out_cache_loc_chunks=[np.array([100], dtype=np.int32)],
        same_batch_chain_verify_write_lens=np.array([16], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([32], dtype=np.int32),
        logits_indices_selector=np.array([0], dtype=np.int32),
        req_pool_indices=np.array([10], dtype=np.int32),
        seq_lens=np.array([12], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            topk_index=np.array([[[1]]], dtype=np.int32),
            verified_id=np.array([101], dtype=np.int32),
            allocate_lens=np.array([32], dtype=np.int32),
        ),
    )
    client.input_queue.put(("spec_split", base_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    kind, payload = client.output_queue.get_nowait()
    assert kind == "spec_verify"
    assert payload is verify0
    assert client.output_queue.empty()
    assert len(client.spec_worker.enqueued) == 2
    assert client.spec_worker.enqueued[0] is base_batch
    assert client.spec_worker.enqueued[1] is not base_batch
    assert client.pending_same_batch_spec_chain_candidate.verify_async_result is async1


def test_worker_prebuilds_chain_before_phase_b_and_publishes_after_enqueue(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    events = []
    async0 = SimpleNamespace(name="async0")
    async1 = SimpleNamespace(name="async1")
    verify0 = SimpleNamespace(
        name="verify0",
        padded_new_seq_lens_host=np.array([14], dtype=np.int32),
    )
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]]], dtype=np.int32),
            verified_id=np.array([110], dtype=np.int32),
            new_seq_lens=np.array([14], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )

    def fake_dispatch(*_args, **_kwargs):
        events.append("dispatch_phase_b")
        return pending

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        fake_dispatch,
    )

    class _OutputQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            events.append(f"publish_{item[0]}")
            return super().put(item, *args, **kwargs)

    class _SpecWorker:
        def __init__(self):
            self.enqueued = []

        def forward_batch_speculative_verify_phase_enqueue(self, model_worker_batch):
            self.enqueued.append(model_worker_batch)
            events.append(f"enqueue_verify_{len(self.enqueued)}")
            return async0 if len(self.enqueued) == 1 else async1

        def materialize_speculative_verify_phase(self, async_result):
            assert async_result is async0
            events.append("materialize_phase_a")
            return verify0

        def forward_batch_speculative_draft_extend_phase(self, *_args, **_kwargs):
            raise AssertionError("pending fast path should not materialize Phase B")

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.pending_same_batch_spec_chain_candidate = None
    client.worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=np.arange(200, dtype=np.int32).reshape(1, 200)
            )
        )
    )
    client.input_queue = queue.Queue()
    client.output_queue = _OutputQueue()

    original_build = client._build_same_batch_spec_chain_candidate_batch

    def recording_build(*args, **kwargs):
        events.append("build_chain_candidate")
        return original_build(*args, **kwargs)

    client._build_same_batch_spec_chain_candidate_batch = recording_build

    base_batch = SimpleNamespace(
        bid=7,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        allow_same_batch_spec_chain=True,
        same_batch_chain_req_pool_indices=np.array([10], dtype=np.int32),
        same_batch_chain_out_cache_loc_chunks=[np.array([100], dtype=np.int32)],
        same_batch_chain_verify_write_lens=np.array([16], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([32], dtype=np.int32),
        logits_indices_selector=np.array([0], dtype=np.int32),
        req_pool_indices=np.array([10], dtype=np.int32),
        seq_lens=np.array([12], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            topk_index=np.array([[[1]]], dtype=np.int32),
            verified_id=np.array([101], dtype=np.int32),
            allocate_lens=np.array([32], dtype=np.int32),
            verify_write_lens=np.array([12], dtype=np.int32),
        ),
    )
    client.input_queue.put(("spec_split", base_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    assert events.index("materialize_phase_a") < events.index("build_chain_candidate")
    assert events.index("build_chain_candidate") < events.index("dispatch_phase_b")
    assert events.index("dispatch_phase_b") < events.index("enqueue_verify_2")
    assert events.index("enqueue_verify_2") < events.index("publish_spec_verify")


def test_worker_consumes_matching_chained_verify_candidate(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    async1 = SimpleNamespace(name="async1")
    verify1 = SimpleNamespace(name="verify1")
    pending2 = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[11]]], dtype=np.int32),
            verified_id=np.array([111], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )
    dispatch_calls = []

    def fake_dispatch(*args, **_kwargs):
        dispatch_calls.append(args)
        return pending2

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        fake_dispatch,
    )

    class _SpecWorker:
        def forward_batch_speculative_verify_phase_enqueue(self, _model_worker_batch):
            raise AssertionError("matching chained candidate should be reused")

        def materialize_speculative_verify_phase(self, async_result):
            assert async_result is async1
            return verify1

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.pending_same_batch_spec_chain_candidate = SimpleNamespace(
        req_pool_indices=np.array([10], dtype=np.int32),
        verify_async_result=async1,
        model_worker_batch=SimpleNamespace(bid=8),
    )
    client.input_queue = queue.Queue()
    client.output_queue = queue.Queue()
    model_worker_batch = SimpleNamespace(
        bid=8,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        allow_same_batch_spec_chain=False,
        req_pool_indices=np.array([10], dtype=np.int32),
        spec_info_padded=EagleDraftInput(),
    )
    client.input_queue.put(("spec_split", model_worker_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    kind, payload = client.output_queue.get_nowait()
    assert kind == "spec_verify"
    assert payload is verify1
    assert client.pending_spec_draft_extend_result is pending2
    assert client.pending_same_batch_spec_chain_candidate is None
    assert dispatch_calls


def test_worker_discards_mismatched_chained_candidate_and_launches_normal(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    async0 = SimpleNamespace(name="async0")
    verify0 = SimpleNamespace(name="verify0")
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]]], dtype=np.int32),
            verified_id=np.array([110], dtype=np.int32),
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )
    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        lambda *_args, **_kwargs: pending,
    )

    class _SpecWorker:
        def __init__(self):
            self.enqueued = []

        def forward_batch_speculative_verify_phase_enqueue(self, model_worker_batch):
            self.enqueued.append(model_worker_batch)
            return async0

        def materialize_speculative_verify_phase(self, async_result):
            assert async_result is async0
            return verify0

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.pending_same_batch_spec_chain_candidate = SimpleNamespace(
        req_pool_indices=np.array([99], dtype=np.int32),
        verify_async_result=SimpleNamespace(name="wrong"),
        model_worker_batch=SimpleNamespace(bid=99),
    )
    client.input_queue = queue.Queue()
    client.output_queue = queue.Queue()
    model_worker_batch = SimpleNamespace(
        bid=8,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        allow_same_batch_spec_chain=False,
        req_pool_indices=np.array([10], dtype=np.int32),
        spec_info_padded=EagleDraftInput(),
    )
    client.input_queue.put(("spec_split", model_worker_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    kind, payload = client.output_queue.get_nowait()
    assert kind == "spec_verify"
    assert payload is verify0
    assert client.spec_worker.enqueued == [model_worker_batch]
    assert client.pending_same_batch_spec_chain_candidate is None


def test_predispatched_phase_b_builds_pending_result_without_materialization():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    topk = np.array([[[10]], [[30]]], dtype=np.int32)
    selected_verified = np.array([110, 130], dtype=np.int32)
    previous_token_list = np.array([[10], [30]], dtype=np.int32)
    dispatch = SimpleNamespace(
        materialize_topk=False,
        topk_index_stacked=topk,
        selected_verified_id=selected_verified,
        previous_token_list=previous_token_list,
    )
    verify_result = SimpleNamespace(
        draft_extend_state=SimpleNamespace(predispatched=dispatch),
    )
    model_worker_batch = SimpleNamespace(
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
    )

    result = draft_extend_fused.spec_decode_pending_draft_extend_result_from_predispatch(
        model_worker_batch,
        verify_result,
    )

    assert isinstance(result, SpecDraftExtendPhaseResult)
    assert result.next_draft_input.topk_index is None
    assert result.next_draft_input.verified_id is None
    assert result.padded_next_draft_input.topk_index is topk
    assert result.padded_next_draft_input.verified_id is selected_verified
    assert result.padded_next_draft_input.previous_token_list is previous_token_list
    np.testing.assert_array_equal(result.req_pool_indices, np.array([10, 30], dtype=np.int32))
    np.testing.assert_array_equal(
        result.padded_req_pool_indices,
        np.array([10, 30], dtype=np.int32),
    )


def test_worker_split_verify_publishes_phase_a_then_dispatches_pending(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    topk = np.array([[[10]]], dtype=np.int32)
    selected_verified = np.array([110], dtype=np.int32)
    previous_token_list = np.array([[10]], dtype=np.int32)
    verify_async_result = SimpleNamespace()
    verify_result = SimpleNamespace()
    pending_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=topk,
            verified_id=selected_verified,
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )
    pending_result.padded_next_draft_input.previous_token_list = previous_token_list
    calls = []

    def fake_dispatch(spec_worker, model_worker_batch, verify_phase_async):
        calls.append((spec_worker, model_worker_batch, verify_phase_async))
        return pending_result

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        fake_dispatch,
    )

    class _SpecWorker:
        def forward_batch_speculative_verify_phase_enqueue(self, model_worker_batch):
            return verify_async_result

        def materialize_speculative_verify_phase(self, async_result):
            assert async_result is verify_async_result
            return verify_result

        def forward_batch_speculative_draft_extend_phase(self, *_args, **_kwargs):
            raise AssertionError("fast pending path should not materialize Phase B")

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.input_queue = queue.Queue()
    client.output_queue = queue.Queue()
    model_worker_batch = SimpleNamespace(
        bid=7,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        logits_indices_selector=np.array([0], dtype=np.int32),
        req_pool_indices=np.array([10], dtype=np.int32),
        spec_info_padded=EagleDraftInput(),
    )
    client.input_queue.put(("spec_split", model_worker_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    kind, queued_result = client.output_queue.get_nowait()
    assert kind == "spec_verify"
    assert queued_result is verify_result
    assert calls == [(client.spec_worker, model_worker_batch, verify_async_result)]
    assert client.pending_spec_draft_extend_result is pending_result


def test_worker_split_verify_relays_phase_a_host_seq_lens_before_stashing_chain(
    monkeypatch,
):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    verify_async_result = SimpleNamespace()
    host_seq_lens = np.array([14], dtype=np.int32)
    verify_result = SimpleNamespace(padded_new_seq_lens_host=host_seq_lens)
    pending_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=np.array([10], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]]], dtype=np.int32),
            verified_id=np.array([110], dtype=np.int32),
            new_seq_lens=jnp.array([14], dtype=jnp.int32),
        ),
        padded_req_pool_indices=np.array([10], dtype=np.int32),
    )
    seen = []

    monkeypatch.setattr(
        draft_extend_fused,
        "spec_decode_dispatch_draft_extend_for_pending",
        lambda *_args, **_kwargs: pending_result,
    )

    def fake_stash(self, model_worker_batch, pending):
        seen.append(getattr(pending, "padded_new_seq_lens_host", None))

    monkeypatch.setattr(
        ModelWorkerClient,
        "_stash_same_batch_spec_chain_candidate",
        fake_stash,
    )

    class _SpecWorker:
        def forward_batch_speculative_verify_phase_enqueue(self, model_worker_batch):
            return verify_async_result

        def materialize_speculative_verify_phase(self, async_result):
            assert async_result is verify_async_result
            return verify_result

        def forward_batch_speculative_draft_extend_phase(self, *_args, **_kwargs):
            raise AssertionError("fast pending path should not materialize Phase B")

    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.spec_worker = _SpecWorker()
    client.pending_spec_draft_extend_result = None
    client.input_queue = queue.Queue()
    client.output_queue = queue.Queue()
    model_worker_batch = SimpleNamespace(
        bid=7,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        logits_indices_selector=np.array([0], dtype=np.int32),
        req_pool_indices=np.array([10], dtype=np.int32),
        spec_info_padded=EagleDraftInput(),
    )
    client.input_queue.put(("spec_split", model_worker_batch, None, None, None))
    client.input_queue.put((None, None, None, None, None))

    client.forward_thread_func_()

    np.testing.assert_array_equal(seen[0], host_seq_lens)


def test_prepare_topk1_uses_relayed_previous_token_list():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    class _NoSliceTopk:
        def __getitem__(self, _):
            raise AssertionError("topk_index should not be sliced when token list is relayed")

    previous_token_list = np.array([[10], [30]], dtype=np.int32)
    spec_info = EagleDraftInput(
        topk_index=_NoSliceTopk(),
        verified_id=np.array([110, 130], dtype=np.int32),
    )
    spec_info.previous_token_list = previous_token_list
    model_worker_batch = SimpleNamespace(
        spec_info_padded=spec_info,
        seq_lens=np.array([4, 5], dtype=np.int32),
        seq_lens_sum=9,
    )
    draft_worker = SimpleNamespace(
        speculative_num_draft_tokens=1,
        speculative_num_steps=1,
        topk=1,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        padding_for_decode=lambda _mwb: None,
    )

    previous_verified_id, token_list = (
        draft_extend_fused._prepare_topk1_verify_placeholders_from_draft_state(
            draft_worker,
            model_worker_batch,
        )
    )

    np.testing.assert_array_equal(previous_verified_id, np.array([110, 130], dtype=np.int32))
    assert token_list is previous_token_list


def test_prepare_topk1_relays_device_new_seq_lens_for_next_verify():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    new_seq_lens = jax.device_put(np.array([10, 20], dtype=np.int32))
    spec_info = EagleDraftInput(
        topk_index=np.array([[[1]], [[2]]], dtype=np.int32),
        verified_id=np.array([101, 102], dtype=np.int32),
        new_seq_lens=new_seq_lens,
    )
    model_worker_batch = SimpleNamespace(
        spec_info_padded=spec_info,
        seq_lens=np.array([9, 19], dtype=np.int32),
        seq_lens_sum=28,
    )
    draft_worker = SimpleNamespace(
        speculative_num_draft_tokens=1,
        speculative_num_steps=1,
        topk=1,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        padding_for_decode=lambda _mwb: None,
        _fused_verify_placeholder_cache={},
    )

    draft_extend_fused._prepare_topk1_verify_placeholders_from_draft_state(
        draft_worker,
        model_worker_batch,
    )

    assert model_worker_batch.target_verify_seq_lens_device is new_seq_lens


def test_forward_batch_init_uses_relayed_device_seq_lens(monkeypatch):
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    seq_lens_device = jax.device_put(
        np.array([10, 20], dtype=np.int32),
        NamedSharding(mesh, P("data")),
    )
    calls = []

    def fake_preserve(value, sharding):
        calls.append(value)
        return value

    monkeypatch.setattr(draft_extend_fused, "_device_array_preserve_device", fake_preserve)
    monkeypatch.setattr(
        draft_extend_fused,
        "_cached_host_device_array_preserve_device",
        lambda _owner, _field, value, _sharding: value,
    )
    monkeypatch.setattr(
        "sgl_jax.srt.eplb.expert_location.get_global_expert_location_metadata",
        lambda: None,
    )
    batch = SimpleNamespace(
        bid=1,
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=np.array([9, 19], dtype=np.int32),
        target_verify_seq_lens_device=seq_lens_device,
        input_ids=np.array([1, 2], dtype=np.int32),
        out_cache_loc=np.array([3, 4], dtype=np.int32),
        positions=np.array([9, 19], dtype=np.int32),
        mrope_positions=None,
        req_pool_indices=np.array([5, 6], dtype=np.int32),
        cache_loc=np.array([7, 8], dtype=np.int32),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        lora_scalings=None,
        lora_token_indices=None,
        lora_ranks=None,
        spec_info_padded=None,
        spec_algorithm=None,
        capture_hidden_mode=None,
        input_embedding=None,
        apply_for_deepstack=False,
        deepstack_visual_embedding=None,
        recurrent_indices=None,
        lora_ids=None,
    )
    model_runner = SimpleNamespace(mesh=mesh, attn_backend=None)

    forward_batch = draft_extend_fused._forward_batch_init_new_preserve_device(
        batch,
        model_runner,
    )

    assert forward_batch.seq_lens is seq_lens_device
    assert any(value is seq_lens_device for value in calls)


def test_prepare_for_extend_after_verify_can_skip_host_logits_metadata(monkeypatch):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    def fail_from_model_worker_batch(*_args, **_kwargs):
        raise AssertionError("fused split path should build preserve-device logits metadata")

    monkeypatch.setattr(
        LogitsMetadata,
        "from_model_worker_batch",
        fail_from_model_worker_batch,
    )

    draft_input = EagleDraftInput(
        hidden_states=np.zeros((2, 4), dtype=np.float32),
        allocate_lens=np.array([64, 64], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        spec_info_padded=None,
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        seq_lens=np.array([10, 20], dtype=np.int32),
        positions=np.array([9, 19], dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=2,
    )
    draft_model_runner = SimpleNamespace(
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        attn_backend=SimpleNamespace(
            get_eagle_forward_metadata=lambda _batch: "metadata",
            forward_metadata=None,
        ),
    )
    batch_output = SimpleNamespace(
        accept_lens=np.array([1, 2], dtype=np.int32),
        next_draft_input=EagleDraftInput(
            hidden_states=np.zeros((2, 4), dtype=np.float32),
            verified_id=np.array([10, 11, 20, 21], dtype=np.int32),
        ),
    )

    mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_model_runner,
        batch_output,
        speculative_num_draft_tokens=4,
        build_logits_metadata=False,
    )

    assert mwb is model_worker_batch
    assert logits_metadata is None
    assert draft_model_runner.attn_backend.forward_metadata == "metadata"
    assert model_worker_batch.input_ids is batch_output.next_draft_input.verified_id
    assert model_worker_batch.forward_mode is ForwardMode.DRAFT_EXTEND


def test_verify_phase_materializes_phase_a_outputs_before_phase_b_predispatch():
    import sgl_jax.srt.speculative.draft_extend_fused as draft_extend_fused

    enqueue_source = inspect.getsource(draft_extend_fused.spec_decode_verify_phase_enqueue)
    source = inspect.getsource(draft_extend_fused.spec_decode_verify_phase)

    assert "copy_to_host_async(accept_lens_device)" in enqueue_source
    assert "copy_to_host_async(predict_device)" in enqueue_source
    assert source.index("spec_decode_materialize_verify_phase(async_result)") < source.index(
        "predispatch_spec_draft_extend_phase"
    )


def test_spec_overlap_decode_stats_are_deferred_after_kv_release():
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    source = inspect.getsource(SchedulerOutputProcessorMixin.process_batch_result_decode)

    assert "self.token_to_kv_pool_allocator.free_group_end()" in source
    assert "self._defer_decode_stats_log(batch)" in source
    assert source.index("self.token_to_kv_pool_allocator.free_group_end()") < source.index(
        "self._defer_decode_stats_log(batch)"
    )


def test_same_batch_device_chain_guard_allows_finish_risk_but_not_finished_req():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        reqs_info=[
            SimpleNamespace(
                reqs=[
                    SimpleNamespace(
                        output_ids=[1, 2, 3],
                        sampling_params=SamplingParams(
                            max_new_tokens=64,
                            ignore_eos=False,
                            stop=None,
                            stop_token_ids=None,
                        ),
                        grammar=None,
                        tokenizer=None,
                        eos_token_ids=set(),
                        finished=lambda: False,
                        is_retracted=False,
                    )
                ],
                req_pool_indices=np.array([10, -1], dtype=np.int32),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([20], dtype=np.int32),
                    verify_write_lens=np.array([16], dtype=np.int32),
                ),
            )
        ],
    )
    previous_req_pool_indices = np.array([10, -1], dtype=np.int32)

    assert scheduler._can_chain_same_batch_spec_decode(
        batch,
        previous_req_pool_indices,
    )

    batch.reqs_info[0].reqs[0].finished = lambda: True

    assert not scheduler._can_chain_same_batch_spec_decode(
        batch,
        previous_req_pool_indices,
    )


def test_same_batch_device_chain_guard_requires_same_padded_layout():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    req = SimpleNamespace(
        output_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=64, ignore_eos=True),
        grammar=None,
        tokenizer=None,
        eos_token_ids=set(),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
                req_pool_indices=np.array([10, -1]),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([20], dtype=np.int32),
                    verify_write_lens=np.array([16], dtype=np.int32),
                ),
            )
        ],
    )

    assert not scheduler._can_chain_same_batch_spec_decode(
        batch,
        np.array([10, 11], dtype=np.int32),
    )


def test_same_batch_device_chain_guard_allows_pending_phase_a_to_check_next_slack():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    req = SimpleNamespace(
        output_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=64, ignore_eos=True),
        grammar=None,
        tokenizer=None,
        eos_token_ids=set(),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
                req_pool_indices=np.array([10], dtype=np.int32),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([20], dtype=np.int32),
                    verify_write_lens=np.array([16], dtype=np.int32),
                ),
            )
        ],
    )

    assert scheduler._can_chain_same_batch_spec_decode(
        batch,
        np.array([10], dtype=np.int32),
    )

    batch.reqs_info[0].spec_info.allocate_lens = np.array([18], dtype=np.int32)

    assert scheduler._can_chain_same_batch_spec_decode(
        batch,
        np.array([10], dtype=np.int32),
    )


def test_same_batch_device_chain_guard_rejects_overwritten_current_frontier():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    req = SimpleNamespace(
        output_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=64, ignore_eos=True),
        grammar=None,
        tokenizer=None,
        eos_token_ids=set(),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        waiting_queue=[],
        pending_dp_reqs=[],
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
                req_pool_indices=np.array([10], dtype=np.int32),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([15], dtype=np.int32),
                    verify_write_lens=np.array([16], dtype=np.int32),
                ),
            )
        ],
    )

    assert not scheduler._can_chain_same_batch_spec_decode(
        batch,
        np.array([10], dtype=np.int32),
    )


def test_same_batch_device_chain_guard_rejects_waiting_or_pending_admit():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    scheduler.waiting_queue = [object()]
    scheduler.pending_dp_reqs = []
    req = SimpleNamespace(
        output_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=64, ignore_eos=True),
        grammar=None,
        tokenizer=None,
        eos_token_ids=set(),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
                req_pool_indices=np.array([10], dtype=np.int32),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([20], dtype=np.int32),
                    verify_write_lens=np.array([16], dtype=np.int32),
                ),
            )
        ],
    )

    assert not scheduler._can_chain_same_batch_spec_decode(batch, np.array([10], dtype=np.int32))

    scheduler.waiting_queue = []
    scheduler.pending_dp_reqs = [object()]

    assert not scheduler._can_chain_same_batch_spec_decode(batch, np.array([10], dtype=np.int32))


def test_scheduler_attaches_same_batch_chain_preview_when_guard_allows():
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    original_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.enable_overlap = True
    scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
    scheduler.waiting_queue = []
    scheduler.pending_dp_reqs = []
    req_to_token = np.zeros((1, 32), dtype=np.int32)
    req_to_token[0, :] = np.arange(100, 132, dtype=np.int32)
    req = SimpleNamespace(
        output_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=64, ignore_eos=True),
        grammar=None,
        tokenizer=None,
        eos_token_ids=set(),
        finished=lambda: False,
        is_retracted=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_grammar=False,
        batch_size=lambda: 1,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
                seq_lens=np.array([10], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
                spec_info=EagleDraftInput(
                    allocate_lens=np.array([16], dtype=np.int32),
                    verify_write_lens=np.array([12], dtype=np.int32),
                ),
            )
        ],
    )
    model_worker_batch = SimpleNamespace(req_pool_indices=np.array([0], dtype=np.int32))

    try:
        scheduler._attach_same_batch_spec_chain_preview(batch, model_worker_batch)
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = original_alloc_len

    assert model_worker_batch.allow_same_batch_spec_chain
    np.testing.assert_array_equal(
        model_worker_batch.same_batch_chain_out_cache_loc_chunks[0],
        np.array([112], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_worker_batch.same_batch_chain_verify_write_lens,
        np.array([13], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_worker_batch.same_batch_chain_allocate_lens,
        np.array([16], dtype=np.int32),
    )


def test_scheduler_disables_same_batch_chain_preview_when_guard_rejects():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = None
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
    model_worker_batch = SimpleNamespace(req_pool_indices=np.array([0], dtype=np.int32))

    scheduler._attach_same_batch_spec_chain_preview(batch, model_worker_batch)

    assert not model_worker_batch.allow_same_batch_spec_chain
    assert model_worker_batch.same_batch_chain_out_cache_loc_chunks is None
    assert model_worker_batch.same_batch_chain_verify_write_lens is None
    assert model_worker_batch.same_batch_chain_allocate_lens is None


def test_check_finished_caches_tokenizer_stop_attrs():
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    class _Tokenizer:
        def __init__(self):
            self.eos_calls = 0
            self.additional_calls = 0

        @property
        def eos_token_id(self):
            self.eos_calls += 1
            return 2

        @property
        def additional_stop_token_ids(self):
            self.additional_calls += 1
            return None

    tokenizer = _Tokenizer()
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=[1],
        sampling_params=SamplingParams(max_new_tokens=16),
        eos_token_ids=set(),
    )
    req.tokenizer = tokenizer
    req.output_ids = [10]

    req.check_finished()
    req.output_ids.append(11)
    req.check_finished()

    assert tokenizer.eos_calls == 1
    assert tokenizer.additional_calls == 1
    assert req.finished_reason is None


def test_worker_falls_back_to_padded_phase_b_rows_when_layout_changes():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    client.pending_spec_draft_extend_result = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(
            topk_index=None,
            topk_p=None,
            hidden_states=None,
            verified_id=np.array([110, 130], dtype=np.int32),
            allocate_lens=np.array([64, 96], dtype=np.int32),
            new_seq_lens=np.array([14, 38], dtype=np.int32),
        ),
        req_pool_indices=np.array([10, 30], dtype=np.int32),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[10]], [[30]], [[-3]]], dtype=np.int32)
        ),
        padded_req_pool_indices=np.array([10, 30, -1], dtype=np.int32),
    )
    spec_info = EagleDraftInput(
        topk_index=np.array([[[-1]], [[-2]]], dtype=np.int32),
        topk_p=np.zeros((2, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((2, 2), dtype=np.float32),
        verified_id=np.array([-1, -2], dtype=np.int32),
        allocate_lens=np.array([1, 2], dtype=np.int32),
        new_seq_lens=np.array([4, 5], dtype=np.int32),
    )
    model_worker_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=np.array([30, 10], dtype=np.int32),
        spec_info_padded=spec_info,
    )

    client._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    np.testing.assert_array_equal(
        spec_info.topk_index,
        np.array([[[30]], [[10]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(spec_info.verified_id, np.array([130, 110]))
    assert client.pending_spec_draft_extend_result is None


def test_worker_keeps_pending_phase_b_state_for_non_decode_batch():
    client = ModelWorkerClient.__new__(ModelWorkerClient)
    pending = SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(verified_id=np.array([110], dtype=np.int32)),
        req_pool_indices=np.array([10], dtype=np.int32),
    )
    client.pending_spec_draft_extend_result = pending
    model_worker_batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))

    client._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    assert client.pending_spec_draft_extend_result is pending


def test_fused_greedy_draft_state_requires_topk_and_verified_id():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = BaseSpecWorker.__new__(BaseSpecWorker)
    batch_without_topk = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            verified_id=np.array([1], dtype=np.int32),
            topk_index=None,
        )
    )
    batch_with_state = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            verified_id=np.array([1], dtype=np.int32),
            topk_index=np.array([[[2]]], dtype=np.int32),
        )
    )

    assert not worker._has_fused_greedy_draft_state(batch_without_topk)
    assert worker._has_fused_greedy_draft_state(batch_with_state)


def test_draft_extend_allocate_lens_accepts_dp_padded_layout():
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        _select_draft_extend_allocate_lens,
    )

    sel = np.array([0, 4, 8, 12], dtype=np.int32)
    padded = np.arange(32, dtype=np.int32) + 100
    real = np.array([100, 104, 108, 112], dtype=np.int32)

    np.testing.assert_array_equal(
        _select_draft_extend_allocate_lens(padded, sel, total_slots=32),
        real,
    )
    np.testing.assert_array_equal(
        _select_draft_extend_allocate_lens(real, sel, total_slots=32),
        real,
    )
