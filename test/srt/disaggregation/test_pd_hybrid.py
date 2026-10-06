"""CPU unit tests for PD disaggregation of hybrid (linear-recurrent + KV) models.

Covers RFC "PD disaggregation for hybrid models" §5.1:

* DP-aware recurrent slot gather (prefill) / scatter (decode) on a real
  ``RecurrentStatePool`` with ``dp_size=1`` and ``dp_size=2``.
* The "skip the last prompt token on P" contract (``pd_transfer_prompt_len``).
* Decode admission / release / abort ownership of the recurrent slot.
* ``HybridReqToTokenPool.alloc`` adopting a pre-owned slot.
* JAX D2H staging and Raiden child reads for the dual (KV + state) payload.

Run:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python -m pytest -q test/srt/disaggregation/test_pd_hybrid.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import types  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll  # noqa: E402
from sgl_jax.srt.disaggregation.base.transfer import (  # noqa: E402
    DecodeAdmission,
    validate_transfer_payload,
)
from sgl_jax.srt.disaggregation.common.recurrent_state import (  # noqa: E402
    CONV_STATE_KEY,
    RECURRENT_STATE_KEY,
    build_recurrent_state_specs,
    gather_recurrent_slot,
    global_recurrent_slot,
    pd_transfer_prompt_len,
    scatter_recurrent_slot,
)
from sgl_jax.srt.disaggregation.decode import (  # noqa: E402
    DecodeBookkeeping,
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sgl_jax.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool  # noqa: E402
from sgl_jax.srt.mem_cache.recurrent_state_pool import (  # noqa: E402
    RecurrentStatePool,
    get_recurrent_state_spec,
)

# Small GDN-like shapes: proj_size = H*D + 2*H*D = 24.
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4
CONV_KERNEL = 4
POOL_SIZE = 6  # global valid slots; dp=2 -> 3 per rank, 4 buffer slots per rank


def _need_devices(n: int) -> None:
    if len(jax.devices()) < n:
        pytest.skip(f"needs {n} devices (set XLA_FLAGS=--xla_force_host_platform_device_count)")


def _mesh(dp: int, tp: int = 1) -> Mesh:
    _need_devices(dp * tp)
    devices = np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp)
    # Match production (mesh_utils.create_device_mesh, use_explicit_sharding=True):
    # the slot gather uses ``.at[].get(out_sharding=...)``, which needs Explicit axes.
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def _make_pool(dp: int, tp: int = 1) -> RecurrentStatePool:
    return RecurrentStatePool(
        linear_recurrent_layer_ids=list(range(NUM_LAYERS)),
        size=POOL_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        mesh=_mesh(dp, tp),
        dp_size=dp,
        temporal_dtype=jnp.float32,
        conv_dtype=jnp.bfloat16,
    )


def _rec_pattern(shape, layer: int) -> np.ndarray:
    # Unique per element and layer; exact in float32.
    return (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + 1.0) + layer * 1e4


def _conv_pattern(shape, layer: int) -> np.ndarray:
    # Constant per (slot, layer); small integers are exact in bfloat16.
    slots = np.arange(shape[0], dtype=np.float32).reshape(-1, 1, 1)
    return np.broadcast_to(slots * 8 + layer + 1, shape).astype(np.float32)


def _fill_pool(rsp: RecurrentStatePool) -> None:
    for layer, buf in enumerate(rsp.recurrent_buffers):
        host = _rec_pattern(buf.shape, layer)
        rsp.recurrent_buffers[layer] = jax.device_put(
            jnp.asarray(host, dtype=buf.dtype), rsp.recurrent_sharding
        )
    for layer, inner in enumerate(rsp.conv_buffers):
        for i, cbuf in enumerate(inner):
            host = _conv_pattern(cbuf.shape, layer)
            inner[i] = jax.device_put(jnp.asarray(host, dtype=cbuf.dtype), rsp.conv_sharding)


def _np(x) -> np.ndarray:
    return np.asarray(jax.device_get(x)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt_len", "is_hybrid", "expected"),
    [
        (1, True, 1),  # 1-token prompt: D's extend has no prefix, state unused
        (2, True, 1),
        (17, True, 16),  # L-1 is a page multiple (page_size=16): one page fewer
        (100, True, 99),
        (100, False, 100),
        (1, False, 1),
    ],
)
def test_pd_transfer_prompt_len(prompt_len, is_hybrid, expected):
    assert pd_transfer_prompt_len(prompt_len, is_hybrid) == expected


def test_global_recurrent_slot_maps_rank_local_slots():
    rsp = SimpleNamespace(dp_size=2, slots_per_rank=3)
    # Guide example: slots_per_rank=3 -> 4 buffer slots per rank.
    assert global_recurrent_slot(rsp, 1, 0) == 1
    assert global_recurrent_slot(rsp, 2, 1) == 6
    # dp_rank=None is treated as rank 0 (dp_size=1 requests).
    assert global_recurrent_slot(SimpleNamespace(dp_size=1, slots_per_rank=3), 3, None) == 3
    with pytest.raises(ValueError):
        global_recurrent_slot(rsp, 0, 0)  # dummy slot is never a request slot


# ---------------------------------------------------------------------------
# Gather / scatter on a real RecurrentStatePool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dp", [1, 2])
def test_gather_returns_the_requested_global_slot(dp):
    rsp = _make_pool(dp)
    _fill_pool(rsp)
    for dp_rank in range(dp):
        for local_slot in range(1, rsp.slots_per_rank + 1):
            g = global_recurrent_slot(rsp, local_slot, dp_rank)
            out = gather_recurrent_slot(rsp, g)
            assert set(out) == {RECURRENT_STATE_KEY, CONV_STATE_KEY}
            assert len(out[RECURRENT_STATE_KEY]) == NUM_LAYERS
            assert len(out[CONV_STATE_KEY]) == NUM_LAYERS
            for layer in range(NUM_LAYERS):
                rec = out[RECURRENT_STATE_KEY][layer]
                assert rec.shape == (NUM_HEADS, HEAD_DIM, HEAD_DIM)
                np.testing.assert_array_equal(
                    _np(rec), _rec_pattern(rsp.recurrent_buffers[layer].shape, layer)[g]
                )
                conv = out[CONV_STATE_KEY][layer][0]
                assert conv.shape == (rsp.proj_size, CONV_KERNEL - 1)
                assert conv.dtype == jnp.bfloat16
                np.testing.assert_array_equal(
                    _np(conv), _conv_pattern(rsp.conv_buffers[layer][0].shape, layer)[g]
                )


def test_build_recurrent_state_specs_match_gathered_leaves():
    rsp = _make_pool(dp=2)
    specs = build_recurrent_state_specs(rsp)
    gathered = gather_recurrent_slot(rsp, global_recurrent_slot(rsp, 1, 1))
    for spec, leaf in zip(specs[RECURRENT_STATE_KEY], gathered[RECURRENT_STATE_KEY]):
        assert (spec.shape, spec.dtype) == (leaf.shape, leaf.dtype)
        assert spec.sharding.spec == leaf.sharding.spec
    for spec_inner, leaf_inner in zip(specs[CONV_STATE_KEY], gathered[CONV_STATE_KEY]):
        for spec, leaf in zip(spec_inner, leaf_inner):
            assert (spec.shape, spec.dtype) == (leaf.shape, leaf.dtype)


@pytest.mark.parametrize(
    ("dp", "p_rank", "p_slot", "d_rank", "d_slot"),
    [
        (1, 0, 2, 0, 3),
        (2, 0, 1, 1, 2),  # cross-rank: P rank 0 -> D rank 1
        (2, 1, 3, 0, 1),
        (2, 1, 2, 1, 2),
    ],
)
def test_scatter_writes_only_the_target_slot(dp, p_rank, p_slot, d_rank, d_slot):
    p_pool = _make_pool(dp)
    _fill_pool(p_pool)
    d_pool = _make_pool(dp)  # zero-initialized

    p_global = global_recurrent_slot(p_pool, p_slot, p_rank)
    d_global = global_recurrent_slot(d_pool, d_slot, d_rank)
    payload = gather_recurrent_slot(p_pool, p_global)
    scatter_recurrent_slot(d_pool, d_global, payload[RECURRENT_STATE_KEY], payload[CONV_STATE_KEY])

    for layer in range(NUM_LAYERS):
        got = _np(d_pool.recurrent_buffers[layer])
        src = _rec_pattern(p_pool.recurrent_buffers[layer].shape, layer)
        np.testing.assert_array_equal(got[d_global], src[p_global])
        others = np.delete(got, d_global, axis=0)
        assert not others.any(), "scatter corrupted another slot or DP rank"
        # Sharding of the pool buffer is preserved by the in-place scatter.
        assert d_pool.recurrent_buffers[layer].sharding == d_pool.recurrent_sharding

        got_conv = _np(d_pool.conv_buffers[layer][0])
        src_conv = _conv_pattern(p_pool.conv_buffers[layer][0].shape, layer)
        np.testing.assert_array_equal(got_conv[d_global], src_conv[p_global])
        assert not np.delete(got_conv, d_global, axis=0).any()


def test_scatter_rejects_layer_count_mismatch():
    rsp = _make_pool(dp=1)
    payload = gather_recurrent_slot(rsp, 1)
    with pytest.raises(ValueError, match="layer count mismatch"):
        scatter_recurrent_slot(rsp, 1, payload[RECURRENT_STATE_KEY][:1], payload[CONV_STATE_KEY])
    with pytest.raises(ValueError, match="expects 1 buffers"):
        scatter_recurrent_slot(
            rsp,
            1,
            payload[RECURRENT_STATE_KEY],
            [list(inner) * 2 for inner in payload[CONV_STATE_KEY]],
        )


# ---------------------------------------------------------------------------
# Prefill scheduler: extraction + payload
# ---------------------------------------------------------------------------


class _PrefillSched:
    _extract_req_recurrent_state = SchedulerDisaggregationPrefillMixin._extract_req_recurrent_state
    _prefill_recurrent_state_pool = (
        SchedulerDisaggregationPrefillMixin._prefill_recurrent_state_pool
    )
    _prefill_recurrent_transfer_info = (
        SchedulerDisaggregationPrefillMixin._prefill_recurrent_transfer_info
    )
    _build_prefill_payload = SchedulerDisaggregationPrefillMixin._build_prefill_payload
    _prefill_transfer_len = SchedulerDisaggregationPrefillMixin._prefill_transfer_len

    def __init__(self, rsp, *, overlap=False):
        self.req_to_token_pool = SimpleNamespace(recurrent_state_pool=rsp)
        self.server_args = SimpleNamespace(disaggregation_enable_overlap_schedule=overlap)
        self.cur_batch = None

    def _extract_req_kv(self, req):
        return [jnp.zeros((1, 2, 2, 2), dtype=jnp.bfloat16)]


def _req(rid="r", *, dp_rank=0, slot=None, prompt_len=8, req_pool_idx=None):
    return SimpleNamespace(
        rid=rid,
        dp_rank=dp_rank,
        recurrent_pool_idx=slot,
        req_pool_idx=req_pool_idx,
        origin_input_ids=list(range(prompt_len)),
        pd_prefill_skip_last_token=False,
    )


@pytest.mark.parametrize(("dp", "dp_rank", "slot"), [(1, 0, 2), (2, 0, 3), (2, 1, 1)])
def test_extract_req_recurrent_state_uses_rank_local_slot(dp, dp_rank, slot):
    rsp = _make_pool(dp)
    _fill_pool(rsp)
    sched = _PrefillSched(rsp)

    out = sched._extract_req_recurrent_state(_req(dp_rank=dp_rank, slot=slot))

    g = slot + dp_rank * (rsp.slots_per_rank + 1)
    for layer in range(NUM_LAYERS):
        np.testing.assert_array_equal(
            _np(out[RECURRENT_STATE_KEY][layer]),
            _rec_pattern(rsp.recurrent_buffers[layer].shape, layer)[g],
        )


def test_extract_req_recurrent_state_waits_for_overlap_fence():
    rsp = _make_pool(dp=1)
    sched = _PrefillSched(rsp, overlap=True)
    sched.cur_batch = SimpleNamespace(launch_done=mock.Mock())

    sched._extract_req_recurrent_state(_req(slot=1))

    sched.cur_batch.launch_done.wait.assert_called_once()


def test_extract_req_recurrent_state_requires_a_slot():
    sched = _PrefillSched(_make_pool(dp=1))
    with pytest.raises(RuntimeError, match="no recurrent slot"):
        sched._extract_req_recurrent_state(_req(slot=None))


def test_prefill_payload_is_dual_and_validates_against_spec():
    rsp = _make_pool(dp=2)
    sched = _PrefillSched(rsp)
    req = _req(dp_rank=1, slot=2)

    slot, spec = sched._prefill_recurrent_transfer_info(req)
    payload = sched._build_prefill_payload(req)

    assert slot == 2
    assert spec == get_recurrent_state_spec(rsp)
    assert set(payload) == {"kv", RECURRENT_STATE_KEY, CONV_STATE_KEY}
    validate_transfer_payload(payload, recurrent_state_spec=spec)


def test_prefill_payload_is_kv_only_for_non_hybrid_models():
    sched = _PrefillSched(None)
    assert sched._prefill_recurrent_transfer_info(_req(slot=None)) == (None, None)
    assert set(sched._build_prefill_payload(_req())) == {"kv"}


def test_prefill_transfer_len_skips_last_token_only_when_flagged():
    sched = _PrefillSched(None)
    req = _req(prompt_len=33)
    assert sched._prefill_transfer_len(req) == 33
    req.pd_prefill_skip_last_token = True
    assert sched._prefill_transfer_len(req) == 32


def test_pd_prefill_req_fill_ids_excludes_last_prompt_token():
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    req = Req("pd-hybrid", "", [11, 12, 13, 14, 15], SamplingParams())
    req.bootstrap_room = 7
    req.init_next_round_input()
    assert req.fill_ids == [11, 12, 13, 14, 15]

    req.pd_prefill_skip_last_token = True
    req.init_next_round_input()
    assert req.fill_ids == [11, 12, 13, 14]
    assert req.extend_input_len == 4
    # The original prompt is untouched (D still sees the full prompt).
    assert req.origin_input_ids == [11, 12, 13, 14, 15]


# ---------------------------------------------------------------------------
# HybridReqToTokenPool: pre-owned slot adoption
# ---------------------------------------------------------------------------


class _FakeStatePool:
    def __init__(self, size):
        self.size = size

    def clear(self):
        pass


def _hybrid_pool(size=6, dp=1):
    return HybridReqToTokenPool(
        size=8,
        max_context_len=32,
        dtype=np.int32,
        recurrent_state_pool=_FakeStatePool(size),
        dp_size=dp,
    )


def _pool_req(dp_rank=0, slot=None):
    return SimpleNamespace(
        req_pool_idx=None,
        is_chunked=0,
        kv_committed_len=0,
        dp_rank=dp_rank,
        recurrent_pool_idx=slot,
        recurrent_ping_pong_track_buffer=None,
        recurrent_next_track_idx=None,
        recurrent_last_track_seqlen=None,
    )


def test_hybrid_pool_alloc_adopts_pre_owned_slot_and_free_returns_it():
    pool = _hybrid_pool(size=6, dp=2)
    slot = pool.alloc_recurrent_slot(1)  # PD decode admission
    req = _pool_req(dp_rank=1, slot=slot)
    free_before = list(pool.recurrent_free_slots[1])

    pool.alloc([req])

    assert req.recurrent_pool_idx == slot
    assert pool.req_index_to_recurrent_index_mapping[req.req_pool_idx] == slot
    assert pool.recurrent_free_slots[1] == free_before  # no second slot consumed

    pool.free(req)
    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[1]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Decode scheduler: admission, release, install
# ---------------------------------------------------------------------------


class _Alloc:
    page_size = 1

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.used = 0
        self.freed = []

    def available_size(self, dp_rank=0):
        return self.capacity - self.used

    def alloc(self, n, dp_rank=0):
        if n > self.available_size(dp_rank):
            return None
        self.used += n
        return list(range(n))

    def free(self, indices, dp_rank=0):
        self.freed.append((list(indices), dp_rank))
        self.used -= len(indices)


class _Receiver:
    def __init__(self, state=KVPoll.TRANSFERRING):
        self.state = state

    def poll(self):
        return self.state

    def commit(self, install):
        install({})


class _KVMgr:
    def __init__(self, mode="admit"):
        self.mode = mode
        self.contexts = []

    def poll_decode_metadata(self, context):
        return True

    def try_start_decode(self, context):
        self.contexts.append(context)
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "defer":
            return DecodeAdmission.deferred("busy")
        return DecodeAdmission.admitted(_Receiver())


_D = SchedulerDisaggregationDecodeMixin


class _DecodeSched:
    _admit_decode_prealloc = _D._admit_decode_prealloc
    _expire_decode_prealloc = _D._expire_decode_prealloc
    _drain_decode_transfer_terminals = _D._drain_decode_transfer_terminals
    _drain_transfer_queue_synced = _D._drain_transfer_queue_synced
    _release_decode_kv_indices = _D._release_decode_kv_indices
    _decode_recurrent_state_pool = _D._decode_recurrent_state_pool
    _decode_transfer_len = _D._decode_transfer_len
    _alloc_decode_recurrent_slot = _D._alloc_decode_recurrent_slot
    _release_decode_recurrent_slot = _D._release_decode_recurrent_slot
    _build_transfer_spec_for_req = _D._build_transfer_spec_for_req
    _write_recurrent_state_to_pool = _D._write_recurrent_state_to_pool

    def __init__(self, req_to_token_pool, *, dp_size=1, mode="admit"):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = _Alloc()
        self.dp_size = dp_size
        self.server_args = SimpleNamespace(
            disaggregation_num_reserved_decode_tokens=0,
            disaggregation_max_inflight_transfers=0,
            disaggregation_pull_timeout_seconds=30.0,
            enable_request_time_stats_logging=False,
        )
        self.running_batch = SimpleNamespace(reqs=[])
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self.disagg_transfer_queue = DecodeTransferQueue()
        self.disagg_kv_manager = _KVMgr(mode)
        self.aborted = []
        self.failures = []
        self.installed = []
        self.recorded_bytes = []

    def _build_kv_spec_for_req(self, req):
        return ["kv-spec"]

    def _pd_mark_time(self, req, name):
        pass

    def _record_decode_transfer_failure(self, reason):
        self.failures.append(reason)

    def _abort_decode_request(self, req, reason, cleanup_transfer=True):
        self.aborted.append((req.rid, reason))

    def _install_received_kv(self, req, kv_indices, payload):
        self.installed.append(req.rid)

    def _set_decode_bookkeeping(self, req, kv_indices):
        pass

    def _enqueue_for_decode(self, req):
        pass

    def _record_decode_transfer_bytes(self, tree):
        self.recorded_bytes.append(tree)


def _adm_req(rid, *, dp_rank=0, prompt_len=8):
    return SimpleNamespace(
        rid=rid,
        disagg_transfer_id=None,
        bootstrap_room=1,
        origin_input_ids=list(range(prompt_len)),
        dp_rank=dp_rank,
        disagg_prefill_dp_rank=dp_rank,
        recurrent_pool_idx=None,
        req_pool_idx=None,
        pd_time_stats=None,
    )


def _enqueue(sched, req):
    entry = DecodeBookkeeping(
        req_id=req.rid,
        req=req,
        p_info={"host": "1.2.3.4", "transfer_port": 1, "side_channel_port": 2},
    )
    sched.disagg_prealloc_queue.add(entry)
    return entry


def test_admission_allocates_kv_pages_and_recurrent_slot_together():
    pool = _hybrid_pool(size=6, dp=2)
    sched = _DecodeSched(pool, dp_size=2)
    req = _enqueue(sched, _adm_req("a", dp_rank=1, prompt_len=9)).req

    with mock.patch(
        "sgl_jax.srt.disaggregation.common.recurrent_state.build_recurrent_state_specs",
        return_value={RECURRENT_STATE_KEY: ["r"], CONV_STATE_KEY: [["c"]]},
    ):
        sched._admit_decode_prealloc()
        ctx = sched.disagg_kv_manager.contexts[0]
        spec = ctx.spec_factory()

    assert len(sched.disagg_transfer_queue) == 1
    assert sched.token_to_kv_pool_allocator.used == 9  # D allocates the full prompt
    assert req.recurrent_pool_idx == 1
    assert pool.recurrent_free_slots[1] == [2, 3]
    assert pool.recurrent_free_slots[0] == [1, 2, 3]  # other rank untouched
    assert ctx.recurrent_slot_idx == 1
    assert ctx.decode_dp_rank == 1
    # P skipped the last token, so only L-1 tokens' KV is transferred.
    assert ctx.prompt_tokens == 8
    assert spec == {"kv": ["kv-spec"], RECURRENT_STATE_KEY: ["r"], CONV_STATE_KEY: [["c"]]}


def test_admission_blocks_rank_when_recurrent_slots_are_exhausted():
    pool = _hybrid_pool(size=2, dp=1)  # 2 slots
    sched = _DecodeSched(pool)
    for rid in ("a", "b", "c"):
        _enqueue(sched, _adm_req(rid))

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 2
    assert [e.req_id for e in sched.disagg_prealloc_queue.items_fifo()] == ["c"]
    # The blocked request's KV pages were returned; no abort.
    assert sched.token_to_kv_pool_allocator.used == 16
    assert sched.aborted == []
    assert pool.recurrent_free_slots[0] == []


@pytest.mark.parametrize(("mode", "aborted"), [("raise", True), ("defer", False)])
def test_failed_or_deferred_admission_returns_kv_and_slot(mode, aborted):
    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool, mode=mode)
    req = _enqueue(sched, _adm_req("a")).req

    sched._admit_decode_prealloc()

    assert len(sched.disagg_transfer_queue) == 0
    assert sched.token_to_kv_pool_allocator.used == 0
    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[0]) == [1, 2, 3, 4, 5, 6]
    assert (sched.aborted == [("a", "receiver_init")]) is aborted


def test_non_hybrid_admission_does_not_touch_recurrent_state():
    sched = _DecodeSched(SimpleNamespace())  # plain pool: no recurrent_state_pool
    req = _enqueue(sched, _adm_req("a", prompt_len=9)).req

    sched._admit_decode_prealloc()

    ctx = sched.disagg_kv_manager.contexts[0]
    assert ctx.recurrent_slot_idx is None
    assert ctx.prompt_tokens == 9
    assert ctx.spec_factory() == ["kv-spec"]
    assert req.recurrent_pool_idx is None


def _admitted_entry(sched, rid="a"):
    req = _enqueue(sched, _adm_req(rid)).req
    sched._admit_decode_prealloc()
    entry = sched.disagg_transfer_queue._entries[rid]
    return req, entry


def test_terminal_failure_releases_kv_and_recurrent_slot():
    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req, entry = _admitted_entry(sched)
    entry.receiver.state = KVPoll.FAILED

    sched._drain_decode_transfer_terminals()

    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[0]) == [1, 2, 3, 4, 5, 6]
    assert sched.token_to_kv_pool_allocator.used == 0
    assert sched.aborted == [("a", "receiver_terminal_failed")]


def test_cancelled_transfer_releases_slot_only_once_terminal():
    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req, entry = _admitted_entry(sched)
    sched.disagg_transfer_queue.cancel_matching("a", abort_all=False)

    sched._drain_decode_transfer_terminals()  # still TRANSFERRING: engine may write
    assert req.recurrent_pool_idx == 1

    entry.receiver.state = KVPoll.FAILED
    sched._drain_decode_transfer_terminals()
    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[0]) == [1, 2, 3, 4, 5, 6]


def test_success_keeps_slot_for_the_request_row():
    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req, entry = _admitted_entry(sched)
    entry.receiver.state = KVPoll.SUCCESS

    sched._drain_decode_transfer_terminals()

    assert sched.installed == ["a"]
    assert req.recurrent_pool_idx == 1  # adopted later by HybridReqToTokenPool.alloc
    assert 1 not in pool.recurrent_free_slots[0]


def test_release_is_a_noop_once_the_request_owns_a_row():
    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req = _adm_req("a")
    req.recurrent_pool_idx = pool.alloc_recurrent_slot(0)
    req.req_pool_idx = 3  # HybridReqToTokenPool.free owns the slot now

    sched._release_decode_recurrent_slot(req)

    assert req.recurrent_pool_idx == 1
    assert 1 not in pool.recurrent_free_slots[0]


def _abort_scheduler(sched, waiting):
    """Bind ``Scheduler.abort_request`` to the decode fake with empty other queues."""

    from sgl_jax.srt.managers.scheduler import Scheduler

    sched.waiting_queue = list(waiting)
    sched.grammar_queue = []
    sched.running_batch = SimpleNamespace(reqs=[], reqs_info=[])
    sched.cur_batch = None
    sched.disagg_prefill_queue = None
    sched._comm_backend = None
    sched.send_to_tokenizer = SimpleNamespace(send_pyobj=mock.Mock())
    sched._engine_paused = False
    sched._sync_chunked_req_owners = lambda: None
    sched._mark_pending_chunked_aborts = lambda recv_req: None
    return lambda recv_req: Scheduler.abort_request(sched, recv_req)


def test_scheduler_abort_of_waiting_decode_request_returns_slot():
    from sgl_jax.srt.managers.io_struct import AbortReq

    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req = _adm_req("waiting")
    req.recurrent_pool_idx = pool.alloc_recurrent_slot(0)  # received, not yet scheduled
    abort = _abort_scheduler(sched, [req])

    abort(AbortReq(rid="waiting"))

    assert sched.waiting_queue == []
    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[0]) == [1, 2, 3, 4, 5, 6]


def test_scheduler_abort_of_prealloc_entry_returns_slot():
    from sgl_jax.srt.managers.io_struct import AbortReq

    pool = _hybrid_pool(size=6, dp=1)
    sched = _DecodeSched(pool)
    req = _enqueue(sched, _adm_req("pre")).req
    req.recurrent_pool_idx = pool.alloc_recurrent_slot(0)
    abort = _abort_scheduler(sched, [])

    abort(AbortReq(rid="pre"))

    assert len(sched.disagg_prealloc_queue) == 0
    assert req.recurrent_pool_idx is None
    assert sorted(pool.recurrent_free_slots[0]) == [1, 2, 3, 4, 5, 6]
    assert sched.aborted == [("pre", "abort_request")]


@pytest.mark.parametrize(("dp", "d_rank", "d_slot"), [(1, 0, 2), (2, 1, 3)])
def test_decode_writes_received_state_into_its_rank_slot(dp, d_rank, d_slot):
    p_pool = _make_pool(dp)
    _fill_pool(p_pool)
    d_pool = _make_pool(dp)
    sched = _DecodeSched(SimpleNamespace(recurrent_state_pool=d_pool), dp_size=dp)
    req = _adm_req("a", dp_rank=d_rank)
    req.recurrent_pool_idx = d_slot

    payload = {"kv": [], **gather_recurrent_slot(p_pool, 1)}
    sched._write_recurrent_state_to_pool(req, payload)

    g = global_recurrent_slot(d_pool, d_slot, d_rank)
    got = _np(d_pool.recurrent_buffers[0])
    np.testing.assert_array_equal(got[g], _rec_pattern(p_pool.recurrent_buffers[0].shape, 0)[1])
    assert not np.delete(got, g, axis=0).any()
    assert len(sched.recorded_bytes) == 1


def test_decode_install_rejects_payload_without_recurrent_state():
    d_pool = _make_pool(1)
    sched = _DecodeSched(SimpleNamespace(recurrent_state_pool=d_pool))
    req = _adm_req("a")
    req.recurrent_pool_idx = 1
    with pytest.raises(RuntimeError, match="no recurrent state"):
        sched._write_recurrent_state_to_pool(req, {"kv": []})


# ---------------------------------------------------------------------------
# JAX transfer engine: D2H staging of the dual payload
# ---------------------------------------------------------------------------


def test_jax_d2h_stages_kv_in_host_pool_and_state_separately(monkeypatch):
    from sgl_jax.srt.disaggregation.jax_transfer import conn as jconn

    registered = {}
    wrapper = SimpleNamespace(
        register_pull=lambda sub, tree: registered.__setitem__(sub, tree),
        release=lambda sub: registered.pop(sub, None),
    )
    host_pool = mock.Mock()
    host_pool.copy_from_device.side_effect = lambda tree, buffer_id: SimpleNamespace(
        array_pytree=("host-kv", tree, buffer_id)
    )
    pinned = []
    monkeypatch.setattr(
        jconn, "_copy_pytree_to_pinned_host", lambda tree: pinned.append(tree) or ("pinned", tree)
    )
    mgr = object.__new__(jconn.JaxTransferKVManager)
    mgr._wrapper = wrapper
    mgr._host_pool = host_pool

    payload = {"kv": ["kv0"], RECURRENT_STATE_KEY: ["r0"], CONV_STATE_KEY: [["c0"]]}
    status = mgr.producer_handoff("u", payload, use_d2h_staging=True, buffer_id=5)

    host_pool.copy_from_device.assert_called_once_with(["kv0"], 5)
    assert pinned == [["r0"], [["c0"]]]
    assert registered == {
        "u:kv": ("host-kv", ["kv0"], 5),
        f"u:{RECURRENT_STATE_KEY}": ("pinned", ["r0"]),
        f"u:{CONV_STATE_KEY}": ("pinned", [["c0"]]),
    }
    assert set(status.sub_uuids) == set(registered)


def test_copy_pytree_to_pinned_host_preserves_values_and_spec():
    from sgl_jax.srt.disaggregation.jax_transfer.conn import _copy_pytree_to_pinned_host

    rsp = _make_pool(dp=1)
    _fill_pool(rsp)
    tree = gather_recurrent_slot(rsp, 1)
    try:
        staged = _copy_pytree_to_pinned_host(tree)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"pinned_host staging unsupported on this backend: {exc}")
    for src, dst in zip(jax.tree.leaves(tree), jax.tree.leaves(staged)):
        assert dst.sharding.spec == src.sharding.spec
        np.testing.assert_array_equal(_np(dst), _np(src))


# ---------------------------------------------------------------------------
# Raiden transfer engine: recurrent child reads
# ---------------------------------------------------------------------------

_GROUPS = (RECURRENT_STATE_KEY, CONV_STATE_KEY)


class _FakeBootstrap:
    def __init__(self):
        self.registered = []
        self.popped = []

    def register_transfer(self, *args, **kwargs):
        self.registered.append((args, kwargs))

    def pop_transfer(self, room, **kwargs):
        self.popped.append((room, kwargs))

    def get_transfer_info(self, _room, **_kwargs):
        return None


class _FakeRaiden:
    control_port = 7777
    dp_size = 1

    def __init__(self, groups=_GROUPS):
        self.recurrent_groups = tuple(groups)
        self.endpoints_by_dp_rank = {0: [{"endpoint": "10.0.0.1:7777", "shards": [0]}]}
        self.recurrent_endpoints_by_dp_rank = {
            0: {
                g: [{"endpoint": f"10.0.0.1:{8000 + i}", "shards": [0]}]
                for i, g in enumerate(self.recurrent_groups)
            }
        }
        self.registered = []
        self.started = []
        self.stats = ([], [], [])
        self.declined_groups = set()
        self.fail_start_groups = set()

    def register_read(self, req_id, uuid, block_ids, *, dp_rank=0, group=None):
        self.registered.append((req_id, list(block_ids), dp_rank, group))
        return group not in self.declined_groups

    def start_read(self, req_id, uuid, endpoint, remote, local, *, decode_dp_rank=0, group=None):
        if group in self.fail_start_groups:
            raise RuntimeError("start_read boom")
        self.started.append((req_id, endpoint, list(remote), list(local), group))

    def poll_stats(self):
        return self.stats


def _raiden_manager(raiden=None):
    from sgl_jax.srt.disaggregation.raiden_transfer.conn import RaidenTransferKVManager

    raiden = raiden or _FakeRaiden()
    bootstrap = _FakeBootstrap()
    return RaidenTransferKVManager(raiden, bootstrap), raiden, bootstrap


def _hybrid_sender(mgr, transfer_id="wire-h", slot=2):
    sender = mgr.create_sender(f"req-{transfer_id}")
    sender.init(None, transfer_id=transfer_id)
    sender.attach_block_ids([3, 4], bootstrap_room=11, dp_rank=0, recurrent_slot=slot)
    return sender


def test_raiden_recurrent_transfer_id():
    from sgl_jax.srt.disaggregation.raiden_transfer.conn import recurrent_transfer_id

    assert recurrent_transfer_id("abc", CONV_STATE_KEY) == f"abc#{CONV_STATE_KEY}"


def test_raiden_sender_registers_slot_reads_and_waits_for_all_groups():
    mgr, raiden, bootstrap = _raiden_manager()
    sender = _hybrid_sender(mgr)

    sender.send()

    assert raiden.registered == [
        ("wire-h", [3, 4], 0, None),
        (f"wire-h#{RECURRENT_STATE_KEY}", [2], 0, RECURRENT_STATE_KEY),
        (f"wire-h#{CONV_STATE_KEY}", [2], 0, CONV_STATE_KEY),
    ]
    meta = bootstrap.registered[0][1]["transport_metadata"]
    assert meta["remote_recurrent_slot"] == 2

    raiden.stats = (["wire-h", f"wire-h#{RECURRENT_STATE_KEY}"], [], [])
    assert sender.poll() == KVPoll.TRANSFERRING  # conv read still pinned
    raiden.stats = ([f"wire-h#{CONV_STATE_KEY}"], [], [])
    assert sender.poll() == KVPoll.SUCCESS


def test_raiden_sender_fails_when_a_recurrent_read_is_declined():
    raiden = _FakeRaiden()
    raiden.declined_groups = {CONV_STATE_KEY}
    mgr, raiden, _ = _raiden_manager(raiden)
    sender = _hybrid_sender(mgr)

    sender.send()
    # The KV and temporal reads were registered; wait for them to settle.
    raiden.stats = (["wire-h", f"wire-h#{RECURRENT_STATE_KEY}"], [], [])
    assert sender.poll() == KVPoll.FAILED


def test_raiden_sender_failed_recurrent_read_fails_transfer():
    mgr, raiden, _ = _raiden_manager()
    sender = _hybrid_sender(mgr)
    sender.send()

    raiden.stats = (["wire-h", f"wire-h#{RECURRENT_STATE_KEY}"], [], [f"wire-h#{CONV_STATE_KEY}"])
    assert sender.poll() == KVPoll.FAILED


def test_raiden_start_prefill_rejects_recurrent_mismatch():
    from sgl_jax.srt.disaggregation.base.transfer import PrefillTransferContext

    mgr, _, _ = _raiden_manager()
    ctx = PrefillTransferContext(
        req_id="r",
        transfer_id="w",
        bootstrap_room=1,
        dp_rank=0,
        buffer_id=None,
        payload_factory=lambda: {},
        block_ids_factory=lambda: [1],
        recurrent_slot_idx=None,  # hybrid engine but no slot
    )
    with pytest.raises(ValueError, match="recurrent state mismatch"):
        mgr.start_prefill(ctx)


def _hybrid_receiver(mgr, *, remote_slot=2, local_slot=5):
    from sgl_jax.srt.disaggregation.raiden_transfer.conn import (
        RaidenMetadata,
        RaidenRecurrentRead,
    )

    receiver = mgr.create_receiver("req-rx")
    receiver.init(
        RaidenMetadata(
            uuid="wire-rx",
            remote_endpoint="10.0.0.1:7777",
            remote_block_ids=(1, 4),
            local_block_ids=(9, 10),
            bootstrap_room=43,
            jax_process_index=0,
            recurrent_reads=tuple(
                RaidenRecurrentRead(
                    group=g,
                    remote_endpoint=f"10.0.0.1:{8000 + i}",
                    remote_slot=remote_slot,
                    local_slot=local_slot,
                )
                for i, g in enumerate(_GROUPS)
            ),
        )
    )
    return receiver


def test_raiden_receiver_starts_slot_reads_and_waits_for_all():
    mgr, raiden, _ = _raiden_manager()
    receiver = _hybrid_receiver(mgr)

    assert receiver.poll() == KVPoll.TRANSFERRING
    assert [(s[0], s[2], s[3], s[4]) for s in raiden.started] == [
        ("wire-rx", [1, 4], [9, 10], None),
        (f"wire-rx#{RECURRENT_STATE_KEY}", [2], [5], RECURRENT_STATE_KEY),
        (f"wire-rx#{CONV_STATE_KEY}", [2], [5], CONV_STATE_KEY),
    ]

    raiden.stats = ([], ["wire-rx", f"wire-rx#{RECURRENT_STATE_KEY}"], [])
    assert receiver.poll() == KVPoll.TRANSFERRING
    raiden.stats = ([], [f"wire-rx#{CONV_STATE_KEY}"], [])
    assert receiver.poll() == KVPoll.SUCCESS


def test_raiden_receiver_recurrent_start_failure_fails_after_kv_settles():
    raiden = _FakeRaiden()
    raiden.fail_start_groups = {CONV_STATE_KEY}
    mgr, raiden, _ = _raiden_manager(raiden)
    receiver = _hybrid_receiver(mgr)

    receiver.poll()
    raiden.stats = ([], ["wire-rx", f"wire-rx#{RECURRENT_STATE_KEY}"], [])
    assert receiver.poll() == KVPoll.FAILED


def test_raiden_receiver_failed_recurrent_read_fails_transfer():
    mgr, raiden, _ = _raiden_manager()
    receiver = _hybrid_receiver(mgr)
    receiver.poll()

    raiden.stats = (
        [],
        ["wire-rx", f"wire-rx#{CONV_STATE_KEY}"],
        [f"wire-rx#{RECURRENT_STATE_KEY}"],
    )
    assert receiver.poll() == KVPoll.FAILED


def test_raiden_wrapper_start_recurrent_creates_one_engine_per_group_and_rank():
    import sys

    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

    created = []

    def _manager_cls(**kwargs):
        engine = mock.MagicMock()
        engine.get_local_endpoints.return_value = [
            {"endpoint": f"127.0.0.1:{9000 + len(created)}", "shards": [0]}
        ]
        engine.register_read.return_value = True
        engine.poll_stats.return_value = ([], [], [])
        created.append((kwargs, engine))
        return engine

    names = [
        "tpu_raiden",
        "tpu_raiden.frameworks.jax._tpu_raiden_jax",
        "tpu_raiden.api",
        "tpu_raiden.api.jax",
        "tpu_raiden.api.jax.kv_cache_manager",
    ]
    modules = {n: types.ModuleType(n) for n in names}
    modules["tpu_raiden.api.jax.kv_cache_manager"].KVCacheManager = _manager_cls
    split = "sgl_jax.srt.disaggregation.raiden_transfer.wrapper._split_kv_caches_by_dp_rank"
    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch(split, side_effect=lambda caches, dp: {r: list(caches) for r in range(dp)}),
    ):
        wrapper = RaidenTransferWrapper("127.0.0.1", 0)
        wrapper.start(["kv"], max_blocks=8, num_slots=4, dp_size=2)
        wrapper.start_recurrent(
            {RECURRENT_STATE_KEY: ["r0", "r1"], CONV_STATE_KEY: ["c0", "c1"]},
            num_slots=4,
            dp_size=2,
        )
        wrapper.register_read("x", 1, [2], dp_rank=1, group=CONV_STATE_KEY)

    recurrent = created[2:]
    assert len(recurrent) == 4  # 2 groups x 2 DP ranks
    assert all(kwargs["max_blocks"] == 1 for kwargs, _ in recurrent)
    assert wrapper.recurrent_groups == _GROUPS
    assert set(wrapper.recurrent_endpoints_by_dp_rank) == {0, 1}
    assert set(wrapper.recurrent_endpoints_by_dp_rank[1]) == set(_GROUPS)
    # (conv_state, rank 1) is the 4th recurrent engine created.
    recurrent[3][1].register_read.assert_called_once_with("x", 1, [2])


def test_raiden_rank_local_view_of_recurrent_buffer_has_rank_local_slots():
    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import _rank_local_array

    rsp = _make_pool(dp=2)
    _fill_pool(rsp)
    buf = rsp.recurrent_buffers[0]
    per_rank = rsp.slots_per_rank + 1
    for rank in range(2):
        view = _rank_local_array(buf, rank, 2)
        assert view.shape[0] == per_rank
        # Raiden block id == rank-local slot index.
        np.testing.assert_array_equal(
            _np(view), _rec_pattern(buf.shape, 0)[rank * per_rank : (rank + 1) * per_rank]
        )


def test_raiden_factory_rejects_chunk_transfer_for_hybrid_models():
    from sgl_jax.srt.disaggregation.factory import create_transfer_backend

    scheduler = SimpleNamespace(
        tree_cache=None,
        req_to_token_pool=SimpleNamespace(recurrent_state_pool=_make_pool(dp=1)),
    )
    args = SimpleNamespace(
        disaggregation_use_raiden=True,
        device="tpu",
        disaggregation_enable_d2h=False,
        disaggregation_max_inflight_transfers=1,
        disable_radix_cache=True,
        disaggregation_enable_chunk_prefill_transfer=True,
    )
    with pytest.raises(ValueError, match="does not support hybrid recurrent"):
        create_transfer_backend(
            scheduler,
            args,
            local_host="127.0.0.1",
            role="prefill",
            shared_secret=None,
            bootstrap_client=None,
        )
