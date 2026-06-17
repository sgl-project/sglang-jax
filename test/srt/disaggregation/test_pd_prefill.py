"""Tests for PD prefill side: sender release, prefill info validation/cache, host KV pool, jax transfer wrapper, KV gather, payload roundtrip, no-chunked-prefill guard."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.bootstrap import (
    PrefillInfo,
    PrefillInfoCache,
    build_app,
    check_prefill_compat,
)
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVSender,
)
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import JaxTransferWrapper
from sgl_jax.srt.disaggregation.prefill import (
    _KV_GATHER_PAGE_BUCKETS,
    _jit_gather_all_layers,
    _jit_gather_one_layer,
    _pad_to_page_bucket,
)
from sgl_jax.srt.managers.utils import validate_pd_no_chunked_prefill
from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool, StagedData

# ---- from test_pd_prefill_sender_release.py ----


class _Status:
    def __init__(self):
        self.sub_uuids = ("req-a:kv",)

    def on_done(self):
        pass


class _Notifier:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_callback(self, uuid_bytes, cb):
        self.registered.append(uuid_bytes)

    def unregister_callback(self, uuid_bytes):
        self.unregistered.append(uuid_bytes)


class _Mgr:
    def __init__(self):
        self._notifier = _Notifier()
        self.handoff_calls = []

    @property
    def zmq_notifier(self):
        return self._notifier

    def producer_handoff(self, uuid, payload, *, use_d2h_staging, buffer_id):
        self.handoff_calls.append((uuid, use_d2h_staging, buffer_id))
        return _Status()


def _make_sender():
    mgr = _Mgr()
    sender = JaxTransferKVSender(mgr, "req-a")
    sender.init(kv_indices=None, transfer_id="req-a")
    return mgr, sender


def test_staging_send_drops_device_payload():
    mgr, sender = _make_sender()
    sender.attach_payload({"kv": [object()]}, use_d2h_staging=True, buffer_id=3)

    sender.send()

    # Staging registered the host copy, so the device payload is freed.
    assert sender._payload is None
    assert sender.poll() == KVPoll.TRANSFERRING
    assert mgr.handoff_calls == [("req-a", True, 3)]


def test_path_b_send_keeps_device_payload():
    mgr, sender = _make_sender()
    payload = {"kv": [object()]}
    sender.attach_payload(payload, use_d2h_staging=False, buffer_id=None)

    sender.send()

    # Path B pulls straight from HBM; the payload must stay alive until ack.
    assert sender._payload is payload
    assert sender.poll() == KVPoll.TRANSFERRING


def test_staging_send_handoff_failure_unregisters_and_keeps_payload():
    mgr, sender = _make_sender()

    def _boom(*a, **k):
        raise RuntimeError("handoff boom")

    mgr.producer_handoff = _boom
    payload = {"kv": [object()]}
    sender.attach_payload(payload, use_d2h_staging=True, buffer_id=1)

    try:
        sender.send()
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    # Failed handoff: callback rolled back, payload NOT dropped (no transfer).
    assert mgr.zmq_notifier.unregistered
    assert sender._payload is payload


# ---- from test_prefill_info_validation.py ----


class TestCheckPrefillCompat:
    def test_matching_config_passes(self):
        info = {"page_size": 128, "kv_dtype": "bfloat16"}
        check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_page_size_mismatch_raises(self):
        info = {"page_size": 64, "kv_dtype": "bfloat16"}
        with pytest.raises(ValueError, match="page_size"):
            check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_kv_dtype_mismatch_raises(self):
        info = {"page_size": 128, "kv_dtype": "float16"}
        with pytest.raises(ValueError, match="kv_dtype"):
            check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_peer_missing_fields_is_backward_compatible(self):
        # Older prefill that never reported these fields → skip the check.
        info = {"page_size": 0, "kv_dtype": ""}
        check_prefill_compat(info, local_page_size=128, local_kv_dtype="bfloat16")

    def test_absent_keys_are_backward_compatible(self):
        check_prefill_compat({}, local_page_size=128, local_kv_dtype="bfloat16")

    def test_local_unknown_skips_check(self):
        info = {"page_size": 64, "kv_dtype": "float16"}
        # Decode that doesn't know its own config yet → don't false-reject.
        check_prefill_compat(info, local_page_size=0, local_kv_dtype="")


class TestPrefillInfoFields:
    def test_to_dict_carries_new_fields(self):
        info = PrefillInfo(
            bootstrap_key="h:1",
            host="h",
            transfer_port=1,
            side_channel_port=2,
            page_size=128,
            kv_dtype="bfloat16",
        )
        d = info.to_dict()
        assert d["page_size"] == 128
        assert d["kv_dtype"] == "bfloat16"

    def test_defaults_are_backward_compatible(self):
        info = PrefillInfo(bootstrap_key="h:1", host="h", transfer_port=1, side_channel_port=2)
        assert info.page_size == 0
        assert info.kv_dtype == ""


class TestBootstrapRoundTrip:
    def test_register_and_get_preserves_layout_fields(self):
        app, _ = build_app()
        client = TestClient(app)
        resp = client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
                "page_size": 128,
                "kv_dtype": "bfloat16",
            },
        )
        assert resp.status_code == 200
        got = client.get("/get_prefill_info", params={"bootstrap_room": 0})
        assert got.status_code == 200
        body = got.json()
        assert body["page_size"] == 128
        assert body["kv_dtype"] == "bfloat16"

    def test_register_without_layout_fields_defaults(self):
        app, _ = build_app()
        client = TestClient(app)
        resp = client.post(
            "/register_prefill",
            json={
                "bootstrap_key": "h:1",
                "host": "h",
                "transfer_port": 1,
                "side_channel_port": 2,
            },
        )
        assert resp.status_code == 200
        body = client.get("/get_prefill_info", params={"bootstrap_room": 0}).json()
        assert body["page_size"] == 0
        assert body["kv_dtype"] == ""


# ---- from test_prefill_info_cache.py ----


def _pf(key, **kw):
    d = {
        "bootstrap_key": key,
        "host": "h",
        "transfer_port": 1,
        "side_channel_port": 2,
        "protocol_version": 1,
    }
    d.update(kw)
    return d


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class _FakeClient:
    def __init__(self, prefills) -> None:
        self.prefills = list(prefills)
        self.list_calls = 0

    def list_prefills(self):
        self.list_calls += 1
        return list(self.prefills)


def test_warm_cache_serves_with_zero_get_after_first():
    clock = _Clock()
    client = _FakeClient([_pf("a")])
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    first = cache.pick_for_room(0)
    assert first["bootstrap_key"] == "a"
    assert client.list_calls == 1  # one refresh to warm the cache

    # Many subsequent lookups at the same instant: all cache hits, no refresh.
    for _ in range(50):
        assert cache.pick_for_room(0)["bootstrap_key"] == "a"
    assert client.list_calls == 1


def test_room_modulo_selection_matches_server():
    clock = _Clock()
    client = _FakeClient([_pf("c"), _pf("a"), _pf("b")])  # sorted -> a, b, c
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    assert cache.pick_for_room(0)["bootstrap_key"] == "a"
    assert cache.pick_for_room(1)["bootstrap_key"] == "b"
    assert cache.pick_for_room(2)["bootstrap_key"] == "c"
    assert cache.pick_for_room(4)["bootstrap_key"] == "b"  # 4 % 3 == 1


def test_miss_is_rate_limited_then_resolves():
    clock = _Clock()
    client = _FakeClient([])  # no prefill registered yet
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    # First miss: refreshes once, still empty -> None (caller defers).
    assert cache.pick_for_room(0) is None
    assert client.list_calls == 1

    # Same instant: rate-limited, no second refresh.
    assert cache.pick_for_room(0) is None
    assert client.list_calls == 1

    # Prefill registers; advance past the interval -> one more refresh resolves.
    client.prefills = [_pf("a")]
    clock.t += 1.0
    info = cache.pick_for_room(0)
    assert info["bootstrap_key"] == "a"
    assert client.list_calls == 2


def test_stale_protocol_peer_rejected():
    clock = _Clock()
    client = _FakeClient([_pf("a", protocol_version=0)])
    cache = PrefillInfoCache(client, refresh_interval_s=1.0, clock=clock)

    with pytest.raises(RuntimeError, match="protocol_version"):
        cache.pick_for_room(0)


# ---- from test_host_kv_pool_per_request.py ----

_LAYER_NUM = 4

_MAX_PAGES = 8

_PER_LAYER_SHAPE = (2, 3, 5)  # (page_size, kv_head, head_dim)-ish tail

_DTYPE = jnp.float32


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_pool(pool_size=2):
    return QueueHostKVPool(
        pool_size=pool_size,
        max_padded_pages=_MAX_PAGES,
        layer_num=_LAYER_NUM,
        per_layer_shape=_PER_LAYER_SHAPE,
        dtype=_DTYPE,
        mesh=_mesh(),
        partition_spec=PartitionSpec(None, "tensor", None),
    )


def _layers(padded_pages, seed=0):
    rng = np.random.default_rng(seed)
    return [
        jnp.asarray(rng.standard_normal((padded_pages, *_PER_LAYER_SHAPE)).astype(np.float32))
        for _ in range(_LAYER_NUM)
    ]


def test_reserve_release_accounting():
    pool = _make_pool(pool_size=2)
    assert pool.total_size() == 2
    assert pool.available_size() == 2
    a = pool.reserve()
    b = pool.reserve()
    assert {a, b} == {0, 1}
    assert pool.available_size() == 0
    assert pool.reserve() is None  # exhausted
    pool.release(a)
    assert pool.available_size() == 1


def test_copy_from_device_values_match_numpy():
    pool = _make_pool(pool_size=2)
    padded_pages = 4
    layers = _layers(padded_pages, seed=7)
    bid = pool.reserve()
    staged = pool.copy_from_device(layers, bid)
    assert isinstance(staged, StagedData)
    assert staged.buffer_id == bid
    assert len(staged.array_pytree) == _LAYER_NUM
    for i, layer in enumerate(layers):
        got = np.asarray(jax.device_get(staged.array_pytree[i]))
        assert got.shape == (padded_pages, *_PER_LAYER_SHAPE)
        np.testing.assert_allclose(got, np.asarray(layer), rtol=0, atol=0)


def test_double_free_guard():
    pool = _make_pool(pool_size=1)
    bid = pool.reserve()
    pool.release(bid)
    with pytest.raises(RuntimeError):
        pool.release(bid)


# ---- from test_jax_transfer_wrapper_pytree.py ----


def test_pull_requires_every_leaf_sharded():
    w = JaxTransferWrapper("127.0.0.1", 0)
    # not started -> sharding validation must happen first and reject a None-sharded leaf
    bad = [jax.ShapeDtypeStruct((2, 2), jnp.float32, sharding=None)]
    with pytest.raises(ValueError):
        w.pull("u", bad, remote_addr="127.0.0.1:1")


def test_nbytes_sums_over_leaves(monkeypatch):
    recorded = {}

    class _Stub:
        def await_pull(self, _uuid_int, data):
            recorded["leaves"] = jax.tree.leaves(data)

    class _FakeMetricChild:
        def inc(self, value):
            recorded["inc"] = value

    class _FakeMetric:
        def labels(self, **kwargs):
            recorded["labels"] = kwargs
            return _FakeMetricChild()

    # register_pull does a local ``from ...common.metrics import
    # PD_TRANSFER_BYTES_TOTAL`` on every call, so patch the symbol at its
    # source module — that is the name the import binds to.
    import sgl_jax.srt.disaggregation.common.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "PD_TRANSFER_BYTES_TOTAL", _FakeMetric())

    w = JaxTransferWrapper("127.0.0.1", 0)
    w._server = _Stub()
    w._started = True
    arrs = [jnp.ones((4,), jnp.float32), jnp.ones((8,), jnp.float32)]
    w.register_pull("u1", arrs)
    assert len(recorded["leaves"]) == 2
    # 4 * float32 (4 bytes) = 16, 8 * float32 = 32, summed = 48.
    assert recorded["inc"] == 48


# ---- from test_kv_gather_per_layer.py ----

_NUM_LAYERS = 4

_NUM_PAGES_POOL = 32

_PAGE_SIZE = 8

_HEAD_NUM_KV = 4  # num_kv_heads

_PACKING = 2

_HEAD_DIM = 16


def _make_mesh():
    """Single-device mesh for CPU testing (1x1 for data x tensor)."""
    devices = jax.devices("cpu")[:1]
    return Mesh(
        np.array(devices).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_kv_buffers(mesh, num_layers=_NUM_LAYERS, rng_seed=42):
    """Create per-layer KV buffers mimicking memory_pool.py layout."""
    rng = np.random.default_rng(rng_seed)
    shape = (_NUM_PAGES_POOL, _PAGE_SIZE, _HEAD_NUM_KV * 2 // _PACKING, _PACKING, _HEAD_DIM)
    sharding = NamedSharding(mesh, P("data", None, "tensor", None, None))
    buffers = []
    for _ in range(num_layers):
        data = rng.standard_normal(shape).astype(np.float32)
        buf = jax.device_put(jnp.array(data), sharding)
        buffers.append(buf)
    return buffers, sharding


class TestPerLayerGather:
    """Verify per-layer gather correctness."""

    @pytest.fixture
    def setup(self):
        mesh = _make_mesh()
        buffers, pool_sharding = _make_kv_buffers(mesh)
        gather_pspec = P(None, *pool_sharding.spec[1:])
        gather_sharding = NamedSharding(mesh, gather_pspec)
        idx_sharding = NamedSharding(mesh, P(None))
        return buffers, gather_sharding, idx_sharding, mesh

    @pytest.mark.parametrize("num_pages", _KV_GATHER_PAGE_BUCKETS)
    def test_gather_matches_naive(self, setup, num_pages):
        """Per-layer gather output matches direct numpy indexing."""
        buffers, gather_sharding, idx_sharding, mesh = setup
        page_indices = jax.device_put(
            jnp.arange(num_pages, dtype=jnp.int32) % _NUM_PAGES_POOL,
            idx_sharding,
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        assert len(results) == len(buffers)
        for i, (result, buf) in enumerate(zip(results, buffers)):
            expected_np = np.asarray(buf)[np.asarray(page_indices)]
            np.testing.assert_array_equal(
                np.asarray(result),
                expected_np,
                err_msg=f"Layer {i}, num_pages={num_pages}",
            )

    @pytest.mark.parametrize("num_pages", [1, 4, 16, 64])
    def test_gather_output_shape(self, setup, num_pages):
        """Output shape is (num_pages, page_size, heads, packing, head_dim)."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.arange(num_pages, dtype=jnp.int32) % _NUM_PAGES_POOL,
            idx_sharding,
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        expected_shape = (num_pages, _PAGE_SIZE, _HEAD_NUM_KV * 2 // _PACKING, _PACKING, _HEAD_DIM)
        for result in results:
            assert result.shape == expected_shape

    def test_gather_single_layer(self, setup):
        """_jit_gather_one_layer works correctly standalone."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(jnp.array([0, 3, 7], dtype=jnp.int32), idx_sharding)
        result = _jit_gather_one_layer(buffers[0], page_indices, gather_sharding)
        expected = np.asarray(buffers[0])[[0, 3, 7]]
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_gather_stack_matches_monolithic(self, setup):
        """jnp.stack(per_layer_results) matches what a monolithic gather would produce."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(jnp.array([1, 5, 10, 20], dtype=jnp.int32), idx_sharding)
        per_layer = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        stacked = jnp.stack(per_layer, axis=0)
        assert stacked.shape[0] == _NUM_LAYERS
        assert stacked.shape[1] == 4  # num_pages queried
        for i, buf in enumerate(buffers):
            expected = np.asarray(buf)[[1, 5, 10, 20]]
            np.testing.assert_array_equal(
                np.asarray(stacked[i]),
                expected,
                err_msg=f"Layer {i}",
            )

    def test_gather_duplicate_indices(self, setup):
        """Gathering same page multiple times works correctly."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(jnp.array([0, 0, 0, 0], dtype=jnp.int32), idx_sharding)
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        for i, result in enumerate(results):
            page0 = np.asarray(buffers[i])[0]
            for j in range(4):
                np.testing.assert_array_equal(np.asarray(result)[j], page0)

    def test_gather_last_page(self, setup):
        """Can gather the last page in the pool without error."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.array([_NUM_PAGES_POOL - 1], dtype=jnp.int32), idx_sharding
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        for i, result in enumerate(results):
            expected = np.asarray(buffers[i])[_NUM_PAGES_POOL - 1 : _NUM_PAGES_POOL]
            np.testing.assert_array_equal(np.asarray(result), expected)


class TestPadToPageBucket:
    """Test the page bucketing utility."""

    @pytest.mark.parametrize(
        "input_pages,expected_bucket",
        [
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 16),
            (16, 16),
            (17, 32),
            (32, 32),
            (33, 64),
            (64, 64),
            (65, 128),
            (128, 128),
            (129, 256),
            (256, 256),
            (257, 512),
            (512, 512),
            (513, 1024),  # beyond largest bucket: rounds up, never truncates
        ],
    )
    def test_bucket_selection(self, input_pages, expected_bucket):
        assert _pad_to_page_bucket(input_pages) == expected_bucket


class TestGatherCompileCaching:
    """Verify that per-layer jit reuses compiled kernels across layers."""

    def test_same_shape_reuses_cache(self):
        """All layers with same buffer shape should hit the same compiled kernel."""
        mesh = _make_mesh()
        buffers, pool_sharding = _make_kv_buffers(mesh, num_layers=4)
        gather_sharding = NamedSharding(mesh, P(None, *pool_sharding.spec[1:]))
        idx_sharding = NamedSharding(mesh, P(None))
        page_indices = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), idx_sharding)

        # Warm up: first call compiles
        _ = _jit_gather_one_layer(buffers[0], page_indices, gather_sharding)

        # Subsequent calls should use cached compilation
        lowered_0 = _jit_gather_one_layer.lower(buffers[0], page_indices, gather_sharding)
        lowered_1 = _jit_gather_one_layer.lower(buffers[1], page_indices, gather_sharding)
        # Same HLO text means same compiled kernel will be reused
        assert lowered_0.as_text() == lowered_1.as_text()


# ---- from test_pd_payload_roundtrip.py ----


class _FakeWrapper:
    def __init__(self):
        self.registered = {}
        self.released = []

    def register_pull(self, uuid, data):
        self.registered[uuid] = data

    def release(self, uuid):
        self.released.append(uuid)


class _FakePool:
    def __init__(self, raise_on_call=None):
        self.copy_calls = []
        self.released = []
        self._raise_on_call = raise_on_call

    def copy_from_device(self, layers, buffer_id):
        self.copy_calls.append((layers, buffer_id))
        if self._raise_on_call is not None and len(self.copy_calls) == self._raise_on_call:
            raise RuntimeError("simulated D2H failure")
        return StagedData(buffer_id=buffer_id, array_pytree=layers)

    def release(self, buffer_id):
        self.released.append(buffer_id)


def test_path_a_uses_reserved_buffer_and_no_op_on_done():
    w = _FakeWrapper()
    pool = _FakePool()
    mgr = JaxTransferKVManager.__new__(JaxTransferKVManager)
    mgr._wrapper = w
    mgr._host_pool = pool
    layers = [jnp.ones((2, 3)), jnp.ones((2, 3))]
    status = mgr.producer_handoff("uuid-1", {"kv": layers}, use_d2h_staging=True, buffer_id=5)
    # copy_from_device called once with the reserved buffer id
    assert pool.copy_calls == [(layers, 5)]
    # registered under the sub-uuid as the pytree
    assert status.sub_uuids == ("uuid-1:kv",)
    assert "uuid-1:kv" in w.registered
    # on_done must NOT release the pool slot (scheduler owns release)
    status.on_done()
    assert pool.released == []


def test_path_a_rollback_releases_only_registered_sub_uuids_no_double_free():
    w = _FakeWrapper()
    # raise on the SECOND copy_from_device call
    pool = _FakePool(raise_on_call=2)
    mgr = JaxTransferKVManager.__new__(JaxTransferKVManager)
    mgr._wrapper = w
    mgr._host_pool = pool
    layers = [jnp.ones((2, 3))]
    # dicts preserve insertion order: "a" succeeds, "b" raises
    with pytest.raises(RuntimeError, match="simulated D2H failure"):
        mgr.producer_handoff(
            "uuid-1", {"a": layers, "b": layers}, use_d2h_staging=True, buffer_id=7
        )
    # only the sub-uuid registered before the failure ("a") was rolled back
    assert w.released == ["uuid-1:a"]
    # pool slot was NEVER released (scheduler prefill-terminal callback owns it)
    assert pool.released == []


# ---- from test_pd_no_chunked_prefill_guard.py ----


def _req(seqlen: int) -> SimpleNamespace:
    return SimpleNamespace(origin_input_ids=list(range(seqlen)))


def test_non_pd_mode_never_rejects():
    err = validate_pd_no_chunked_prefill(_req(100000), "null", 4096)
    assert err is None


def test_pd_under_limit_passes():
    err = validate_pd_no_chunked_prefill(_req(4096), "prefill", 4096)
    assert err is None


def test_pd_over_limit_rejected():
    err = validate_pd_no_chunked_prefill(_req(4097), "prefill", 4096)
    assert err is not None
    assert "chunked_prefill_size" in err


def test_pd_decode_mode_over_limit_rejected():
    err = validate_pd_no_chunked_prefill(_req(5000), "decode", 4096)
    assert err is not None


def test_disabled_chunked_prefill_never_rejects():
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", None) is None
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", 0) is None
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", -1) is None
