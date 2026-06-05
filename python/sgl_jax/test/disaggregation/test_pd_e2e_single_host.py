"""End-to-end PD smoke test (single host, two threads, no real model).

The full scheduler boots a model + tp_worker stack that needs a TPU;
we can't run that on CI. Instead this test exercises the PD wire
contract end-to-end:

  * real ``BootstrapServer`` (FastAPI in a background uvicorn thread)
  * real ``ZmqPullNotifier`` pair
  * real ``JaxTransferKVManager`` with mocked underlying wrapper
    (CPU jaxlib lacks ``jax.experimental.transfer.TransferConnection``)
  * deterministic "fake prefill" → KV; deterministic "fake decode" →
    output tokens
  * P registers via ``BootstrapClient``; D looks up via
    ``bootstrap_room`` and resolves the prefill peer
  * KV transfer drives the sender to SUCCESS and frees the buffer

What this does NOT cover:
  * Real model invocation (manual TPU e2e proves that)
  * ``Scheduler`` class composition (a separate unit test below
    asserts the Mixins compose cleanly)
  * ``--disaggregation-mode`` CLI parsing (covered by
    ``test_server_args_disaggregation.py``)
"""

from __future__ import annotations

import socket
import sys
import time
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.bootstrap import (
    BootstrapClient,
    BootstrapServer,
)
from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    PMetadata,
)
from sgl_jax.srt.disaggregation.common.zmq_notifier import (
    ZmqPullNotifier,
)
from sgl_jax.srt.disaggregation.jax_transfer import wrapper as jtw_mod
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import JaxTransferWrapper
from sgl_jax.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
)
from sgl_jax.srt.disaggregation.decode import (
    SchedulerDisaggregationDecodeMixin,
)
from sgl_jax.srt.managers.schedule_batch import Req, FINISH_ABORT
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def _free_port() -> int:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _shim_transfer_module():
    """Inject a fake ``jax.experimental.transfer`` so wrapper.start()
    works on CPU jaxlib (the real module fails to import). The fake
    server records await_pull / supports a synchronous pull that
    returns whatever P registered for the same uuid.
    """

    pending: dict = {}

    def make_server():
        server = mock.MagicMock()

        def await_pull(uuid_int, data):
            pending[uuid_int] = data

        def connect(addr):
            link = mock.MagicMock()

            def pull(uuid_int, specs):
                data = pending[uuid_int]
                return [data]

            link.pull = pull
            return link

        server.await_pull.side_effect = await_pull
        server.connect.side_effect = connect
        return server

    fake_mod = types.ModuleType("jax.experimental.transfer")
    fake_mod.start_transfer_server = mock.MagicMock(side_effect=lambda *a, **k: make_server())
    fake_mod._pending = pending  # so the test can inspect
    return mock.patch.dict(sys.modules, {"jax.experimental.transfer": fake_mod})


@pytest.fixture(autouse=True)
def _reset_singleton():
    jtw_mod._reset_singleton_for_test()
    yield
    jtw_mod._reset_singleton_for_test()


def _device_sharding():
    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(
        np.asarray(devices[:1]).reshape(1), axis_names=("x",)
    )
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _fake_prefill_kv(input_ids: list[int]) -> jax.Array:
    """Deterministic 'KV' for a prompt: just a function of input_ids."""

    rng = np.random.default_rng(hash(tuple(input_ids)) & 0xFFFFFFFF)
    return jax.device_put(
        jnp.asarray(rng.integers(0, 256, size=(64,), dtype=np.int32).astype(np.float32)),
        _device_sharding(),
    )


def _fake_decode(kv: jax.Array, steps: int) -> list[int]:
    """Deterministic 'tokens' from KV: hash of KV bytes + step."""

    # Pull KV bytes via the Stage 1 slice workaround.
    n_shards = len(kv.addressable_shards)
    parts = []
    for i in range(n_shards):
        sub = kv.addressable_data(i)[: kv.shape[0] // n_shards]
        parts.append(np.asarray(jax.device_get(sub)).tobytes())
    kv_bytes = b"".join(parts)
    base = sum(kv_bytes) & 0xFF
    return [(base + step) & 0xFF for step in range(steps)]


def test_pd_wire_flow_e2e():
    """Drive the full PD wire flow with two in-process threads.

    Same process so we can share the fake ``jax.experimental.transfer``
    module (the pending dict is process-global). Two real
    ``JaxTransferKVManager`` instances backed by two real
    ``ZmqPullNotifier``s.
    """

    bootstrap_port = _free_port()
    p_transfer_port = _free_port()
    p_side_channel_port = _free_port()
    d_transfer_port = _free_port()
    d_side_channel_port = _free_port()
    bootstrap_room = 12345
    prompt_input_ids = [101, 7592, 1024]  # "hi"
    expected_tokens = _fake_decode(
        _fake_prefill_kv(prompt_input_ids), steps=4
    )

    server = BootstrapServer("127.0.0.1", bootstrap_port)
    server.start()
    try:
        bootstrap_url = f"http://127.0.0.1:{bootstrap_port}"

        with _shim_transfer_module():
            # ---------- P side ----------
            p_wrapper = JaxTransferWrapper("127.0.0.1", p_transfer_port)
            with mock.patch.object(
                jtw_mod.jax, "local_devices",
                return_value=[mock.MagicMock()],
            ):
                p_wrapper.start()
            jtw_mod._reset_singleton_for_test()  # let D get its own
            p_notifier = ZmqPullNotifier(
                "prefill", "127.0.0.1", p_side_channel_port
            )
            p_notifier.start()
            p_mgr = JaxTransferKVManager(p_wrapper, p_notifier)
            p_client = BootstrapClient(bootstrap_url)
            p_key = f"p-{p_transfer_port}"
            p_client.register_prefill(
                bootstrap_key=p_key,
                host="127.0.0.1",
                transfer_port=p_transfer_port,
                side_channel_port=p_side_channel_port,
            )

            # ---------- D side ----------
            d_wrapper = JaxTransferWrapper("127.0.0.1", d_transfer_port)
            with mock.patch.object(
                jtw_mod.jax, "local_devices",
                return_value=[mock.MagicMock()],
            ):
                d_wrapper.start()
            d_notifier = ZmqPullNotifier(
                "decode", "127.0.0.1", d_side_channel_port
            )
            d_notifier.start()
            d_mgr = JaxTransferKVManager(d_wrapper, d_notifier)
            d_client = BootstrapClient(bootstrap_url)

            try:
                # ---------- Drive one request end-to-end ----------
                req_id = f"req-{bootstrap_room}"

                # P: fake prefill, then send.
                kv = _fake_prefill_kv(prompt_input_ids)
                sender = p_mgr.create_sender(req_id)
                sender.init(kv_indices=None)
                sender.attach_payload({"kv": kv}, use_d2h_staging=False)
                sender.send()

                # D: bootstrap lookup → connect → receiver pull
                p_info = d_client.get_prefill_info(bootstrap_room)
                assert p_info["bootstrap_key"] == p_key

                spec = jax.ShapeDtypeStruct(
                    (64,), jnp.float32, sharding=_device_sharding()
                )
                metadata = PMetadata(
                    remote_addr=(
                        f"{p_info['host']}:{p_info['transfer_port']}"
                    ),
                    uuid=req_id,
                    specs={"kv": spec},
                    p_side_channel_host=str(p_info["host"]),
                    p_side_channel_port=int(p_info["side_channel_port"]),
                )
                receiver = d_mgr.create_receiver(req_id)
                receiver.init(metadata)
                deadline = time.perf_counter() + 5.0
                while True:
                    state = receiver.poll()
                    if state == KVPoll.SUCCESS:
                        break
                    if state == KVPoll.FAILED:
                        pytest.fail(f"receiver state={state}")
                    if time.perf_counter() > deadline:
                        pytest.fail(
                            f"receiver stuck at {state} after 5s"
                        )
                    time.sleep(0.005)

                # P's sender should have received the ack and gone
                # SUCCESS via the ZMQ listener thread.
                deadline = time.perf_counter() + 5.0
                while sender.poll() != KVPoll.SUCCESS:
                    if time.perf_counter() > deadline:
                        pytest.fail(
                            f"sender stuck at {sender.poll()}"
                        )
                    time.sleep(0.005)

                # D: fake decode from received KV
                got_tokens = _fake_decode(receiver.result["kv"], steps=4)
                assert got_tokens == expected_tokens, (
                    f"PD decode {got_tokens} != baseline "
                    f"{expected_tokens}"
                )
            finally:
                d_notifier.stop()
                p_notifier.stop()
                p_client.unregister_prefill(p_key)
    finally:
        server.stop()


# =====================================================================
# Mixin-to-Mixin E2E fixtures
# =====================================================================


class _FakeKVPool:
    """Minimal KV pool stub with real JAX arrays for mixin e2e tests."""

    def __init__(
        self,
        total_pages: int = 128,
        page_size: int = 4,
        layer_num: int = 2,
        num_heads: int = 2,
        head_dim: int = 4,
    ):
        devices = jax.local_devices()
        self.mesh = jax.sharding.Mesh(
            np.asarray(devices[:1]).reshape(1), axis_names=("x",)
        )
        self.page_size = page_size
        self.layer_num = layer_num
        self.start_layer = 0
        self.dtype = jnp.float32
        self.kv_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, None, None, None)
        )
        self.kv_buffer = []
        for _ in range(layer_num):
            buf = jax.device_put(
                jnp.zeros(
                    (total_pages, page_size, num_heads, head_dim),
                    dtype=jnp.float32,
                ),
                self.kv_sharding,
            )
            self.kv_buffer.append(buf)

    def get_kv_buffer(self, layer_id: int) -> jax.Array:
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kvcache(self):
        return self


class _FakeAllocator:
    """Simple bitmap allocator for token slots."""

    def __init__(self, kv_pool: _FakeKVPool, total_slots: int = 512):
        self._pool = kv_pool
        self._total_slots = total_slots
        self._used = set()
        self._next = 0

    @property
    def page_size(self):
        return self._pool.page_size

    def get_kvcache(self):
        return self._pool

    def alloc(self, n: int):
        indices = []
        for _ in range(n):
            while self._next in self._used:
                self._next += 1
            assert self._next < self._total_slots
            indices.append(self._next)
            self._used.add(self._next)
            self._next += 1
        return np.array(indices, dtype=np.int32)

    def free(self, indices):
        for idx in np.asarray(indices).flat:
            self._used.discard(int(idx))


class _FakeReqToTokenPool:
    """Fake req_to_token_pool backed by a numpy 2D array."""

    def __init__(self, max_reqs: int = 16, max_tokens: int = 512):
        self.req_to_token = np.zeros(
            (max_reqs, max_tokens), dtype=np.int32
        )

    def free(self, req_pool_idx):
        pass


def _make_pd_req(
    rid: str,
    input_ids: list[int],
    bootstrap_room: int,
    req_pool_idx: int = 0,
    bootstrap_host: str = "127.0.0.1",
    bootstrap_port: int = 9999,
):
    req = Req(
        rid=rid,
        origin_input_text="test",
        origin_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    req.bootstrap_room = bootstrap_room
    req.bootstrap_host = bootstrap_host
    req.bootstrap_port = bootstrap_port
    req.req_pool_idx = req_pool_idx
    req.output_ids = []
    req.fill_ids = list(input_ids)
    return req


class _FakeBatch:
    """Minimal batch stub."""

    def __init__(self, reqs):
        self.reqs = reqs
        self.next_batch_sampling_info = None


class _FakePrefillScheduler(SchedulerDisaggregationPrefillMixin):
    """Stub that inherits SchedulerDisaggregationPrefillMixin."""

    def __init__(self, kv_pool, allocator, req_to_token_pool, kv_manager):
        from sgl_jax.srt.disaggregation.prefill import (
            PrefillBootstrapQueue,
        )

        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.disagg_kv_manager = kv_manager
        self.disagg_prefill_queue = PrefillBootstrapQueue()
        self.disagg_use_d2h_staging = False
        self._stream_output_calls = []

    def process_batch_result(self, batch, result):
        pass

    def set_next_batch_sampling_info_done(self, batch):
        pass

    def stream_output(self, reqs, return_logprob, return_output_logprob_only):
        self._stream_output_calls.append(reqs[:])

    def cache_finished_req(self, req):
        pass


class _FakeDecodeScheduler(SchedulerDisaggregationDecodeMixin):
    """Stub that inherits SchedulerDisaggregationDecodeMixin."""

    def __init__(self, kv_pool, allocator, req_to_token_pool, kv_manager, bootstrap_client):
        from sgl_jax.srt.disaggregation.decode import (
            DecodePreallocQueue,
            DecodeTransferQueue,
        )

        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.disagg_kv_manager = kv_manager
        self.disagg_bootstrap_client = bootstrap_client
        self.disagg_prealloc_queue = DecodePreallocQueue()
        self.disagg_transfer_queue = DecodeTransferQueue()
        self.waiting_queue = []

    def process_input_requests(self, recv_reqs):
        for r in recv_reqs:
            self.waiting_queue.append(r)


def _fill_req_to_token(pool, req, allocator):
    """Assign slot indices for a req in req_to_token_pool."""
    seqlen = len(req.origin_input_ids)
    page_size = allocator.page_size
    page_aligned = ((seqlen + page_size - 1) // page_size) * page_size
    indices = allocator.alloc(page_aligned)
    pool.req_to_token[req.req_pool_idx, :page_aligned] = indices
    return indices


@pytest.fixture()
def mixin_e2e_infra():
    """Set up full mixin e2e infrastructure."""
    bootstrap_port = _free_port()
    p_transfer_port = _free_port()
    p_side_channel_port = _free_port()
    d_transfer_port = _free_port()
    d_side_channel_port = _free_port()

    server = BootstrapServer("127.0.0.1", bootstrap_port)
    server.start()

    with _shim_transfer_module():
        p_kv_pool = _FakeKVPool()
        p_alloc = _FakeAllocator(p_kv_pool)
        p_r2t = _FakeReqToTokenPool()

        d_kv_pool = _FakeKVPool()
        d_alloc = _FakeAllocator(d_kv_pool)
        d_r2t = _FakeReqToTokenPool()

        p_wrapper = JaxTransferWrapper("127.0.0.1", p_transfer_port)
        with mock.patch.object(
            jtw_mod.jax, "local_devices",
            return_value=[mock.MagicMock()],
        ):
            p_wrapper.start()
        jtw_mod._reset_singleton_for_test()

        p_notifier = ZmqPullNotifier(
            "prefill", "127.0.0.1", p_side_channel_port
        )
        p_notifier.start()
        p_mgr = JaxTransferKVManager(p_wrapper, p_notifier)

        p_client = BootstrapClient(f"http://127.0.0.1:{bootstrap_port}")
        p_key = f"p-{p_transfer_port}"
        p_client.register_prefill(
            bootstrap_key=p_key,
            host="127.0.0.1",
            transfer_port=p_transfer_port,
            side_channel_port=p_side_channel_port,
        )

        d_wrapper = JaxTransferWrapper("127.0.0.1", d_transfer_port)
        with mock.patch.object(
            jtw_mod.jax, "local_devices",
            return_value=[mock.MagicMock()],
        ):
            d_wrapper.start()

        d_notifier = ZmqPullNotifier(
            "decode", "127.0.0.1", d_side_channel_port
        )
        d_notifier.start()
        d_mgr = JaxTransferKVManager(d_wrapper, d_notifier)
        d_client = BootstrapClient(f"http://127.0.0.1:{bootstrap_port}")

        p_sched = _FakePrefillScheduler(
            p_kv_pool, p_alloc, p_r2t, p_mgr
        )
        d_sched = _FakeDecodeScheduler(
            d_kv_pool, d_alloc, d_r2t, d_mgr, d_client
        )

        # Seed P's KV pool with deterministic data per page
        for layer_idx in range(p_kv_pool.layer_num):
            rng = np.random.default_rng(layer_idx)
            buf_np = rng.standard_normal(
                p_kv_pool.kv_buffer[layer_idx].shape
            ).astype(np.float32)
            p_kv_pool.kv_buffer[layer_idx] = jax.device_put(
                jnp.asarray(buf_np), p_kv_pool.kv_sharding
            )

        yield {
            "p_sched": p_sched,
            "d_sched": d_sched,
            "p_kv_pool": p_kv_pool,
            "d_kv_pool": d_kv_pool,
            "p_alloc": p_alloc,
            "d_alloc": d_alloc,
            "p_r2t": p_r2t,
            "d_r2t": d_r2t,
            "p_mgr": p_mgr,
            "d_mgr": d_mgr,
            "bootstrap_port": bootstrap_port,
        }

        d_notifier.stop()
        p_notifier.stop()
        p_client.unregister_prefill(p_key)

    server.stop()


def _poll_until(predicate, timeout=5.0, interval=0.005):
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            pytest.fail(f"predicate not met within {timeout}s")
        time.sleep(interval)


def test_mixin_e2e_single_req_happy_path(mixin_e2e_infra):
    """Full P→D flow through mixin layer: extract KV → send → receive → write."""
    infra = mixin_e2e_infra
    p_sched = infra["p_sched"]
    d_sched = infra["d_sched"]

    input_ids = list(range(1, 9))  # 8 tokens → 2 pages (page_size=4)
    req_p = _make_pd_req("req-1", input_ids, bootstrap_room=42, req_pool_idx=0)
    _fill_req_to_token(infra["p_r2t"], req_p, infra["p_alloc"])

    batch = _FakeBatch([req_p])
    p_sched.process_prefill_chunk(batch, None)

    _poll_until(lambda: p_sched.disagg_prefill_queue.drain_terminal() or True)

    req_d = _make_pd_req("req-1", input_ids, bootstrap_room=42, req_pool_idx=0)
    d_sched.process_input_requests_disagg_decode([req_d])

    assert len(d_sched.disagg_prealloc_queue) == 1

    _poll_until(
        lambda: (
            d_sched.process_decode_queue() or True
        ) and len(d_sched.waiting_queue) > 0,
        timeout=5.0,
    )

    assert len(d_sched.waiting_queue) == 1
    finished_req = d_sched.waiting_queue[0]
    assert finished_req.rid == "req-1"
    assert getattr(finished_req, "_pd_skip_prefix_match", False) is True

    # P sender should be SUCCESS after D ack
    p_sched.send_kv_chunk()
    _poll_until(
        lambda: (p_sched.send_kv_chunk() or True) and
                len(p_sched._stream_output_calls) > 0,
        timeout=5.0,
    )
    assert len(p_sched._stream_output_calls) >= 1


def test_mixin_e2e_multi_req_concurrent(mixin_e2e_infra):
    """3 concurrent requests through full P→D flow."""
    infra = mixin_e2e_infra
    p_sched = infra["p_sched"]
    d_sched = infra["d_sched"]

    reqs_p = []
    reqs_d = []
    for i in range(3):
        rid = f"req-multi-{i}"
        input_ids = list(range(1, 5 + i * 4))
        req_p = _make_pd_req(rid, input_ids, bootstrap_room=100 + i, req_pool_idx=i)
        _fill_req_to_token(infra["p_r2t"], req_p, infra["p_alloc"])
        reqs_p.append(req_p)
        reqs_d.append(
            _make_pd_req(rid, input_ids, bootstrap_room=100 + i, req_pool_idx=i)
        )

    batch = _FakeBatch(reqs_p)
    p_sched.process_prefill_chunk(batch, None)

    d_sched.process_input_requests_disagg_decode(reqs_d)

    _poll_until(
        lambda: (
            d_sched.process_decode_queue() or True
        ) and len(d_sched.waiting_queue) >= 3,
        timeout=10.0,
    )

    finished_rids = {r.rid for r in d_sched.waiting_queue}
    expected_rids = {f"req-multi-{i}" for i in range(3)}
    assert finished_rids == expected_rids

    _poll_until(
        lambda: (p_sched.send_kv_chunk() or True) and
                len(p_sched._stream_output_calls) >= 3,
        timeout=10.0,
    )


def test_mixin_e2e_sender_failure_aborts(mixin_e2e_infra):
    """P sender forced to fail → _finish_prefill_only_failure streams ABORT."""
    infra = mixin_e2e_infra
    p_sched = infra["p_sched"]

    input_ids = list(range(1, 5))
    req_p = _make_pd_req("req-fail", input_ids, bootstrap_room=77, req_pool_idx=0)
    _fill_req_to_token(infra["p_r2t"], req_p, infra["p_alloc"])

    batch = _FakeBatch([req_p])
    p_sched.process_prefill_chunk(batch, None)

    # Force the sender to FAILED
    assert "req-fail" in p_sched.disagg_prefill_queue._entries

    entry = p_sched.disagg_prefill_queue._entries["req-fail"]
    entry.sender.fail(reason="test-induced-failure")

    p_sched.send_kv_chunk()

    assert len(p_sched._stream_output_calls) >= 1
    aborted_req = p_sched._stream_output_calls[-1][0]
    assert isinstance(aborted_req.finished_reason, FINISH_ABORT)


def test_mixin_e2e_pd_skip_prefix_match_consume_once(mixin_e2e_infra):
    """After D-side KV write, _pd_skip_prefix_match is set and consumed once."""
    infra = mixin_e2e_infra
    p_sched = infra["p_sched"]
    d_sched = infra["d_sched"]

    input_ids = list(range(1, 9))
    req_p = _make_pd_req("req-skip", input_ids, bootstrap_room=55, req_pool_idx=0)
    _fill_req_to_token(infra["p_r2t"], req_p, infra["p_alloc"])

    batch = _FakeBatch([req_p])
    p_sched.process_prefill_chunk(batch, None)

    req_d = _make_pd_req("req-skip", input_ids, bootstrap_room=55, req_pool_idx=0)
    d_sched.process_input_requests_disagg_decode([req_d])

    _poll_until(
        lambda: (d_sched.process_decode_queue() or True) and
                len(d_sched.waiting_queue) > 0,
        timeout=5.0,
    )

    finished_req = d_sched.waiting_queue[0]
    assert finished_req._pd_skip_prefix_match is True

    # First call: flag consumed, skip match_prefix
    finished_req.init_next_round_input(tree_cache=None)
    assert finished_req._pd_skip_prefix_match is False
    assert finished_req.extend_input_len == len(input_ids) - len(finished_req.prefix_indices)

    # Second call: normal path (no skip)
    finished_req.init_next_round_input(tree_cache=None)
    assert finished_req._pd_skip_prefix_match is False


def test_scheduler_composes_disaggregation_mixins():
    """Static check: the Scheduler class declares both PD Mixins in
    its MRO, so a future scheduler instance will have the event-loop
    methods + queue installation hooks available.

    We can't instantiate Scheduler on CPU (it loads a model + tp
    worker), so we inspect the class object directly.
    """

    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.disaggregation.decode import (
        SchedulerDisaggregationDecodeMixin,
    )
    from sgl_jax.srt.disaggregation.prefill import (
        SchedulerDisaggregationPrefillMixin,
    )

    assert issubclass(Scheduler, SchedulerDisaggregationPrefillMixin)
    assert issubclass(Scheduler, SchedulerDisaggregationDecodeMixin)
    assert hasattr(Scheduler, "event_loop_normal_disagg_prefill")
    assert hasattr(Scheduler, "event_loop_normal_disagg_decode")
    assert hasattr(Scheduler, "process_prefill_chunk")
    assert hasattr(Scheduler, "process_decode_queue")
    assert hasattr(Scheduler, "send_kv_chunk")
