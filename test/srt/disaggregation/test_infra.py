"""Tests for PD infrastructure: metrics, host KV pool, and server args."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.disaggregation.common import metrics as M
from sgl_jax.srt.mem_cache.host_kv_pool import (
    HostBufferHandle,
    QueueHostKVPool,
    StagedData,
)
from sgl_jax.srt.server_args import ServerArgs


# ------------------------------------------------------------------
# PD metrics (from test_metrics)
# ------------------------------------------------------------------


def test_state_transition_counter_accepts_labels():
    M.PD_STATE_TRANSITION_TOTAL.labels(
        from_state="bootstrapping",
        to_state="waiting_for_input",
        role="prefill",
    ).inc()


def test_transfer_bytes_counter_directions():
    for direction in ("d2h", "h2d", "net"):
        for role in ("prefill", "decode"):
            M.PD_TRANSFER_BYTES_TOTAL.labels(direction=direction, role=role).inc(1024)


def test_transfer_duration_phases():
    for phase in ("bootstrap", "pull", "ack"):
        for role in ("prefill", "decode"):
            with M.time_phase(phase, role):
                pass


def test_transfer_inflight_inc_dec():
    M.PD_TRANSFER_INFLIGHT.labels(role="prefill").inc()
    M.PD_TRANSFER_INFLIGHT.labels(role="prefill").dec()


def test_transfer_failures_reasons():
    for reason in (
        "timeout",
        "peer_crash",
        "network",
        "auth",
        "bootstrap_lookup",
        "receiver_init",
        "shutdown",
        "other",
    ):
        M.PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()


def test_host_pool_helpers_balance():
    M.host_pool_alloc("test-pool", 3)
    M.host_pool_alloc("test-pool", 1)
    M.host_pool_free("test-pool", 2)
    M.host_pool_free("test-pool", 2)
    M.host_pool_free("test-pool", 5)


def test_bootstrap_registry_gauge_set():
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(0)
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(7)
    M.PD_BOOTSTRAP_REGISTRY_SIZE.set(0)


def test_is_prometheus_available_is_bool():
    assert isinstance(M.is_prometheus_available(), bool)


def test_noop_when_prom_missing(monkeypatch):
    class _BoomMetric:
        def labels(self, **_):
            return self

        def inc(self, _amount=1.0):
            raise RuntimeError("boom")

        def set(self, _value):
            raise RuntimeError("boom")

        def observe(self, _value):
            raise RuntimeError("boom")

    monkeypatch.setattr(M, "PD_HOST_POOL_USED_BUFFERS", _BoomMetric())
    M.host_pool_alloc("boom-pool", 1)
    M.host_pool_free("boom-pool", 1)


# ------------------------------------------------------------------
# QueueHostKVPool (from test_queue_host_kv_pool)
# ------------------------------------------------------------------


def _single_device_mesh() -> Mesh:
    devices = jax.local_devices()
    return Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))


def _make_pool(
    pool_size: int = 4,
    max_tokens_per_buffer: int = 16,
    layer_num: int = 2,
    kv_head_per_rank: int = 2,
    head_dim: int = 8,
) -> QueueHostKVPool:
    mesh = _single_device_mesh()
    spec = PartitionSpec()
    return QueueHostKVPool(
        pool_size=pool_size,
        max_tokens_per_buffer=max_tokens_per_buffer,
        layer_num=layer_num,
        kv_head_per_rank=kv_head_per_rank,
        head_dim=head_dim,
        dtype=jnp.float32,
        mesh=mesh,
        partition_spec=spec,
    )


def test_init_validates_sizes():
    with pytest.raises(ValueError, match="pool_size"):
        _make_pool(pool_size=0)
    with pytest.raises(ValueError, match="max_tokens_per_buffer"):
        _make_pool(max_tokens_per_buffer=0)


def test_alloc_returns_handle_and_decrements_available():
    pool = _make_pool(pool_size=4, max_tokens_per_buffer=16)
    assert pool.total_size() == 4
    assert pool.available_size() == 4

    h = pool.alloc(num_tokens=8)
    assert isinstance(h, HostBufferHandle)
    assert h.num_tokens == 8
    assert h.buffer.shape == (16, 2, 2, 8)
    assert pool.available_size() == 3


def test_alloc_returns_none_when_exhausted():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=16)
    h1 = pool.alloc(8)
    h2 = pool.alloc(8)
    h3 = pool.alloc(8)
    assert h1 is not None
    assert h2 is not None
    assert h3 is None
    assert pool.available_size() == 0


def test_alloc_returns_none_when_num_tokens_exceeds_per_buffer():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    assert pool.alloc(num_tokens=9) is None
    assert pool.available_size() == 2


def test_alloc_rejects_non_positive_tokens():
    pool = _make_pool()
    with pytest.raises(ValueError, match="positive"):
        pool.alloc(0)
    with pytest.raises(ValueError, match="positive"):
        pool.alloc(-1)


def test_free_returns_buffer_to_pool_and_allows_realloc():
    pool = _make_pool(pool_size=2)
    h1 = pool.alloc(4)
    h2 = pool.alloc(4)
    assert pool.alloc(4) is None
    pool.free(h1)
    assert pool.available_size() == 1
    h3 = pool.alloc(4)
    assert h3 is not None
    pool.free(h2)
    pool.free(h3)
    assert pool.available_size() == 2


def test_double_free_raises():
    pool = _make_pool(pool_size=2)
    h = pool.alloc(4)
    pool.free(h)
    with pytest.raises(RuntimeError, match="double free"):
        pool.free(h)


def test_get_put_buffer_low_level():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    bid1, h1 = pool.get_buffer()
    bid2, h2 = pool.get_buffer()
    assert bid1 != bid2
    assert h1.num_tokens == 8 and h2.num_tokens == 8
    with pytest.raises(RuntimeError, match="empty"):
        pool.get_buffer()
    pool.put_buffer(bid1)
    bid3, _ = pool.get_buffer()
    assert bid3 == bid1


def test_copy_from_device_byte_equal_for_prefix():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    rng = np.random.default_rng(0)
    src_np = rng.integers(0, 200, size=(4, 2, 2, 8), dtype=np.int32).astype(np.float32)
    device_kv = jnp.asarray(src_np)
    staged: StagedData = pool.copy_from_device(device_kv)

    assert isinstance(staged, StagedData)
    assert staged.array.shape == (8, 2, 2, 8)
    got = np.asarray(jax.device_get(staged.array))
    np.testing.assert_array_equal(got[:4], src_np)
    np.testing.assert_array_equal(got[4:], np.zeros((4, 2, 2, 8), dtype=np.float32))

    pool.put_buffer(staged.buffer_id)
    drained = []
    while pool.available_size() > 1:
        bid_drain, _ = pool.get_buffer()
        drained.append(bid_drain)
    bid, handle = pool.get_buffer()
    assert bid == staged.buffer_id
    got2 = np.asarray(jax.device_get(handle.buffer))
    np.testing.assert_array_equal(got2[:4], src_np)


def test_copy_from_device_rejects_oversize():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=4)
    big = jnp.zeros((5, 2, 2, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="max_tokens_per_buffer"):
        pool.copy_from_device(big)


def test_copy_from_device_raises_when_pool_exhausted():
    pool = _make_pool(pool_size=1, max_tokens_per_buffer=8)
    pool.alloc(8)
    src = jnp.zeros((4, 2, 2, 8), dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.copy_from_device(src)


# ------------------------------------------------------------------
# ServerArgs PD flags (from test_server_args_stage1_smoke)
# ------------------------------------------------------------------


def _make_server_args(**overrides) -> ServerArgs:
    defaults = dict(
        model_path="dummy/model",
        device="cpu",
        random_seed=42,
        mem_fraction_static=0.5,
    )
    defaults.update(overrides)
    return ServerArgs(**defaults)


def test_pd_server_args_defaults_are_stable():
    args = _make_server_args()
    assert args.disaggregation_mode == "null"
    assert args.disaggregation_bootstrap_url is None
    assert args.disaggregation_transfer_port == 30001
    assert args.disaggregation_side_channel_port == 9600
    assert args.disaggregation_enable_d2h is False
    assert args.disaggregation_channel_number == 4
    assert args.disaggregation_host_ip is None
    assert args.disaggregation_pull_timeout_seconds == 30.0
    assert args.disaggregation_ack_timeout_seconds == 60.0


def test_pd_cli_flags_round_trip_through_parser():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(
        [
            "--model-path",
            "dummy/model",
            "--device",
            "cpu",
            "--mem-fraction-static",
            "0.5",
            "--page-size",
            "128",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-url",
            "http://127.0.0.1:8998",
            "--disaggregation-transfer-port",
            "31001",
            "--disaggregation-side-channel-port",
            "31002",
            "--disaggregation-enable-d2h",
            "--disaggregation-host-ip",
            "10.0.0.10",
            "--disaggregation-pull-timeout-seconds",
            "45",
            "--disaggregation-ack-timeout-seconds",
            "90",
        ]
    )
    ns.tensor_parallel_size = getattr(ns, "tensor_parallel_size", 1) or 1
    ns.data_parallel_size = getattr(ns, "data_parallel_size", 1) or 1
    args = ServerArgs.from_cli_args(ns)
    assert args.disaggregation_mode == "prefill"
    assert args.disaggregation_bootstrap_url == "http://127.0.0.1:8998"
    assert args.disaggregation_transfer_port == 31001
    assert args.disaggregation_side_channel_port == 31002
    assert args.disaggregation_enable_d2h is True
    assert args.disaggregation_host_ip == "10.0.0.10"
    assert args.disaggregation_pull_timeout_seconds == 45.0
    assert args.disaggregation_ack_timeout_seconds == 90.0


def test_pd_shared_secret_env_overrides_args(monkeypatch):
    monkeypatch.setenv("SGL_JAX_PD_SHARED_SECRET", "env-secret")
    args = _make_server_args(disaggregation_shared_secret="cli-secret")
    assert args.disaggregation_shared_secret == "env-secret"


@pytest.mark.parametrize("bad_host_ip", ["0.0.0.0", "127.0.0.1"])
def test_pd_host_ip_rejects_bind_and_loopback_addresses(bad_host_ip):
    with pytest.raises(ValueError, match="disaggregation-host-ip"):
        _make_server_args(disaggregation_host_ip=bad_host_ip)
