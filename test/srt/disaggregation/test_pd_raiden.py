"""CPU contract tests for the optional tpu-raiden PD data plane."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types
from unittest import mock

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.raiden import raiden_requested
from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.base.transfer import (
    AdmissionState,
    DecodeTransferContext,
    slots_to_page_ids,
)
from sgl_jax.srt.disaggregation.common.capacity import per_rank_inflight_limit
from sgl_jax.srt.disaggregation.factory import create_transfer_backend
from sgl_jax.srt.disaggregation.raiden_transfer.conn import (
    RaidenMetadata,
    RaidenTransferKVManager,
    _uuid_to_int,
)
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import (
    RaidenTransferWrapper,
    _rank_local_array,
)
from sgl_jax.srt.server_args import ServerArgs


class _FakeBootstrap:
    def __init__(self) -> None:
        self.registered: list[tuple[tuple, dict]] = []
        self.popped: list[tuple[int, dict]] = []
        self.transfer_info = None
        self.fail_register = False

    def register_transfer(self, *args, **kwargs) -> None:
        if self.fail_register:
            raise RuntimeError("bootstrap unavailable")
        self.registered.append((args, kwargs))

    def pop_transfer(self, room: int, **kwargs) -> None:
        self.popped.append((room, kwargs))

    def get_transfer_info(self, _room: int, **_kwargs):
        return self.transfer_info


class _FakeRaiden:
    control_port = 7777

    def __init__(self, dp_size: int = 1) -> None:
        self.dp_size = dp_size
        self._endpoints_by_dp_rank = {
            rank: [
                {
                    "endpoint": f"10.0.0.1:{7777 + rank * 10}",
                    "shards": [0],
                }
            ]
            for rank in range(dp_size)
        }
        self.registered: list[tuple] = []
        self.started: list[tuple] = []
        self.stats = ([], [], [])
        self.register_result = True

    @property
    def endpoints_by_dp_rank(self):
        return {rank: list(endpoints) for rank, endpoints in self._endpoints_by_dp_rank.items()}

    @property
    def endpoints(self):
        return self._endpoints_by_dp_rank.get(0, [])

    @endpoints.setter
    def endpoints(self, value):
        self._endpoints_by_dp_rank[0] = value

    def register_read(self, *args, **kwargs):
        self.registered.append((args, kwargs))
        return self.register_result

    def start_read(self, *args, **kwargs):
        self.started.append((args, kwargs))

    def poll_stats(self):
        return self.stats


def _manager(fake_raiden: _FakeRaiden, bootstrap: _FakeBootstrap):
    return RaidenTransferKVManager(fake_raiden, bootstrap)


def test_raiden_sender_registers_once_and_completes_from_poll_stats():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    sender = manager.create_sender("req-1")
    sender.init(None, transfer_id="wire-1")
    sender.attach_block_ids([3, 8, 13], bootstrap_room=42, dp_rank=0)

    sender.send()

    assert raiden.registered == [(("wire-1", _uuid_to_int("wire-1"), [3, 8, 13]), {"dp_rank": 0})]
    assert bootstrap.registered[0][0] == (42, "wire-1")
    assert bootstrap.registered[0][1] == {
        "jax_process_index": 0,
        "prefill_dp_rank": 0,
        "transport_metadata": {"remote_block_ids": [3, 8, 13]},
    }
    assert sender.poll() == KVPoll.TRANSFERRING

    raiden.stats = (["wire-1"], [], [])
    assert sender.poll() == KVPoll.SUCCESS
    assert "req-1" not in manager._senders


def test_raiden_receiver_starts_direct_block_read_and_pops_metadata():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    receiver = manager.create_receiver("req-2")
    receiver.init(
        RaidenMetadata(
            uuid="wire-2",
            remote_endpoint="10.0.0.1:7777",
            remote_block_ids=(1, 4),
            local_block_ids=(9, 10),
            bootstrap_room=43,
            jax_process_index=3,
        )
    )

    assert receiver.poll() == KVPoll.TRANSFERRING
    assert raiden.started == [
        (
            ("wire-2", _uuid_to_int("wire-2"), "10.0.0.1:7777", [1, 4], [9, 10]),
            {"decode_dp_rank": 0},
        )
    ]

    raiden.stats = ([], ["wire-2"], [])
    assert receiver.poll() == KVPoll.SUCCESS
    assert bootstrap.popped == [(43, {"jax_process_index": 3, "prefill_dp_rank": 0})]
    assert "req-2" not in manager._receivers


def test_raiden_commit_runs_direct_observability_hook():
    raiden = _FakeRaiden()
    manager = _manager(raiden, _FakeBootstrap())
    committed = []
    receiver = manager.create_receiver("req-commit")
    expected = {"global_digest": "abc"}
    receiver.init(
        RaidenMetadata(
            uuid="wire-commit",
            remote_endpoint="10.0.0.1:7777",
            remote_block_ids=(1,),
            local_block_ids=(2,),
            bootstrap_room=None,
            direct_commit=committed.append,
            expected_debug=expected,
        )
    )
    assert receiver.poll() == KVPoll.TRANSFERRING
    raiden.stats = ([], ["wire-commit"], [])
    assert receiver.poll() == KVPoll.SUCCESS

    receiver.commit(lambda _: None)

    assert committed == [expected]


def test_raiden_failure_and_abort_cleanup_are_terminal_and_idempotent():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    receiver = manager.create_receiver("req-3")
    receiver.init(
        RaidenMetadata(
            uuid="wire-3",
            remote_endpoint="10.0.0.1:7777",
            remote_block_ids=(1,),
            local_block_ids=(2,),
            bootstrap_room=44,
        )
    )
    assert receiver.poll() == KVPoll.TRANSFERRING
    raiden.stats = ([], [], ["wire-3"])
    assert receiver.poll() == KVPoll.FAILED
    receiver.abort()
    assert bootstrap.popped == [(44, {"jax_process_index": 0, "prefill_dp_rank": 0})]


def test_raiden_abort_waits_for_engine_terminal_before_cleanup():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    sender = manager.create_sender("req-abort")
    sender.init(None, transfer_id="wire-abort")
    sender.attach_block_ids([1], bootstrap_room=48)
    sender.send()

    sender.abort()

    assert sender.poll() == KVPoll.TRANSFERRING
    assert "req-abort" in manager._senders
    assert bootstrap.popped == []

    raiden.stats = (["wire-abort"], [], [])
    assert sender.poll() == KVPoll.FAILED
    assert "req-abort" not in manager._senders
    assert bootstrap.popped == [(48, {"jax_process_index": 0, "prefill_dp_rank": 0})]


def test_raiden_receiver_abort_waits_for_engine_terminal():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    receiver = manager.create_receiver("req-abort")
    receiver.init(
        RaidenMetadata(
            uuid="wire-abort",
            remote_endpoint="10.0.0.1:7777",
            remote_block_ids=(1,),
            local_block_ids=(2,),
            bootstrap_room=49,
        )
    )
    assert receiver.poll() == KVPoll.TRANSFERRING

    receiver.abort()

    assert receiver.poll() == KVPoll.TRANSFERRING
    assert "req-abort" in manager._receivers
    assert bootstrap.popped == []

    raiden.stats = ([], ["wire-abort"], [])
    assert receiver.poll() == KVPoll.FAILED
    assert "req-abort" not in manager._receivers
    assert bootstrap.popped == [(49, {"jax_process_index": 0, "prefill_dp_rank": 0})]


def test_raiden_reaper_marks_timeout_without_pruning_sender():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    manager = RaidenTransferKVManager(
        raiden,
        bootstrap,
        ack_timeout_seconds=1.0,
        pull_timeout_seconds=1.0,
    )
    sender = manager.create_sender("req-timeout")
    sender.init(None, transfer_id="wire-timeout")
    sender.attach_block_ids([1], bootstrap_room=52)
    sender.send()

    timed_out, _ = manager.reap_once(sender.transfer_started_at + 2.0)

    assert timed_out == ["req-timeout"]
    assert sender.state == KVPoll.TRANSFERRING
    assert "req-timeout" in manager._senders
    assert bootstrap.popped == []


def test_raiden_metadata_publish_failure_keeps_pages_until_terminal():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    bootstrap.fail_register = True
    manager = _manager(raiden, bootstrap)
    sender = manager.create_sender("req-publish")
    sender.init(None, transfer_id="wire-publish")
    sender.attach_block_ids([1], bootstrap_room=53)

    sender.send()

    assert sender.state == KVPoll.TRANSFERRING
    assert "req-publish" in manager._senders
    raiden.stats = (["wire-publish"], [], [])
    assert sender.poll() == KVPoll.FAILED


def test_raiden_manager_owns_decode_admission_and_endpoint_mapping():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-4",
        "prefill_dp_rank": 0,
        "transport_metadata": {"remote_block_ids": [3, 4]},
    }
    manager = _manager(raiden, bootstrap)
    context = DecodeTransferContext(
        req_id="req-4",
        transfer_id="wire-4",
        bootstrap_room=45,
        decode_dp_rank=0,
        prefill_dp_rank=0,
        peer_info={
            "host": "10.0.0.1",
            "transport_metadata": {
                "engine": "raiden",
                "dp_rank": 0,
                "dp_size": 1,
                "local_control_port": 7777,
                "endpoints": raiden.endpoints,
            },
        },
        kv_indices=[18, 19, 20, 21],
        page_size=2,
        prompt_tokens=4,
        spec_factory=lambda: None,
    )

    admission = manager.try_start_decode(context)

    assert admission.state == AdmissionState.ADMITTED
    assert admission.receiver is not None
    assert admission.receiver.poll() == KVPoll.TRANSFERRING
    assert raiden.started == [
        (
            ("wire-4", _uuid_to_int("wire-4"), "10.0.0.1:7777", [3, 4], [9, 10]),
            {"decode_dp_rank": 0},
        )
    ]


@pytest.mark.parametrize(
    ("prefill_dp_rank", "decode_dp_rank"),
    [(prefill, decode) for prefill in range(4) for decode in range(4)],
)
def test_raiden_manager_routes_all_dp4_prefill_decode_pairs(prefill_dp_rank, decode_dp_rank):
    raiden = _FakeRaiden(dp_size=4)
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-cross",
        "prefill_dp_rank": prefill_dp_rank,
        "transport_metadata": {"remote_block_ids": [1]},
    }
    manager = _manager(raiden, bootstrap)
    prefill_endpoints = raiden.endpoints_by_dp_rank[prefill_dp_rank]

    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-cross",
            transfer_id="wire-cross",
            bootstrap_room=91,
            decode_dp_rank=decode_dp_rank,
            prefill_dp_rank=prefill_dp_rank,
            peer_info={
                "host": "10.0.0.9",
                "system_dp_rank": prefill_dp_rank,
                "transport_metadata": {
                    "engine": "raiden",
                    "dp_rank": prefill_dp_rank,
                    "dp_size": 4,
                    "endpoints": prefill_endpoints,
                },
            },
            kv_indices=[2, 3],
            page_size=2,
            prompt_tokens=2,
            spec_factory=lambda: None,
        )
    )

    assert admission.state == AdmissionState.ADMITTED
    assert bootstrap.transfer_info is not None
    assert admission.receiver.poll() == KVPoll.TRANSFERRING
    args, kwargs = raiden.started[-1]
    assert args[2] == f"10.0.0.1:{7777 + prefill_dp_rank * 10}"
    assert args[3:] == ([1], [1])
    assert kwargs == {"decode_dp_rank": decode_dp_rank}


def test_raiden_manager_rejects_peer_without_endpoint_descriptors():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-legacy",
        "prefill_dp_rank": 0,
        "transport_metadata": {"remote_block_ids": [3]},
    }
    manager = _manager(raiden, bootstrap)

    with pytest.raises(ValueError, match="endpoint descriptors"):
        manager.try_start_decode(
            DecodeTransferContext(
                req_id="req-legacy",
                transfer_id="wire-legacy",
                bootstrap_room=47,
                decode_dp_rank=0,
                prefill_dp_rank=0,
                peer_info={
                    "host": "10.0.0.1",
                    "local_control_port": 7777,
                },
                kv_indices=[18, 19],
                page_size=2,
                prompt_tokens=2,
                spec_factory=lambda: None,
            )
        )


def test_raiden_preserves_published_shard_endpoints():
    raiden = _FakeRaiden()
    raiden.endpoints = [
        {"endpoint": "10.0.0.2:8000", "shards": [0, 2]},
        {"endpoint": "10.0.0.2:8100", "shards": [1, 3]},
    ]
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-shards",
        "transport_metadata": {"remote_block_ids": [3]},
    }
    manager = _manager(raiden, bootstrap)
    endpoints = [
        {"endpoint": "0.0.0.0:7001", "shards": [0, 2]},
        {"endpoint": "10.0.0.1:7013", "shards": [1, 3]},
    ]

    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-shards",
            transfer_id="wire-shards",
            bootstrap_room=50,
            decode_dp_rank=0,
            prefill_dp_rank=0,
            peer_info={
                "host": "10.0.0.1",
                "transport_metadata": {"engine": "raiden", "endpoints": endpoints},
            },
            kv_indices=[18, 19],
            page_size=2,
            prompt_tokens=2,
            spec_factory=lambda: None,
        )
    )

    assert admission.receiver is not None
    admission.receiver.poll()
    assert raiden.started[0][0][2] == [
        {"endpoint": "10.0.0.1:7001", "shards": [0, 2]},
        {"endpoint": "10.0.0.1:7013", "shards": [1, 3]},
    ]


def test_raiden_manager_preserves_each_published_endpoint_port_and_shards():
    raiden = _FakeRaiden()
    raiden._endpoints_by_dp_rank[0] = [
        {"endpoint": "0.0.0.0:7001", "shards": [0, 2]},
        {"endpoint": "0.0.0.0:7999", "shards": [1, 3]},
    ]
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-endpoints",
        "prefill_dp_rank": 0,
        "transport_metadata": {"remote_block_ids": [1]},
    }
    manager = _manager(raiden, bootstrap)
    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-endpoints",
            transfer_id="wire-endpoints",
            bootstrap_room=48,
            decode_dp_rank=0,
            prefill_dp_rank=0,
            peer_info={
                "host": "10.0.0.8",
                "transport_metadata": {
                    "engine": "raiden",
                    "dp_rank": 0,
                    "dp_size": 1,
                    "endpoints": raiden.endpoints,
                },
            },
            kv_indices=[2, 3],
            page_size=2,
            prompt_tokens=2,
            spec_factory=lambda: None,
        )
    )
    assert admission.receiver.poll() == KVPoll.TRANSFERRING
    remote = raiden.started[-1][0][2]
    assert remote == [
        {"endpoint": "10.0.0.8:7001", "shards": [0, 2]},
        {"endpoint": "10.0.0.8:7999", "shards": [1, 3]},
    ]


def test_raiden_manager_defers_until_request_metadata_is_published():
    manager = _manager(_FakeRaiden(), _FakeBootstrap())
    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-5",
            transfer_id="wire-5",
            bootstrap_room=46,
            decode_dp_rank=0,
            prefill_dp_rank=0,
            peer_info={},
            kv_indices=[],
            page_size=128,
            prompt_tokens=1,
            spec_factory=lambda: None,
        )
    )
    assert admission.state == AdmissionState.DEFERRED


def test_raiden_prefill_metadata_requires_a_bound_control_port():
    raiden = _FakeRaiden()
    raiden.endpoints = []
    raiden.control_port = 0

    with pytest.raises(RuntimeError, match="did not publish endpoints"):
        _manager(raiden, _FakeBootstrap()).prefill_transport_metadata()


def test_raiden_prefill_batch_waits_for_kv_buffers_once():
    manager = _manager(_FakeRaiden(), _FakeBootstrap())
    buffers = [object(), object()]
    with mock.patch("jax.block_until_ready") as block_until_ready:
        manager.prepare_prefill_batch(buffers)
    block_until_ready.assert_called_once_with(buffers)


def test_raiden_uuid_is_stable_and_json_safe():
    assert _uuid_to_int("wire") == _uuid_to_int("wire")
    assert _uuid_to_int("wire") != _uuid_to_int("other")
    assert 0 <= _uuid_to_int("wire") < 2**50


def test_raiden_page_mapping_requires_aligned_contiguous_slots():
    assert slots_to_page_ids([8, 9, 10, 11, 20, 21], 4, 6) == (2, 5)
    with pytest.raises(ValueError, match="non-aligned"):
        slots_to_page_ids([9, 10], 2, 2)
    with pytest.raises(ValueError, match="not contiguous"):
        slots_to_page_ids([8, 10], 2, 2)


def test_register_read_false_does_not_publish_stale_metadata():
    raiden = _FakeRaiden()
    raiden.register_result = False
    bootstrap = _FakeBootstrap()
    manager = _manager(raiden, bootstrap)
    sender = manager.create_sender("req-skip")
    sender.init(None, transfer_id="wire-skip")
    sender.attach_block_ids([3], bootstrap_room=51)

    sender.send()

    assert sender.poll() == KVPoll.SUCCESS
    assert bootstrap.registered == []
    assert "req-skip" not in manager._senders


def test_raiden_loader_is_opt_in():
    assert not raiden_requested([])
    assert raiden_requested(["--disaggregation-use-raiden"])
    assert not raiden_requested(["--disaggregation-use-raiden", "--no-disaggregation-use-raiden"])


def test_launch_server_spawn_reimport_preloads_raiden_before_jax():
    code = """
import runpy
import sys
import types

extension = "tpu_raiden.frameworks.jax._tpu_raiden_jax"
sys.modules[extension] = types.ModuleType(extension)
sys.argv = ["sgl_jax.launch_server", "--disaggregation-use-raiden"]
runpy.run_module("sgl_jax.launch_server", run_name="__mp_main__")
assert extension in sys.modules
assert "jax" not in sys.modules
"""
    env = os.environ.copy()
    python_path = os.path.abspath("python")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (python_path, env.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_raiden_cli_is_opt_in():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    defaults = parser.parse_args(["--model-path", "dummy"])
    selected = parser.parse_args(["--model-path", "dummy", "--disaggregation-use-raiden"])
    assert defaults.disaggregation_use_raiden is False
    assert selected.disaggregation_use_raiden is True


@pytest.mark.parametrize(
    ("max_inflight", "dp_size", "expected"),
    [(8, 1, 8), (8, 2, 4), (32, 4, 8), (10, 4, 3), (0, 4, 0)],
)
def test_raiden_inflight_capacity_is_partitioned_per_rank(max_inflight, dp_size, expected):
    assert per_rank_inflight_limit(max_inflight, dp_size) == expected


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"device": "cpu"}, "device=tpu"),
        ({"disaggregation_enable_d2h": True}, "D2H staging"),
        ({"disaggregation_max_inflight_transfers": 0}, "max_inflight_transfers"),
        ({"disable_radix_cache": False}, "disable-radix-cache"),
    ],
)
def test_raiden_factory_rejects_invalid_config(override, error):
    config = {
        "disaggregation_use_raiden": True,
        "device": "tpu",
        "disaggregation_enable_d2h": False,
        "disaggregation_max_inflight_transfers": 1,
        "disable_radix_cache": True,
    }
    config.update(override)

    with pytest.raises(ValueError, match=error):
        create_transfer_backend(
            None,
            types.SimpleNamespace(**config),
            local_host="127.0.0.1",
            role="prefill",
            shared_secret=None,
            bootstrap_client=None,
        )


def test_raiden_wrapper_uses_public_jax_api_and_configured_parallelism():
    engine = mock.MagicMock()
    engine.get_local_endpoints.return_value = [{"endpoint": "127.0.0.1:7788", "shards": [0]}]
    manager_cls = mock.MagicMock(return_value=engine)
    modules = {
        "tpu_raiden": types.ModuleType("tpu_raiden"),
        "tpu_raiden.api": types.ModuleType("tpu_raiden.api"),
        "tpu_raiden.api.jax": types.ModuleType("tpu_raiden.api.jax"),
        "tpu_raiden.api.jax.kv_cache_manager": types.ModuleType(
            "tpu_raiden.api.jax.kv_cache_manager"
        ),
    }
    modules["tpu_raiden.api.jax.kv_cache_manager"].KVCacheManager = manager_cls

    with mock.patch.dict(sys.modules, modules):
        wrapper = RaidenTransferWrapper("127.0.0.1", 0, parallelism=3)
        wrapper.start([object()], max_blocks=64, num_slots=8, timeout_s=12.0)
        wrapper.start_read("req", 11, "remote:1", [1], [2])

    kwargs = manager_cls.call_args.kwargs
    assert kwargs["max_blocks"] == 64
    assert kwargs["num_slots"] == 8
    assert kwargs["unsafe_skip_buffer_lock"] is True
    engine.start_read.assert_called_once_with("req", 11, "remote:1", [1], [2], 3)


def test_raiden_wrapper_routes_each_operation_to_its_dp_manager():
    engines = [mock.MagicMock() for _ in range(4)]
    for rank, engine in enumerate(engines):
        engine.get_local_endpoints.return_value = [
            {"endpoint": f"127.0.0.1:{7800 + rank}", "shards": [0]}
        ]
        engine.register_read.return_value = True
        engine.poll_stats.return_value = ([], [], [])
    manager_cls = mock.MagicMock(side_effect=engines)
    modules = {
        "tpu_raiden": types.ModuleType("tpu_raiden"),
        "tpu_raiden.api": types.ModuleType("tpu_raiden.api"),
        "tpu_raiden.api.jax": types.ModuleType("tpu_raiden.api.jax"),
        "tpu_raiden.api.jax.kv_cache_manager": types.ModuleType(
            "tpu_raiden.api.jax.kv_cache_manager"
        ),
    }
    modules["tpu_raiden.api.jax.kv_cache_manager"].KVCacheManager = manager_cls

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch(
            "sgl_jax.srt.disaggregation.raiden_transfer.wrapper._split_kv_caches_by_dp_rank",
            return_value={rank: [f"rank-{rank}"] for rank in range(4)},
        ),
    ):
        wrapper = RaidenTransferWrapper("127.0.0.1", 0, parallelism=2)
        wrapper.start([object()], max_blocks=8, num_slots=4, dp_size=4)
        wrapper.register_read("req", 7, [1], dp_rank=2)
        wrapper.start_read(
            "req",
            7,
            "remote:1",
            [1],
            [2],
            decode_dp_rank=3,
        )

    assert wrapper.dp_size == 4
    assert sorted(wrapper.endpoints_by_dp_rank) == [0, 1, 2, 3]
    engines[2].register_read.assert_called_once_with("req", 7, [1])
    engines[3].start_read.assert_called_once_with("req", 7, "remote:1", [1], [2], 2)


def test_raiden_wrapper_preserves_drained_events_when_one_rank_poll_fails(caplog):
    engines = [mock.MagicMock() for _ in range(3)]
    engines[0].poll_stats.return_value = (["sent-0"], [], ["failed-0"])
    engines[1].poll_stats.side_effect = RuntimeError("rank poll failed")
    engines[2].poll_stats.return_value = ([], ["received-2"], [])
    wrapper = RaidenTransferWrapper("127.0.0.1")
    wrapper._engines = {rank: engine for rank, engine in enumerate(engines)}

    with caplog.at_level("ERROR"):
        stats = wrapper.poll_stats()

    assert stats == (["sent-0"], ["received-2"], ["failed-0"])
    assert "Raiden poll_stats failed for dp_rank=1" in caplog.text


def test_rank_local_array_builds_real_views_for_each_data_rank():
    devices = jax.local_devices()
    if jax.process_count() != 1 or len(devices) < 2 or len(devices) % 2:
        pytest.skip("requires an even number of locally addressable JAX devices")
    mesh = Mesh(
        np.asarray(devices).reshape(2, len(devices) // 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, PartitionSpec("data", None))
    source = np.arange(32, dtype=np.int32).reshape(8, 4)
    array = jax.device_put(source, sharding)

    rank0 = _rank_local_array(array, dp_rank=0, dp_size=2)
    rank1 = _rank_local_array(array, dp_rank=1, dp_size=2)

    assert rank0.shape == rank1.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(rank0), source[:4])
    np.testing.assert_array_equal(np.asarray(rank1), source[4:])


def test_rank_local_array_rejects_kv_replicated_across_data_axis():
    devices = jax.local_devices()
    if jax.process_count() != 1 or len(devices) < 2 or len(devices) % 2:
        pytest.skip("requires an even number of locally addressable JAX devices")
    mesh = Mesh(
        np.asarray(devices).reshape(2, len(devices) // 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, PartitionSpec(None, None))
    array = jax.device_put(np.arange(32, dtype=np.int32).reshape(8, 4), sharding)

    with pytest.raises(ValueError, match="KV PartitionSpec"):
        _rank_local_array(array, dp_rank=0, dp_size=2)
