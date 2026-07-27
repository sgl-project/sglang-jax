"""CPU contract tests for the optional tpu-raiden PD data plane."""

from __future__ import annotations

import argparse
import sys
import types
from unittest import mock

import pytest

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.base.transfer import (
    AdmissionState,
    DecodeTransferContext,
)
from sgl_jax.srt.disaggregation.bootstrap import _Registry, check_prefill_compat
from sgl_jax.srt.disaggregation.raiden_transfer.conn import (
    RaidenMetadata,
    RaidenTransferKVManager,
    _uuid_to_int,
)
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import (
    RaidenTransferWrapper,
)
from sgl_jax.srt.server_args import ServerArgs


class _FakeBootstrap:
    def __init__(self) -> None:
        self.registered: list[tuple[tuple, dict]] = []
        self.popped: list[tuple[int, dict]] = []
        self.transfer_info = None

    def register_transfer(self, *args, **kwargs) -> None:
        self.registered.append((args, kwargs))

    def pop_transfer(self, room: int, **kwargs) -> None:
        self.popped.append((room, kwargs))

    def get_transfer_info(self, room: int, **kwargs):  # noqa: ARG002
        return self.transfer_info


class _FakeRaiden:
    endpoints = [{"endpoint": "10.0.0.1:7777", "shards": [0]}]
    control_port = 7777

    def __init__(self) -> None:
        self.registered: list[tuple] = []
        self.started: list[tuple] = []
        self.stats = ([], [], [])

    def register_read(self, *args):
        self.registered.append(args)
        return True

    def start_read(self, *args):
        self.started.append(args)

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
    sender.attach_block_ids([3, 8, 13], bootstrap_room=42)

    sender.send()

    assert raiden.registered == [("wire-1", _uuid_to_int("wire-1"), [3, 8, 13])]
    assert bootstrap.registered[0][0] == (42, "wire-1")
    assert bootstrap.registered[0][1] == {
        "jax_process_index": 0,
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
        )
    )

    assert receiver.poll() == KVPoll.TRANSFERRING
    assert raiden.started == [("wire-2", _uuid_to_int("wire-2"), "10.0.0.1:7777", [1, 4], [9, 10])]

    raiden.stats = ([], ["wire-2"], [])
    assert receiver.poll() == KVPoll.SUCCESS
    assert bootstrap.popped == [(43, {"jax_process_index": 0})]
    assert "req-2" not in manager._receivers


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
    assert bootstrap.popped == [(44, {"jax_process_index": 0})]


def test_raiden_manager_owns_decode_admission_and_endpoint_mapping():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-4",
        "transport_metadata": {"remote_block_ids": [3, 4]},
    }
    manager = _manager(raiden, bootstrap)
    context = DecodeTransferContext(
        req_id="req-4",
        transfer_id="wire-4",
        bootstrap_room=45,
        peer_info={
            "host": "10.0.0.1",
            "transport_metadata": {
                "engine": "raiden",
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
    assert raiden.started == [("wire-4", _uuid_to_int("wire-4"), "10.0.0.1:7777", [3, 4], [9, 10])]


def test_raiden_manager_accepts_legacy_flat_peer_metadata():
    raiden = _FakeRaiden()
    bootstrap = _FakeBootstrap()
    bootstrap.transfer_info = {
        "transfer_id": "wire-legacy",
        "transport_metadata": {"remote_block_ids": [3]},
    }
    manager = _manager(raiden, bootstrap)

    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-legacy",
            transfer_id="wire-legacy",
            bootstrap_room=47,
            peer_info={
                "host": "10.0.0.1",
                "local_control_port": 7777,
                "raiden_endpoints_json": '[{"endpoint":"10.0.0.1:7777","shards":[0]}]',
            },
            kv_indices=[18, 19],
            page_size=2,
            prompt_tokens=2,
            spec_factory=lambda: None,
        )
    )

    assert admission.state == AdmissionState.ADMITTED


def test_raiden_manager_defers_until_request_metadata_is_published():
    manager = _manager(_FakeRaiden(), _FakeBootstrap())
    admission = manager.try_start_decode(
        DecodeTransferContext(
            req_id="req-5",
            transfer_id="wire-5",
            bootstrap_room=46,
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

    with pytest.raises(RuntimeError, match="valid control endpoint"):
        _manager(raiden, _FakeBootstrap()).prefill_transport_metadata()


def test_raiden_uuid_is_stable_and_json_safe():
    assert _uuid_to_int("wire") == _uuid_to_int("wire")
    assert _uuid_to_int("wire") != _uuid_to_int("other")
    assert 0 <= _uuid_to_int("wire") < 2**50


def test_transfer_metadata_is_single_shot_reusable_and_ttl_bounded():
    now = [100.0]
    registry = _Registry(clock=lambda: now[0], transfer_ttl_seconds=5.0)
    registry.register_transfer(
        {
            "bootstrap_room": 7,
            "transfer_id": "first",
            "remote_block_ids": [1, 2],
        }
    )
    assert registry.get_transfer(7)["transfer_id"] == "first"

    registry.register_transfer(
        {
            "bootstrap_room": 7,
            "transfer_id": "second",
            "jax_process_index": 0,
            "transport_metadata": {"remote_block_ids": [9]},
        }
    )
    registry.register_transfer(
        {
            "bootstrap_room": 7,
            "transfer_id": "peer",
            "jax_process_index": 1,
            "transport_metadata": {"remote_block_ids": [11]},
        }
    )
    assert registry.get_transfer(7, 0)["transfer_id"] == "second"
    assert registry.get_transfer(7, 1)["transfer_id"] == "peer"
    registry.pop_room(7, 0)
    assert registry.get_transfer(7, 0) is None
    assert registry.get_transfer(7, 1)["transfer_id"] == "peer"
    now[0] += 6.0
    assert registry.get_transfer(7, 1) is None


def test_prefill_decode_transfer_engines_must_match():
    info = {
        "protocol_version": 3,
        "page_size": 128,
        "kv_dtype": "bfloat16",
        "transport_metadata": {"engine": "raiden"},
    }
    check_prefill_compat(
        info,
        local_page_size=128,
        local_kv_dtype="bfloat16",
        expected_transfer_engine="raiden",
    )
    with pytest.raises(ValueError, match="engine mismatch"):
        check_prefill_compat(
            info,
            local_page_size=128,
            local_kv_dtype="bfloat16",
            expected_transfer_engine="jax",
        )


def test_raiden_cli_is_opt_in_and_exposes_control_port():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    defaults = parser.parse_args(["--model-path", "dummy"])
    selected = parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--disaggregation-use-raiden",
            "--disaggregation-raiden-control-port",
            "7777",
        ]
    )
    assert defaults.disaggregation_use_raiden is False
    assert selected.disaggregation_use_raiden is True
    assert selected.disaggregation_raiden_control_port == 7777


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
