"""Smoke coverage for PD-related ServerArgs and CLI flags.

This file intentionally stays narrower than the fuller end-to-end
validation suite. Its goal is to keep the CLI/config surface reviewable
on its own.
"""

from __future__ import annotations

import argparse

from sgl_jax.srt.server_args import ServerArgs


def _make_args(**overrides) -> ServerArgs:
    defaults = dict(
        model_path="dummy/model",
        device="cpu",
        random_seed=42,
        mem_fraction_static=0.5,
    )
    defaults.update(overrides)
    return ServerArgs(**defaults)


def test_pd_server_args_defaults_are_stable():
    args = _make_args()
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
