"""CLI + mesh checks for --dcp-size (Slice 1)."""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("pydantic")

from sgl_jax.srt.server_args import ServerArgs


def _parse(extra: list[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(["--model-path", "dummy", *extra])
    return ServerArgs.from_cli_args(namespace)


def test_dcp_size_default_is_one():
    args = _parse([])
    assert args.dcp_size == 1
    args.check_server_args()


def test_dcp_size_alias_and_long_name():
    assert _parse(["--dcp-size", "16"]).dcp_size == 16
    assert _parse(["--decode-context-parallel-size", "8"]).dcp_size == 8


def test_dcp_size_16_on_tp16_accepted():
    args = _parse(["--tp-size", "16", "--dcp-size", "16"])
    args.check_server_args()


def test_dcp_size_3_on_tp16_rejected():
    args = _parse(["--tp-size", "16", "--dcp-size", "3"])
    with pytest.raises(ValueError, match="must divide attention_tp"):
        args.check_server_args()


def test_dp16_dcp16_on_tp16_rejected():
    args = _parse(["--tp-size", "16", "--dp-size", "16", "--dcp-size", "16"])
    with pytest.raises(ValueError, match="must divide attention_tp"):
        args.check_server_args()
