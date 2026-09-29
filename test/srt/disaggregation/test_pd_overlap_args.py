"""CPU configuration gates for the experimental Raiden PD scheduler loops."""

import argparse

import pytest

from sgl_jax.srt.server_args import ServerArgs


def overlap_config(**overrides):
    config = {
        "model_path": "dummy",
        "device": "tpu",
        "disaggregation_mode": "decode",
        "disaggregation_bootstrap_url": "http://bootstrap",
        "page_size": 128,
        "disaggregation_enable_overlap_schedule": True,
        "disaggregation_use_raiden": True,
        "disable_radix_cache": True,
    }
    config.update(overrides)
    return config


def test_overlap_accepts_prefill_and_decode():
    for mode in ("prefill", "decode"):
        assert ServerArgs(
            **overlap_config(disaggregation_mode=mode)
        ).disaggregation_enable_overlap_schedule


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"disaggregation_mode": "null"}, "requires --disaggregation-mode"),
        ({"disaggregation_use_raiden": False}, "requires --disaggregation-use-raiden"),
        ({"disable_radix_cache": False}, "requires --disable-radix-cache"),
        ({"nnodes": 2}, "requires --nnodes=1"),
        ({"disable_overlap_schedule": True}, "conflicts with --disable-overlap-schedule"),
    ],
)
def test_overlap_rejects_unsupported_configuration(override, error):
    with pytest.raises(ValueError, match=error):
        ServerArgs(**overlap_config(**override))


def test_overlap_disabled_preserves_existing_default():
    args = ServerArgs(model_path="dummy", device="tpu")
    assert not args.disaggregation_enable_overlap_schedule


def test_overlap_cli_opt_in_and_explicit_disable():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    base = ["--model-path", "dummy"]
    assert not parser.parse_args(base).disaggregation_enable_overlap_schedule
    assert parser.parse_args(
        base + ["--disaggregation-enable-overlap-schedule"]
    ).disaggregation_enable_overlap_schedule
    assert not parser.parse_args(
        base
        + [
            "--disaggregation-enable-overlap-schedule",
            "--no-disaggregation-enable-overlap-schedule",
        ]
    ).disaggregation_enable_overlap_schedule
