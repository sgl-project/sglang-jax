"""CPU regressions for the V4 Flash TPU7x scheduling boundary."""

import pytest

from sgl_jax.srt.kernels.mhc.tune import (
    collapse_vmem_bytes,
    post_vmem_bytes,
    select_collapse_block_tokens,
    select_gates_block_tokens,
    select_post_backend,
    select_post_block_tokens,
)


@pytest.mark.parametrize("device_kind", ["TPU7x", "TPU v7x", "v7x"])
def test_v7x_first_prefill_accepts_reported_device_kind(device_kind):
    assert (
        select_collapse_block_tokens(
            device_kind, tokens=128, hc_mult=4, hidden=4096, activation_bytes=2
        )
        == 128
    )
    assert select_gates_block_tokens(device_kind, tokens=128, hc_mult=4) == 128
    assert (
        select_post_backend(device_kind, tokens=128, hc_mult=4, hidden=4096, activation_bytes=2)
        == "xla"
    )


@pytest.mark.parametrize("activation_bytes,highest_precision", [(2, False), (4, True)])
def test_v7x_collapse_stays_within_scoped_vmem(activation_bytes, highest_precision):
    block = select_collapse_block_tokens(
        "TPU7x",
        tokens=4096,
        hc_mult=4,
        hidden=4096,
        activation_bytes=activation_bytes,
        highest_precision=highest_precision,
    )

    def cost(tokens):
        return collapse_vmem_bytes(
            tokens, hc_mult=4, hidden=4096, rows=24, activation_bytes=activation_bytes
        )

    assert cost(block) <= 32 * 1024 * 1024 < cost(2 * block)


def test_v7x_long_prefill_post_stays_within_scoped_vmem():
    block = select_post_block_tokens(
        "TPU7x", tokens=4096, hc_mult=4, hidden=4096, x_bytes=2, residual_bytes=2
    )

    def cost(tokens):
        return post_vmem_bytes(tokens, hc_mult=4, hidden=4096, x_bytes=2, residual_bytes=2)

    assert cost(block) <= 32 * 1024 * 1024 < cost(2 * block)


@pytest.mark.parametrize("tokens,v7x_backend", [(640, "xla"), (768, "pallas"), (769, "xla")])
def test_v7x_post_switches_at_its_own_xla_vmem_boundary(tokens, v7x_backend):
    kwargs = dict(tokens=tokens, hc_mult=4, hidden=4096, activation_bytes=2)
    assert select_post_backend("TPU7x", **kwargs) == v7x_backend
    assert select_post_backend("TPU v6e", **kwargs) == "xla"


def test_unknown_tpu_still_requires_an_explicit_schedule():
    with pytest.raises(ValueError, match="mHC has no schedule"):
        select_collapse_block_tokens(
            "unknown TPU", tokens=128, hc_mult=4, hidden=4096, activation_bytes=2
        )
