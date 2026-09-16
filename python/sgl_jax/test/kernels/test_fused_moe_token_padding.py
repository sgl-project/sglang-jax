"""Host-only contracts for fused-MoE token padding."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sgl_jax.srt.kernels.fused_moe.token_padding import (
    align_fused_moe_v1_local_tokens,
    align_fused_moe_v1_num_tokens,
)


@pytest.mark.parametrize(
    ("logical", "packing", "physical"),
    [
        (1, 1, 2),
        (1, 2, 2),
        (1, 4, 4),
        (2, 1, 2),
        (2, 2, 2),
        (2, 4, 4),
        (3, 1, 4),
        (3, 2, 4),
        (3, 4, 4),
        (4, 1, 4),
        (4, 2, 4),
        (4, 4, 4),
        (5, 1, 8),
        (5, 2, 8),
        (5, 4, 8),
        (6, 2, 8),
        (7, 2, 8),
        (8, 2, 8),
        (9, 2, 16),
        (127, 2, 128),
        (128, 2, 128),
    ],
)
def test_v1_local_token_alignment(logical: int, packing: int, physical: int) -> None:
    assert align_fused_moe_v1_local_tokens(logical, packing) == physical


@pytest.mark.parametrize(("logical", "packing"), [(0, 2), (-1, 2), (1, 0), (1, -1)])
def test_v1_local_token_alignment_rejects_non_positive_inputs(logical: int, packing: int) -> None:
    with pytest.raises(ValueError):
        align_fused_moe_v1_local_tokens(logical, packing)


def test_odd_local_extent_keeps_logical_bucket_while_aligning_each_ep_rank() -> None:
    # Observed on TPU v6e-16 at EP16: the 2032 precompile bucket gives 2032 // 16 = 127
    # local tokens, which the v1 kernel rejects for bf16 (t_packing=2).
    logical_global_tokens = 2032
    ep_size = 16
    logical_local_tokens = logical_global_tokens // ep_size
    physical_launch_by_logical_bucket = {
        bucket: align_fused_moe_v1_num_tokens(bucket, ep_size, t_packing=2)
        for bucket in (2032, 2048)
    }

    assert logical_local_tokens == 127
    assert align_fused_moe_v1_local_tokens(logical_local_tokens, t_packing=2) == 128
    assert physical_launch_by_logical_bucket == {2032: 2048, 2048: 2048}
    assert list(physical_launch_by_logical_bucket) == [2032, 2048]


@pytest.mark.parametrize(
    ("num_tokens", "ep_size", "packing"),
    [(0, 16, 2), (-1, 16, 2), (2032, 0, 2), (2031, 16, 2), (2052, 16, 2)],
)
def test_global_alignment_rejects_invalid_contract(
    num_tokens: int,
    ep_size: int,
    packing: int,
) -> None:
    with pytest.raises(ValueError):
        align_fused_moe_v1_num_tokens(num_tokens, ep_size, packing)


def test_import_does_not_load_jax() -> None:
    code = """
import sys
assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
import sgl_jax.srt.kernels.fused_moe.token_padding
assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
"""
    python_root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=python_root)
