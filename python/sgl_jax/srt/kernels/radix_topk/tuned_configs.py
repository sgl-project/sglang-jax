"""Shape-tuned configurations for the SparseCore radix top-k kernel."""

from __future__ import annotations

import dataclasses

import jax

SUPPORTED_RADIX_DIGIT_CONFIGS = frozenset({(8, 4), (4, 8)})


@dataclasses.dataclass(frozen=True)
class RadixTopKConfig:
    """Static radix-selection parameters that preserve exact FP32 semantics."""

    num_seq_windows: int = 1
    digit_width: int = 8
    num_digits: int = 4
    use_tc_tiling_on_sc: bool = False

    def __post_init__(self):
        if self.num_seq_windows != 1:
            raise ValueError("exact radix top-k currently supports only num_seq_windows=1")
        digit_config = (self.digit_width, self.num_digits)
        if digit_config not in SUPPORTED_RADIX_DIGIT_CONFIGS:
            raise ValueError(
                f"unsupported (digit_width, num_digits)={digit_config}; "
                f"expected one of {sorted(SUPPORTED_RADIX_DIGIT_CONFIGS)}"
            )
        if self.use_tc_tiling_on_sc and self.digit_width == 4:
            raise ValueError("TC tiling is not supported for the 4x8 radix configuration")

    @property
    def input_alignment(self) -> int:
        """Required long-row alignment for 16 subcores with 16 lanes on TPU v7."""

        return 256 * self.num_seq_windows


DEFAULT_RADIX_TOPK_CONFIG = RadixTopKConfig()


def make_radix_topk_config(
    *,
    num_seq_windows: int,
    digit_width: int,
    num_digits: int,
    use_tc_tiling_on_sc: bool,
) -> RadixTopKConfig:
    """Build one of the exact digit configurations supported by the kernel."""

    return RadixTopKConfig(
        num_seq_windows=num_seq_windows,
        digit_width=digit_width,
        num_digits=num_digits,
        use_tc_tiling_on_sc=use_tc_tiling_on_sc,
    )


def _device_name() -> str:
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("not a TPU device")
    if kind.endswith(" lite"):
        return kind[: -len(" lite")] + "e"
    if kind == "TPU7x":
        return "TPU v7"
    return kind


# device_name -> {(score_size, topk): RadixTopKConfig}
# Keep entries measurement-backed: paste tuner output here after a TPU sweep.
TUNED_RADIX_TOPK_CONFIGS: dict[str, dict[tuple[int, int], RadixTopKConfig]] = {
    "TPU v7": {
        (135168, 2048): RadixTopKConfig(
            num_seq_windows=1,
            digit_width=8,
            num_digits=4,
            use_tc_tiling_on_sc=True,
        ),
    },
}


def get_tuned_radix_topk_config(score_size: int, topk: int) -> RadixTopKConfig | None:
    """Return the measured config for this exact ``(score_size, topk)`` pair."""

    try:
        device = _device_name()
    except Exception:  # noqa: BLE001
        return None
    return TUNED_RADIX_TOPK_CONFIGS.get(device, {}).get((score_size, topk))
