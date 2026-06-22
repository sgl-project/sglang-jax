"""Tuned `block_tokens` (BT) for `grouped_topk_pallas`, keyed by device + routing shape.

Populated by `benchmark/kernels/grouped_topk/tune_grouped_topk_bt.py` on real TPU. BT is the
strongest perf lever for this kernel (bigger BT = fewer grid steps = less launch/pipeline
overhead); the tuned value is the largest BT that fits VMEM. Lookup falls back to None on a miss
so the caller can use a safe default.

Key: (next_power_of_2(T_local), E, G, Gtop, k)  where T_local is the per-device token count.
Source: exp-gu55bjx7yd (v7x, single-host v7x-8). +29%..+66% over block_tokens=512 for T>=1024.

Self-contained on purpose (only `jax`): keeps the grouped_topk package dependency-light so the
kernel stays embeddable / fast to import.
"""

import logging

import jax

logger = logging.getLogger(__name__)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _device_name() -> str:
    """Normalized TPU name, mirrors sgl_jax.srt.utils.jax_utils.get_device_name (e.g. 'TPU v7')."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("not a TPU device")
    if kind.endswith(" lite"):
        return kind[: -len(" lite")] + "e"
    if kind == "TPU7x":
        return "TPU v7"
    if kind and kind[-1] in ("e", "p"):
        return kind  # already e.g. "TPU v6e" / "TPU v5p"
    return kind


# device_name -> {(pow2(T), E, G, Gtop, k): BT}
TUNED_BT: dict[str, dict[tuple, int]] = {
    "TPU v7": {
        # E=256 (DeepSeek-V3 / Ling)
        (64, 256, 8, 4, 8): 64,
        (128, 256, 8, 4, 8): 128,
        (256, 256, 8, 4, 8): 256,
        (512, 256, 8, 4, 8): 512,
        (1024, 256, 8, 4, 8): 1024,
        (2048, 256, 8, 4, 8): 2048,
        (4096, 256, 8, 4, 8): 2048,  # BT=4096 exceeds v7x scoped VMEM with padded outputs
        (8192, 256, 8, 4, 8): 2048,  # multi-block double-buffer caps BT at 2048
        (16384, 256, 8, 4, 8): 2048,
        (32768, 256, 8, 4, 8): 2048,
        # E=512 (MaxText)
        (64, 512, 8, 4, 8): 64,
        (128, 512, 8, 4, 8): 128,
        (256, 512, 8, 4, 8): 256,
        (512, 512, 8, 4, 8): 512,
        (1024, 512, 8, 4, 8): 1024,
        (2048, 512, 8, 4, 8): 2048,
        (4096, 512, 8, 4, 8): 2048,  # E=512 doubles [BT,E] VMEM -> 4096 OOMs, cap 2048
        (8192, 512, 8, 4, 8): 2048,
        (16384, 512, 8, 4, 8): 2048,
        (32768, 512, 8, 4, 8): 2048,
    },
}

_WARNED_MISSES: set[tuple] = set()


def get_tuned_bt(T: int, E: int, G: int, Gtop: int, k: int) -> int | None:
    """Tuned block_tokens for this device + (T_local, E, G, Gtop, k), or None on miss.

    Returns None (not an error) on a non-TPU device or an unseen shape, so callers fall back to a
    safe default.
    """
    try:
        device = _device_name()
    except Exception:  # noqa: BLE001  (non-TPU device, etc.)
        return None
    key = (_next_power_of_2(T), E, G, Gtop, k)
    bt = TUNED_BT.get(device, {}).get(key)
    if bt is None and (device, key) not in _WARNED_MISSES:
        _WARNED_MISSES.add((device, key))
        logger.warning(
            "grouped_topk: no tuned block_tokens for device=%s key=(pow2T=%d,E=%d,G=%d,Gtop=%d,k=%d)"
            "; using default. Run benchmark/kernels/grouped_topk/tune_grouped_topk_bt.py to tune.",
            device,
            key[0],
            E,
            G,
            Gtop,
            k,
        )
    return bt
