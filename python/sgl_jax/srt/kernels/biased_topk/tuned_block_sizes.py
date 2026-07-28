"""Tuned token block sizes for the biased top-k Pallas kernel."""

import jax


def _device_name() -> str:
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("not a TPU device")
    if kind.endswith(" lite"):
        return kind[: -len(" lite")] + "e"
    if kind == "TPU7x":
        return "TPU v7"
    return kind


# device_name -> {(T, E, k): block_tokens}
# TPU v7 E=384/k=8 entries are populated from the real-device tuner.
TUNED_BT: dict[str, dict[tuple[int, int, int], int]] = {
    "TPU v7": {
        (64, 384, 8): 64,
        (128, 384, 8): 128,
        (256, 384, 8): 256,
        (512, 384, 8): 512,
        (1024, 384, 8): 512,
        (2048, 384, 8): 512,
        (4096, 384, 8): 1024,
        (8192, 384, 8): 1024,
        (16384, 384, 8): 1024,
        (32768, 384, 8): 1024,
    },
}


def get_tuned_bt(tokens: int, experts: int, topk: int) -> int | None:
    """Return a measured block size for this exact routing shape."""
    try:
        device = _device_name()
    except Exception:  # noqa: BLE001
        return None
    return TUNED_BT.get(device, {}).get((tokens, experts, topk))
