"""Plain XLA execution of the CSA threshold bisection for numerical checks."""

from sgl_jax.srt.kernels.dsv4.topk_threshold import (
    _bisect,
    _from_signed,
    _to_signed,
    score_key,
)


def topk_threshold_ref(scores, k: int):
    """Reference / fallback: same bisection in plain XLA. Returns uint32 [T]."""
    skey = _to_signed(score_key(scores))
    return _from_signed(_bisect(skey, k))[:, 0]
