"""CORE: vision/audio-tower weight-replication contract (design §3.3.5 / §5.7 G2-a).

Understanding-plane encoders (ViT / audio tower) must run FULLY REPLICATED under the
multi-chip AR mesh. They were authored for the staged 1-device mesh (where sharding is
trivial); their dynamic shapes and spatial reshapes do not survive a real TP split, so the
in-model path runs them replicated (design §3.3.5). The single source of truth for "this
mapping loads replicated" is :func:`replicate_mappings`; :func:`assert_replicated` is the
load-time guard that catches a tower mapping that slipped through with a TP axis (the failure
mode review §5.7 G2-(a) calls out: relying on name-inference instead of an explicit ()).
"""

from __future__ import annotations

import dataclasses


def replicate_mappings(mappings: dict) -> dict:
    """Return ``mappings`` with every WeightMapping's ``sharding`` forced to all-None
    (replicated), preserving ``target_path`` / ``transpose``. Idempotent. Use for ViT /
    audio-tower mappings so the encoder loads replicated regardless of the source mapping's
    declared sharding (which was written for the staged 1-device mesh)."""
    out = {}
    for k, m in mappings.items():
        sh = getattr(m, "sharding", None)
        if sh:
            m = dataclasses.replace(m, sharding=tuple(None for _ in sh))
        out[k] = m
    return out


def assert_replicated(mappings: dict, *, where: str = "vision/audio tower") -> None:
    """Raise if any mapping is not fully replicated (some axis names a real mesh axis). The
    encoders must run replicated (design §5.7 G2-a); a TP-sharded tower mapping would break the
    spatial reshapes under the AR mesh. Call at load time, after composing the tower mappings
    (and ideally after :func:`replicate_mappings`), as a regression guard."""
    bad = [
        (k, m.sharding)
        for k, m in mappings.items()
        if getattr(m, "sharding", None) is not None and not all(ax is None for ax in m.sharding)
    ]
    if bad:
        raise AssertionError(
            f"{where} weight mappings must be fully replicated (sharding all-None), but "
            f"{len(bad)} are TP-sharded, e.g. {bad[:3]}. Pass them through "
            f"mm_core.weights.replicate_mappings()."
        )
