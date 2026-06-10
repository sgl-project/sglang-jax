"""Multimodal pad_value derivation (refactor M2 / design doc §3.6.3).

A per-item ``pad_value`` is baked into the placeholder rows of ``input_ids`` and serves
two roles: (a) the :func:`sgl_jax.srt.mm_core.merge.merge` scatter key, and (b) the
RadixAttention prefix-cache key (same media -> same hash -> same pad_value -> cache hit).

This ports upstream's ``MM_PAD_SHIFT`` regime to fix the as-is ``pad_value = hash % 2**24``
(`modality_enum.py`), which has no anti-collision guard: with a large vocab a pad_value
could equal a real text token id. The ``+ MM_PAD_SHIFT_VALUE`` base offset guarantees
every pad_value sits above the real token-id range.

Pure int math — no jax — so it is unit-testable on any interpreter.

WIRING NOTE (coordinated change, do together): pad_value is consumed in three places that
must agree — where it is written into input_ids (`pad_input_tokens`), where merge() reads
it (the scatter key), and the radix cache key. Swapping the as-is ``hash % 2**24`` for this
must update all three atomically and be validated (a multimodal request still scatters
correctly and still gets radix hits). See design doc §3.6.3 / Ch4 M2.
"""

from __future__ import annotations

MM_PAD_SHIFT_VALUE = 1_000_000
_PAD_MOD = 1 << 30  # 2**30


def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    """Raise if the model's vocab could collide with the pad_value range.

    pad_value lives in ``[MM_PAD_SHIFT_VALUE, MM_PAD_SHIFT_VALUE + 2**30)``. If the real
    token-id space reaches into ``MM_PAD_SHIFT_VALUE``, the base offset no longer guarantees
    separation and ``MM_PAD_SHIFT_VALUE`` must be raised.
    """
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"vocab_size={vocab_size} exceeds MM_PAD_SHIFT_VALUE={MM_PAD_SHIFT_VALUE}; "
            "multimodal pad_value could collide with a real token id. "
            "Raise MM_PAD_SHIFT_VALUE."
        )


def derive_pad_value(content_hash: int) -> int:
    """Map a content hash to a pad_value in ``[MM_PAD_SHIFT_VALUE, +2**30)``.

    ``content_hash`` is the item's content digest (e.g. the leading 8 bytes of a SHA-256
    over the feature, as the as-is code already computes for cross-device determinism).
    """
    return MM_PAD_SHIFT_VALUE + (content_hash % _PAD_MOD)
