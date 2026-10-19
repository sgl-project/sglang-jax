"""Indexer geometry derived from the compressed page and attention query tile."""

from dataclasses import dataclass

from sgl_jax.srt.kernels.csa_attention.tune import LANES


@dataclass(frozen=True)
class CSAIndexerSchedule:
    kv_pages_per_block: int
    query_tile: int
    decode_request_tile: int = 1


def get_indexer_schedule(page_size: int, attention_schedule) -> CSAIndexerSchedule:
    if page_size not in (128, 256):
        raise ValueError("CSA page_size must be 128 or 256")
    return CSAIndexerSchedule(
        kv_pages_per_block=LANES // (page_size // 4),
        query_tile=attention_schedule.query_tile,
    )
