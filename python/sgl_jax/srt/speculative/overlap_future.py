from dataclasses import dataclass

import jax
import numpy as np


@dataclass
class SpecDecodeFutureResult:
    logits_output: object
    next_token_ids: object
    accept_lens: object
    new_seq_lens: object
    allocate_lens: object
    next_draft_input: object
    bid: int
    cache_miss_count: int


@dataclass
class SpecDecodeSchedulerFields:
    next_token_ids: np.ndarray
    accept_lens: np.ndarray | None
    new_seq_lens: np.ndarray | None


def make_spec_decode_future_result(batch_output):
    next_draft_input = batch_output.next_draft_input
    return SpecDecodeFutureResult(
        logits_output=batch_output.logits_output,
        next_token_ids=batch_output.next_token_ids,
        accept_lens=batch_output.accept_lens,
        new_seq_lens=next_draft_input.new_seq_lens,
        allocate_lens=batch_output.allocate_lens,
        next_draft_input=next_draft_input,
        bid=batch_output.bid,
        cache_miss_count=batch_output.cache_miss_count,
    )


def _device_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "copy_to_host_async"):
        value.copy_to_host_async()
    return np.asarray(jax.device_get(value))


def resolve_spec_decode_scheduler_fields(future_result):
    return SpecDecodeSchedulerFields(
        next_token_ids=_device_to_numpy(future_result.next_token_ids),
        accept_lens=_device_to_numpy(future_result.accept_lens),
        new_seq_lens=_device_to_numpy(future_result.new_seq_lens),
    )
