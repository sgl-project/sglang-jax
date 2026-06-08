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


class SpecDecodePublishFields:
    def __init__(self, *, new_seq_lens):
        self._new_seq_lens = new_seq_lens
        self._new_seq_lens_host = None

        _prefetch_to_host(new_seq_lens)

    @property
    def new_seq_lens(self) -> np.ndarray | None:
        if self._new_seq_lens_host is None:
            self._new_seq_lens_host = _device_to_numpy(self._new_seq_lens)
        return self._new_seq_lens_host


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


def _prefetch_to_host(value):
    if value is None:
        return
    if hasattr(value, "copy_to_host_async"):
        value.copy_to_host_async()


def _device_to_numpy(value):
    if value is None:
        return None
    _prefetch_to_host(value)
    return np.asarray(jax.device_get(value))


def publish_spec_decode_new_seq_lens(future_result):
    new_seq_lens = getattr(future_result, "new_seq_lens", None)
    if new_seq_lens is None and getattr(future_result, "next_draft_input", None) is not None:
        new_seq_lens = future_result.next_draft_input.new_seq_lens
    return SpecDecodePublishFields(
        new_seq_lens=new_seq_lens,
    )


def can_use_spec_decode_overlap(enable_overlap, spec_algorithm, batch) -> bool:
    if not enable_overlap:
        return False
    if spec_algorithm is None or spec_algorithm.is_none():
        return False
    if not batch.forward_mode.is_decode():
        return False
    return not (batch.return_logprob or batch.return_output_logprob_only)
