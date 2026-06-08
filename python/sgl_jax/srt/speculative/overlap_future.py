from dataclasses import dataclass


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
