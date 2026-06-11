import numpy as np


def publish_spec_decode_new_seq_lens(batch_output):
    new_seq_lens = batch_output.next_draft_input.new_seq_lens
    if new_seq_lens is not None and hasattr(new_seq_lens, "copy_to_host_async"):
        new_seq_lens.copy_to_host_async()
    return new_seq_lens


def can_use_spec_decode_overlap(enable_overlap, spec_algorithm, batch) -> bool:
    if not enable_overlap:
        return False
    if spec_algorithm is None or spec_algorithm.is_none():
        return False
    if batch.forward_mode.is_mixed():
        return False
    if not batch.forward_mode.is_decode():
        return False
    if any(info.decoding_reqs for info in batch.reqs_info):
        return False
    return not (batch.return_logprob or batch.return_output_logprob_only)


def can_use_spec_prefill_overlap(enable_overlap, spec_algorithm, batch) -> bool:
    if not enable_overlap:
        return False
    if spec_algorithm is None or spec_algorithm.is_none():
        return False
    if not batch.forward_mode.is_extend():
        return False
    return not (batch.return_logprob or batch.return_output_logprob_only)


def resolve_spec_decode_token_ids(result, batch, draft_token_num: int):
    """Resolve per-request accepted token ids from a speculative verify result."""
    if hasattr(result.next_token_ids, "copy_to_host_async"):
        result.next_token_ids.copy_to_host_async()
    if hasattr(result.accept_lens, "copy_to_host_async"):
        result.accept_lens.copy_to_host_async()
    next_token_ids = np.asarray(result.next_token_ids)
    accept_lens = np.asarray(result.accept_lens).tolist()
    per_dp_bs = batch.per_dp_bs_size
    total_bs = per_dp_bs * batch.dp_size
    predict_tokens: list[list[int]] = [[] for _ in range(total_bs)]
    for dp_rank, info in enumerate(batch.reqs_info):
        base = dp_rank * per_dp_bs
        for j, _req in enumerate(info.reqs or []):
            i = base + j
            a = accept_lens[i]
            predict_tokens[i] = next_token_ids[
                i * draft_token_num : i * draft_token_num + a
            ].tolist()
    return predict_tokens, accept_lens
