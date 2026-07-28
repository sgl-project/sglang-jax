import numpy as np


def resolve_spec_prefill_token_ids(result):
    """Resolve prefill next-token ids prepared by the producer path."""
    token_ids_arr = result.next_token_ids
    if token_ids_arr is None:
        raise RuntimeError(
            "Spec prefill overlap result is missing prepared next_token_ids "
            f"for bid={getattr(result, 'bid', None)}"
        )
    if hasattr(token_ids_arr, "copy_to_host_async"):
        token_ids_arr.copy_to_host_async()
    return np.asarray(token_ids_arr).tolist()


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
    # TODO(spec-overlap): neither the fused overlap verify kernel
    # (draft_extend_fused.spec_decode_verify) nor the non-overlap verify path
    # currently applies grammar / vocab_mask / penalty constraints during
    # speculative verification. Batches carrying these constraints will have
    # them silently ignored on the spec decode path. We do NOT exclude them in
    # this gate because the non-overlap fallback has the same gap, so routing
    # them off the overlap path would not make them correct. Revisit once
    # constrained speculative verify is implemented.
    return not (batch.return_logprob or batch.return_output_logprob_only)


def can_use_spec_prefill_overlap(enable_overlap, spec_algorithm, batch) -> bool:
    # Cheap pre-filter only. The authoritative check is
    # EAGLEWorker._can_use_fused_spec_prefill(model_worker_batch), which the
    # scheduler ANDs in before dispatching to the fused prefill overlap entry
    # (it needs the merged model_worker_batch sampling_info plus the worker's
    # NEXTN/topk/num_steps config). Keep this gate to the conditions decidable
    # from the ScheduleBatch alone so the two never drift.
    if not enable_overlap:
        return False
    if spec_algorithm is None or spec_algorithm.is_none():
        return False
    if not batch.forward_mode.is_extend():
        return False
    return not (batch.return_logprob or batch.return_output_logprob_only)


def use_legacy_eagle3_non_overlap(enable_overlap, spec_algorithm) -> bool:
    return (
        not enable_overlap
        and spec_algorithm is not None
        and not spec_algorithm.is_none()
        and spec_algorithm.is_eagle3()
    )


def can_merge_spec_non_overlap_prefill(enable_overlap, spec_algorithm) -> bool:
    """Whether non-overlap spec decode may merge completed prefills into decode.

    Eagle3 historically used the ``use_legacy_eagle3_non_overlap`` path for
    this. DFlash needs the same scheduling behavior to keep decode batches
    full, but it must keep its own accepted-length KV accounting, so do not
    fold it into the legacy Eagle3 helper.
    """
    return (
        not enable_overlap
        and spec_algorithm is not None
        and not spec_algorithm.is_none()
        and (spec_algorithm.is_eagle3() or spec_algorithm.is_dflash())
    )


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
