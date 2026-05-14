import jax
import jax.numpy as jnp


def create_extend_after_decode_spec_info(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
):
    accept_lens_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(accept_lens[:-1])])

    def compute_position_updates():
        bs = seq_lens.shape[0]
        # total_positions = jnp.sum(accept_lens)

        batch_ids = jnp.repeat(jnp.arange(bs), accept_lens)
        within_batch_offsets = jnp.concatenate([jnp.arange(accept_lens[i]) for i in range(bs)])

        position_indices = accept_lens_cumsum[batch_ids] + within_batch_offsets

        position_values = seq_lens[batch_ids] - accept_lens[batch_ids] + within_batch_offsets

        return position_indices, position_values

    position_indices, position_values = compute_position_updates()
    positions_updated = positions.at[position_indices].set(position_values)

    verified_id_indices = accept_lens_cumsum + accept_lens - 1
    verified_id_data = verified_id[verified_id_indices]
    new_verified_id_updated = new_verified_id.at[: len(seq_lens)].set(verified_id_data)

    return positions_updated, new_verified_id_updated


def verify_tree_greedy(
    predicts: jax.Array,
    accept_index: jax.Array,
    accept_token_num: jax.Array,
    candidates: jax.Array,
    retrive_index: jax.Array,
    retrive_next_token: jax.Array,
    retrive_next_sibling: jax.Array,
    target_predict: jax.Array,
):
    """Verify the draft tree with greedy sample policy.

    Args:
      predicts: draft probabilities, this array will be modified and return. Shape: (bs*draft_token_num,).
      accept_index: the index of accept token, this array will be modified and return. Shape: (bs, num_spec_step).
      accept_token_num: accept token number, this array will be modified and return. Shape: (bs,)
      candidates: candidate draft tokens # shape: (bs, draft_token_num)
      retrive_index: store the predict array index of token, shape: (bs, draft_token_num)
      retrive_next_token: store the first children in tree, shape: (bs, draft_token_num)
      retrive_next_sibling: store the next brother node in tree, shape: (bs, draft_token_num)
      target_predict: the probs from target model, shape: (bs*draft_token_num,)

    Returns:
      predicts: draft probabilities, shape: (bs*draft_token_num,).
      accept_index: the index of accept token, shape: (bs, num_spec_step).
      accept_token_num: accept token number, shape: (bs,)
    """
    num_speculative_tokens = accept_index.shape[1]
    for bid, _ in enumerate(candidates):
        last_accepted_retrive_idx = retrive_index[bid, 0]
        accept_index = accept_index.at[bid, 0].set(last_accepted_retrive_idx)
        num_accepted_tokens = 0
        cur_index = 0
        for j in range(1, num_speculative_tokens):
            cur_index = retrive_next_token[bid][cur_index]
            while cur_index != -1:
                draft_index = retrive_index[bid, cur_index]
                draft_token_id = candidates[bid, cur_index]
                target_token_id = target_predict[last_accepted_retrive_idx]
                if draft_token_id == target_token_id:
                    predicts = predicts.at[last_accepted_retrive_idx].set(target_token_id)
                    num_accepted_tokens += 1
                    accept_index = accept_index.at[bid, num_accepted_tokens].set(draft_index)
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    cur_index = retrive_next_sibling[bid][cur_index]
            if cur_index == -1:
                break
        accept_token_num = accept_token_num.at[bid].set(num_accepted_tokens)
        predicts = predicts.at[last_accepted_retrive_idx].set(
            target_predict[last_accepted_retrive_idx]
        )
    return accept_index, accept_token_num, predicts


def top_k_top_p_renorm_prob(probs, top_k_values, top_p_values, max_k=1024):
    """Fused top-k and top-p renormalization in a single pass.

    This is more efficient than calling top_k_renorm_prob followed by top_p_renorm_prob
    because it:
    1. Only calls jax.lax.top_k once (instead of twice)
    2. Only reconstructs the full distribution once (instead of twice)
    3. Better cache locality by processing both filters together

    Complexity: O(V + k log k) for the fused operation.

    Args:
      probs: probabilities, shape: (batch_size, num_classes).
      top_k_values: the top-k threshold, shape: (batch_size,).
      top_p_values: the top-p threshold, shape: (batch_size,).
      max_k: maximum k value for static compilation (default: 1024).
             Will be clamped to vocab_size if larger.

    Returns:
      Renormalized probabilities, shape ``(batch_size, num_classes)``.
    """
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert (
        probs.shape[0] == top_k_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_k_values.shape[0]: {top_k_values.shape}"
    assert (
        probs.shape[0] == top_p_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_p_values.shape[0]: {top_p_values.shape}"

    # Clamp max_k to vocab_size
    vocab_size = probs.shape[1]
    effective_max_k = min(max_k, vocab_size)

    def process_single_sample(prob_row, k, p):
        # Step 1: Extract top-k candidates (single top_k call)
        top_k_probs, top_k_indices = jax.lax.top_k(prob_row, effective_max_k)

        # Step 2: Apply top-k filter
        mask_k = jnp.arange(effective_max_k) < k
        filtered_probs = jnp.where(mask_k, top_k_probs, 0.0)

        # Step 3: Apply top-p filter on the top-k filtered results
        cumsum_probs = jnp.cumsum(filtered_probs)
        total_prob = jnp.sum(filtered_probs)

        # Find cutoff index for top-p
        # When p >= 1.0, keep all top-k tokens
        cutoff_idx = jnp.where(
            p >= 1.0,
            jnp.sum(mask_k) - 1,  # Keep all k tokens
            jnp.argmax(cumsum_probs >= p * total_prob),
        )

        # Step 4: Combine both masks
        mask_p = jnp.arange(effective_max_k) <= cutoff_idx
        final_mask = mask_k & mask_p
        selected_probs = jnp.where(final_mask, top_k_probs, 0.0)

        # Step 5: Reconstruct full distribution (only once!)
        result = jnp.zeros(prob_row.shape, dtype=prob_row.dtype)
        result = result.at[top_k_indices].set(selected_probs)

        return result / jnp.sum(result)

    return jax.vmap(process_single_sample, in_axes=(0, 0, 0))(probs, top_k_values, top_p_values)


def top_k_renorm_prob(probs, top_k_values, max_k=1024):
    """Renormalizing probabilities by top-k thresholding.

    Optimized implementation using jax.lax.top_k instead of full sorting.
    Complexity: O(V + k log k) instead of O(V log V), where V is vocab_size.

    Args:
      probs: probabilities, shape: (batch_size, num_classes).
      top_k_values: the top-k threshold for re-normalizing probabilities, shape: (batch_size,).
      max_k: maximum k value for static compilation (default: 1024).
             Will be clamped to vocab_size if larger.

    Returns:
      Renormalized probabilities, shape ``(batch_size, num_classes)``.
    """
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert (
        probs.shape[0] == top_k_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_k_values.shape[0]: {top_k_values.shape}"

    # Clamp max_k to vocab_size
    vocab_size = probs.shape[1]
    effective_max_k = min(max_k, vocab_size)

    def process_single_sample(prob_row, k):
        # Use jax.lax.top_k for efficient partial sorting
        # Note: effective_max_k must be static for JIT compilation
        top_k_probs, top_k_indices = jax.lax.top_k(prob_row, effective_max_k)

        # Dynamically mask based on actual k value
        mask = jnp.arange(effective_max_k) < k
        selected_probs = jnp.where(mask, top_k_probs, 0.0)

        # Reconstruct full probability distribution
        result = jnp.zeros_like(prob_row)
        result = result.at[top_k_indices].set(selected_probs)

        return result / jnp.sum(result)

    return jax.vmap(process_single_sample, in_axes=(0, 0))(probs, top_k_values)


def top_p_renorm_prob(probs, top_p_values, max_top_k=1024):
    """Renormalizing probabilities by top-p thresholding.

    Optimized implementation using jax.lax.top_k for pre-filtering before top-p.
    Complexity: O(V + k log k) instead of O(V log V), where V is vocab_size.

    Args:
      probs: probabilities, shape: (batch_size, num_classes).
      top_p_values: the top-p threshold for re-normalizing probabilities, shape: (batch_size,).
      max_top_k: maximum number of top tokens to consider (default: 1024).
                 Pre-filters to top max_top_k tokens before applying top-p.
                 Will be clamped to vocab_size if larger.

    Returns:
      Renormalized probabilities, shape ``(batch_size, num_classes)``.
    """
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert (
        probs.shape[0] == top_p_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_k_values.shape[0]: {top_p_values.shape}"

    # Clamp max_top_k to vocab_size
    vocab_size = probs.shape[1]
    effective_max_k = min(max_top_k, vocab_size)

    def process_single_sample(prob_row, top_p):
        # Step 1: Pre-filter using top_k (avoids full sorting)
        top_k_probs, top_k_indices = jax.lax.top_k(prob_row, effective_max_k)

        # Step 2: Apply top-p on the filtered top_k results
        cumsum_probs = jnp.cumsum(top_k_probs)

        # Find the cutoff index
        # Special handling for top_p >= 1.0: keep all non-zero tokens
        num_nonzero = jnp.sum(top_k_probs > 0)
        cutoff_idx = jnp.where(
            top_p >= 1.0,
            num_nonzero - 1,  # Keep all non-zero tokens
            jnp.argmax(cumsum_probs >= top_p),
        )

        # Step 3: Create mask for tokens within top-p threshold
        mask_in_topk = jnp.arange(effective_max_k) <= cutoff_idx
        selected_probs = jnp.where(mask_in_topk, top_k_probs, 0.0)

        # Step 4: Reconstruct full probability distribution
        result = jnp.zeros_like(prob_row)
        result = result.at[top_k_indices].set(selected_probs)

        return result / jnp.sum(result)

    return jax.vmap(process_single_sample, in_axes=(0, 0))(probs, top_p_values)


def _sampling_from_prob(probs: jax.Array, threshold: jax.Array):
    """Sample from probability distribution using cumulative distribution function.

    This function implements robust sampling with fallback mechanism:
    1. Filter valid probabilities (> 0)
    2. Compute cumulative distribution function (CDF)
    3. Find first index where CDF > threshold
    4. If no valid sample found (threshold too large), use last valid index

    Args:
        probs: Probability distribution, shape: (vocab_size,)
        threshold: Sampling threshold (u = coin * sum(probs)), scalar

    Returns:
        sampled_id: Sampled token index, scalar
    """
    vocab_size = probs.shape[0]

    # Step 1: Filter valid probabilities (> 0)
    valid_probs = jnp.where(probs > 0, probs, 0)

    # Step 2: Compute cumulative distribution function (CDF)
    cumsum_probs = jnp.cumsum(valid_probs)

    # Step 3: Find first index where CDF > threshold
    greater_than_threshold = cumsum_probs > threshold
    selected_idx = jnp.argmax(greater_than_threshold)

    # Step 4: Fallback mechanism for edge cases
    # Case 1: threshold is too large (u very close to 1.0)
    # Case 2: sum of probabilities < threshold due to floating point errors
    # In these cases, argmax returns 0 even though no valid sample exists

    # Check if we actually found a valid sample
    found_valid_sample = greater_than_threshold[selected_idx]

    # Find last valid index (last non-zero probability)
    valid_mask = valid_probs > 0
    valid_indices = jnp.where(valid_mask, jnp.arange(vocab_size), -1)
    last_valid_id = jnp.max(valid_indices)

    # Fallback logic:
    # - If found valid sample: use selected_idx
    # - If no valid sample and last_valid_id exists: use last_valid_id
    # - If no valid indices at all: use vocab_size - 1 (extreme fallback)
    sampled_id = jnp.where(
        found_valid_sample,
        selected_idx,
        jnp.where(last_valid_id >= 0, last_valid_id, vocab_size - 1),
    )

    return sampled_id


def tree_speculative_sampling_target_only(
    predicts: jax.Array,
    accept_index: jax.Array,
    accept_token_num: jax.Array,
    candidates: jax.Array,
    retrive_index: jax.Array,
    retrive_next_token: jax.Array,
    retrive_next_sibling: jax.Array,
    uniform_samples: jax.Array,
    uniform_samples_for_final_sampling: jax.Array,
    target_probs: jax.Array,
    draft_probs: jax.Array,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
):
    """Verify the draft tree with specific sample policy.

    Args:
      predicts: draft probabilities, this array will be modified and return. Shape: (bs * draft_token_num,).
      accept_index: the index of accept token, this array will be modified and return. Shape: (bs, num_spec_step).
      accept_token_num: accept token number, this array will be modified and return. Shape: (bs,)
      candidates: candidate draft tokens # shape: (bs, draft_token_num)
      retrive_index: store the predict array index of token, shape: (bs, draft_token_num)
      retrive_next_token: store the first children in tree, shape: (bs, draft_token_num)
      retrive_next_sibling: store the next brother node in tree, shape: (bs, draft_token_num)
      uniform_samples: uniform samples, shape: (bs, draft_token_num)
      uniform_samples_for_final_sampling:  shape: (bs,)
      target_probs: the probs from target model, shape: (bs * draft_token_num, vocab_size)
      draft_probs: shape: (bs * draft_token_num, vocab_size)
      threshold_single:
      threshold_acc:
      deterministic:

    Returns:
      predicts: draft probabilities, shape: (bs*draft_token_num,).
      accept_index: the index of accept token, shape: (bs, num_spec_step).
      accept_token_num: accept token number, shape: (bs,)
    """
    num_spec_step = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]
    vocab_size = target_probs.shape[1]
    dtype = uniform_samples.dtype

    for bid, _ in enumerate(candidates):
        prob_acc = jnp.array(0, dtype=dtype)
        cur_prob_offset = bid * num_draft_tokens
        cur_index = jnp.array(0, dtype=jnp.int32)
        coin = uniform_samples[bid, cur_index]
        last_accepted_retrive_idx = retrive_index[bid, 0]
        num_accepted_tokens = 0
        accept_index = accept_index.at[bid, 0].set(last_accepted_retrive_idx)

        for _ in range(1, num_spec_step):
            cur_index = retrive_next_token[bid, cur_index]
            while cur_index != -1:
                draft_index = retrive_index[bid, cur_index]
                draft_token_id = candidates[bid, cur_index]
                target_prob_single = target_probs[cur_prob_offset, draft_token_id]
                prob_acc += target_prob_single
                if coin <= prob_acc / threshold_acc or target_prob_single >= threshold_single:
                    # accept token
                    # reset prob_acc
                    prob_acc = jnp.array(0, dtype=dtype)
                    cur_prob_offset = bid * num_draft_tokens + cur_index
                    coin = uniform_samples[bid, cur_index]
                    predicts = predicts.at[last_accepted_retrive_idx].set(draft_token_id)
                    num_accepted_tokens += 1
                    accept_index = accept_index.at[bid, num_accepted_tokens].set(draft_index)
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    # FIXME: leverage draft probs
                    draft_probs = draft_probs.at[cur_prob_offset, draft_token_id].set(
                        target_probs[cur_prob_offset, draft_token_id]
                    )
                    cur_index = retrive_next_sibling[bid, cur_index]
            if cur_index == -1:
                break
        accept_token_num = accept_token_num.at[bid].set(num_accepted_tokens)

        # we need a different coin for the final sampling
        coin = uniform_samples_for_final_sampling[bid]

        q_vec = target_probs[cur_prob_offset, :]
        if num_accepted_tokens != num_spec_step - 1:
            p_vec = draft_probs[cur_prob_offset, :]
        else:
            p_vec = jnp.zeros((vocab_size,), dtype=dtype)

        relu_q_minus_p_vec = jnp.maximum(q_vec - p_vec, jnp.array(0, dtype=dtype))
        sum_relu_q_minus_p = jnp.sum(relu_q_minus_p_vec)
        u = coin * sum_relu_q_minus_p
        sampled_id = _sampling_from_prob(relu_q_minus_p_vec, u)
        predicts = predicts.at[last_accepted_retrive_idx].set(sampled_id)

    return accept_index, accept_token_num, predicts


def tree_speculative_sampling_target_only_jit(
    predicts: jax.Array,
    accept_index: jax.Array,
    accept_token_num: jax.Array,
    candidates: jax.Array,
    retrive_index: jax.Array,
    retrive_next_token: jax.Array,
    retrive_next_sibling: jax.Array,
    uniform_samples: jax.Array,
    uniform_samples_for_final_sampling: jax.Array,
    target_probs: jax.Array,
    draft_probs: jax.Array,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
):
    """JIT-compiled non-greedy speculative rejection sampling.

    Uses jax.lax control flow (fori_loop/while_loop/cond) for XLA compilation.
    Must be called via jax.jit() inside a mesh context for TP sharding support.

    Same args and returns as tree_speculative_sampling_target_only.
    """
    bs = candidates.shape[0]
    num_spec_step = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]
    vocab_size = target_probs.shape[1]
    dtype = uniform_samples.dtype

    def batch_loop(bid, state):
        predicts_s, accept_index_s, accept_token_num_s, draft_probs_s = state

        # Initialize this batch element
        accept_index_s = accept_index_s.at[bid, 0].set(retrive_index[bid, 0])

        cur_index_init = jnp.array(0, dtype=jnp.int32)
        coin_init = uniform_samples[bid, 0]
        last_accepted_init = retrive_index[bid, 0]
        num_acc_init = jnp.array(0, dtype=jnp.int32)
        prob_acc_init = jnp.array(0, dtype=dtype)
        cur_prob_offset_init = bid * num_draft_tokens

        # --- Step loop (fori_loop over speculation steps) ---
        def step_body(step, step_state):
            (cur_idx, last_acc, n_acc, stop_fori, p_acc, coin, cp_off, dp_s, pred_s, ai_s) = (
                step_state
            )

            def verify(v_state):
                (cur_idx, last_acc, n_acc, stop_fori, p_acc, coin, cp_off, dp_s, pred_s, ai_s) = (
                    v_state
                )

                cur_idx = retrive_next_token[bid, cur_idx]

                # --- Sibling traversal (while_loop) ---
                def while_cond(w):
                    return (w[0] != -1) & (w[1] != 1)

                def while_body(w):
                    (ci, sw, la, na, pa, co, cpo, dp, pr, ai) = w
                    d_idx = retrive_index[bid, ci]
                    d_tok = candidates[bid, ci]
                    tp_single = target_probs[cpo, d_tok]
                    pa = pa + tp_single
                    do_accept = (co <= pa / threshold_acc) | (tp_single >= threshold_single)

                    def accept_fn(ops):
                        ci_, sw_, la_, na_, pa_, co_, cpo_, d_idx_, d_tok_, dp_, pr_, ai_ = ops
                        pr_ = pr_.at[la_].set(d_tok_)
                        na_ = na_ + 1
                        ai_ = ai_.at[bid, na_].set(d_idx_)
                        return (
                            ci_,
                            jnp.int32(1),
                            d_idx_,
                            na_,
                            jnp.array(0, dtype=dtype),
                            uniform_samples[bid, ci_],
                            bid * num_draft_tokens + ci_,
                            d_idx_,
                            d_tok_,
                            dp_,
                            pr_,
                            ai_,
                        )

                    def reject_fn(ops):
                        ci_, sw_, la_, na_, pa_, co_, cpo_, d_idx_, d_tok_, dp_, pr_, ai_ = ops
                        dp_ = dp_.at[cpo_, d_tok_].set(target_probs[cpo_, d_tok_])
                        ci_ = retrive_next_sibling[bid, ci_]
                        return (ci_, sw_, la_, na_, pa_, co_, cpo_, d_idx_, d_tok_, dp_, pr_, ai_)

                    (ci, sw, la, na, pa, co, cpo, _, _, dp, pr, ai) = jax.lax.cond(
                        do_accept,
                        accept_fn,
                        reject_fn,
                        (ci, sw, la, na, pa, co, cpo, d_idx, d_tok, dp, pr, ai),
                    )
                    return (ci, sw, la, na, pa, co, cpo, dp, pr, ai)

                (cur_idx, _, last_acc, n_acc, p_acc, coin, cp_off, dp_s, pred_s, ai_s) = (
                    jax.lax.while_loop(
                        while_cond,
                        while_body,
                        (
                            cur_idx,
                            jnp.int32(0),
                            last_acc,
                            n_acc,
                            p_acc,
                            coin,
                            cp_off,
                            dp_s,
                            pred_s,
                            ai_s,
                        ),
                    )
                )
                stop_fori = jax.lax.select(cur_idx == -1, 1, 0)
                return (
                    cur_idx,
                    last_acc,
                    n_acc,
                    stop_fori,
                    p_acc,
                    coin,
                    cp_off,
                    dp_s,
                    pred_s,
                    ai_s,
                )

            return jax.lax.cond(
                stop_fori != 1,
                verify,
                lambda x: x,
                (cur_idx, last_acc, n_acc, stop_fori, p_acc, coin, cp_off, dp_s, pred_s, ai_s),
            )

        (
            _,
            last_acc_final,
            n_acc_final,
            _,
            _,
            _,
            cp_off_final,
            draft_probs_s,
            predicts_s,
            accept_index_s,
        ) = jax.lax.fori_loop(
            1,
            num_spec_step,
            step_body,
            (
                cur_index_init,
                last_accepted_init,
                num_acc_init,
                jnp.int32(0),
                prob_acc_init,
                coin_init,
                cur_prob_offset_init,
                draft_probs_s,
                predicts_s,
                accept_index_s,
            ),
        )

        # --- Final sampling ---
        final_coin = uniform_samples_for_final_sampling[bid]
        q_vec = target_probs[cp_off_final, :]
        p_vec = jax.lax.cond(
            n_acc_final != num_spec_step - 1,
            lambda _: draft_probs_s[cp_off_final, :],
            lambda _: jnp.zeros((vocab_size,), dtype=dtype),
            None,
        )
        relu_qp = jnp.maximum(q_vec - p_vec, jnp.array(0, dtype=dtype))
        u = final_coin * jnp.sum(relu_qp)
        sampled_id = _sampling_from_prob(relu_qp, u)
        predicts_s = predicts_s.at[last_acc_final].set(sampled_id)
        accept_token_num_s = accept_token_num_s.at[bid].set(n_acc_final)

        return predicts_s, accept_index_s, accept_token_num_s, draft_probs_s

    predicts, accept_index, accept_token_num, _ = jax.lax.fori_loop(
        0,
        bs,
        batch_loop,
        (predicts, accept_index, accept_token_num, draft_probs),
    )

    return accept_index, accept_token_num, predicts


def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size,
    num_draft_tokens,
) -> jax.Array:
    for i, seq_len in enumerate(seq_lens):
        evict_draft_token_mask = evict_mask[i * num_draft_tokens : (i + 1) * num_draft_tokens]
        evict_num = jnp.sum(evict_draft_token_mask)
        accept_num = num_draft_tokens - evict_num
        start = (seq_len + accept_num - 1) // page_size * page_size - seq_len
        for j in range(max(start, 0), min(start + page_size, num_draft_tokens)):
            evict_mask = evict_mask.at[i * num_draft_tokens + j].set(False)

    return evict_mask


def get_target_cache_loc(
    accept_length: jnp.array,
    to_free_num_slots: jnp.array,
    out_cache_loc: jnp.array,
    num_verify_tokens: int,
) -> tuple[jnp.array, jnp.array]:
    # batch_size = accept_length.shape[0]

    # process accepted token
    copy_lens_accepted = accept_length + 1
    max_accepted_len = jnp.max(copy_lens_accepted)

    # create mask matrix
    token_indices = jnp.arange(max_accepted_len)[None, :]  # [1, max_len]
    # [batch_size, max_len]
    accepted_mask = token_indices < copy_lens_accepted[:, None]

    # select accepted position with mask matrix
    accepted_positions = jnp.where(accepted_mask, out_cache_loc[:, :max_accepted_len], -1)

    # remove padding
    tgt_cache_loc = accepted_positions.flatten()
    tgt_cache_loc = tgt_cache_loc[tgt_cache_loc != -1]

    # process released token
    max_to_free = jnp.max(to_free_num_slots)
    free_indices = jnp.arange(max_to_free)[None, :] + (
        num_verify_tokens - to_free_num_slots[:, None]
    )
    free_mask = jnp.arange(max_to_free)[None, :] < to_free_num_slots[:, None]
    free_indices = jnp.clip(free_indices, 0, num_verify_tokens - 1)

    # use advanced indexing to select released position
    free_positions = jnp.where(
        free_mask, jnp.take_along_axis(out_cache_loc, free_indices, axis=1), -1
    )

    to_free_slots = free_positions.flatten()
    to_free_slots = to_free_slots[to_free_slots != -1]

    return tgt_cache_loc, to_free_slots


def filter_finished_cache_loc_kernel(
    tgt_cache_loc: jnp.array,
    accept_length: jnp.array,
    accept_length_filter: jnp.array,
) -> jnp.array:
    batch_size = accept_length.shape[0]
    max_length = jnp.max(accept_length_filter)

    if max_length == 0:
        return jnp.array([], dtype=tgt_cache_loc.dtype)

    accept_length_cumsum = jnp.cumsum(jnp.concatenate([jnp.array([0]), accept_length[:-1]]))
    old_starts = accept_length_cumsum + jnp.arange(batch_size)
    # new_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), accept_length_filter[:-1]]))

    # batch_indices = jnp.arange(batch_size)[:, None]  # [batch_size, 1]
    token_indices = jnp.arange(max_length)[None, :]  # [1, max_length]

    source_indices = old_starts[:, None] + token_indices  # [batch_size, max_length]

    # [batch_size, max_length]
    valid_mask = token_indices < accept_length_filter[:, None]

    source_indices = jnp.clip(source_indices, 0, tgt_cache_loc.shape[0] - 1)

    gathered_data = tgt_cache_loc[source_indices]  # [batch_size, max_length]

    masked_data = jnp.where(valid_mask, gathered_data, -1)

    flattened = masked_data.flatten()
    output_data = flattened[flattened != -1]

    return output_data
