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


def top_k_renorm_prob(probs, top_k_values):
    """Renormalizing probabilities by top-k thresholding.

    Args:
      probs: probabilities, shape: (batch_size, num_classes).
      top_k_values: the top-k threshold for re-normalizing probabilities, shape: (batch_size, 1).

    Returns:
      Renormalized probabilities, shape ``(batch_size, num_classes)``.
    """
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert (
        probs.shape[0] == top_k_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_k_values.shape[0]: {top_k_values.shape}"

    # TODO: optimize alg of top_k by avoiding sort
    def process_single_sample(prob_row, k):
        ranks = jnp.argsort(jnp.argsort(prob_row)[::-1])
        mask = ranks < k
        masked_probs = jnp.where(mask, prob_row, 0.0)
        return masked_probs / jnp.sum(masked_probs)

    return jax.vmap(process_single_sample, in_axes=(0, 0))(probs, top_k_values)


def top_p_renorm_prob(probs, top_p_values):
    """Renormalizing probabilities by top-p thresholding.

    Args:
      probs: probabilities, shape: (batch_size, num_classes).
      top_p_values: the top-p threshold for re-normalizing probabilities, shape: (batch_size, 1).

    Returns:
      Renormalized probabilities, shape ``(batch_size, num_classes)``.
    """
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert (
        probs.shape[0] == top_p_values.shape[0]
    ), f"probs.shape[0]: {probs.shape[0]} should equal to top_k_values.shape[0]: {top_p_values.shape}"

    # TODO: optimize alg of top_p by avoiding sort
    def process_single_sample(prob_row, top_p):
        sorted_indices = jnp.argsort(prob_row)[::-1]
        sorted_probs = prob_row[sorted_indices]

        cumsum_probs = jnp.cumsum(sorted_probs)
        cutoff_idx = jnp.argmax(cumsum_probs >= top_p)

        ranks = jnp.argsort(jnp.argsort(prob_row)[::-1])

        mask = ranks <= cutoff_idx

        masked_probs = jnp.where(mask, prob_row, 0.0)
        return masked_probs / jnp.sum(masked_probs)

    return jax.vmap(process_single_sample, in_axes=(0, 0))(probs, top_p_values)


def _sampling_from_prob(probs: jax.Array, threshold: jax.Array):
    valid_probs = jnp.where(probs > 0, probs, 0)
    cumsum_probs = jnp.cumsum(valid_probs)
    selected_idx = jnp.argmax(cumsum_probs > threshold)
    return selected_idx


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

        for j in range(1, num_spec_step):
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
    accepted_positions = jnp.where(accepted_mask, out_cache_loc[:, :max_accepted_len], -1)  # 填充值

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
