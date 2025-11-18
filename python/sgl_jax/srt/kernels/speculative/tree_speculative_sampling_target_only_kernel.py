import functools

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic.helpers import sync_copy
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.utils.common_utils import cdiv


def _tree_speculative_sampling_target_only_kernel(
    # Prefetch
    candidates_ref,  # shape: (bs*draft_token_num,)
    retrive_index_ref,  # shape: (bs*draft_token_num,)
    retrive_next_token_ref,  # shape: (bs*draft_token_num,)
    retrive_next_sibling_ref,  # shape: (bs*draft_token_num,)
    uniform_samples_ref,  # shape: (bs*draft_token_num,)
    uniform_samples_for_final_sampling_ref,  # shape: (bs,)
    # Input
    target_probs_ref,  # shape: (bs*draft_token_num, vocab_size)
    draft_probs_ref,  # shape: (bs*draft_token_num, vocab_size)
    zeros_ref,  # shape: (vocab_size, )
    # Output
    o_accept_index_ref,  #  the index of accept token, shape: (bs, num_spec_step).
    o_accept_token_num_ref,  # accept token number, shape: (bs,)
    o_predicts_ref,  # draft probabilities, shape: (bs*draft_token_num,).
    # Scratch
    target_probs_buffer_ref,  # on SMEM, shape: (1, 128)
    draft_probs_buffer_ref,  # on SMEM, shape: (1, 128)
    q_vec_ref,  # on VMEM, shape: (1, vocab_size)
    p_vec_ref,
    *,
    num_draft_tokens: int,
    num_spec_tokens: int,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
):
    bid = pl.program_id(0)
    # vocab_size = target_probs_ref.shape[1]
    dtype = uniform_samples_ref.dtype

    offset = bid * num_draft_tokens
    prob_acc = jnp.array(0, dtype=dtype)
    cur_prob_offset = bid * num_draft_tokens
    cur_index = 0
    coin = uniform_samples_ref[offset + cur_index]
    last_accepted_retrive_idx = retrive_index_ref[offset]
    num_accepted_tokens = 0
    o_accept_index_ref.at[bid, 0].set(last_accepted_retrive_idx)

    def init_accept_index():
        def body(i, _):
            o_accept_index_ref.at[bid, i].set(-1)
            return ()

        jax.lax.fori_loop(
            0,
            num_spec_tokens,
            body,
            None,
            unroll=num_spec_tokens,
        )

    # init accept_index to -1
    init_accept_index()

    def probs_cumsum(arr):
        csum_arr = jnp.empty(arr.shape, dtype=arr.dtype)
        sum = jnp.array(0, dtype=arr.dtype)

        def body(i, state):
            sum, csum_arr = state
            new_sum = sum + jax.lax.dynamic_slice(arr, (i,), (1,))
            csum_arr.at[i].set(new_sum)
            return sum, csum_arr

        _, csum_arr = jax.lax.fori_loop(
            0,
            arr.shape[0],
            body,
            (sum, csum_arr),
        )
        return csum_arr

    def _sampling_from_prob(probs, threshold):
        valid_probs = jnp.where(probs > 0, probs, 0)
        cumsum_probs = probs_cumsum(valid_probs)
        selected_idx = jax.lax.argmax(cumsum_probs > threshold, axis=0, index_dtype=jnp.int32)
        return selected_idx

    def body(i, state):
        (
            cur_index,
            coin,
            prob_acc,
            cur_prob_offset,
            num_accepted_tokens,
            last_accepted_retrive_idx,
            stop_fori_loop,
        ) = state

        def verify(state):
            (
                cur_index,
                coin,
                prob_acc,
                cur_prob_offset,
                num_accepted_tokens,
                last_accepted_retrive_idx,
                stop_fori_loop,
            ) = state
            cur_index = retrive_next_token_ref[offset + cur_index]

            def while_body(state):
                (
                    cur_index,
                    coin,
                    prob_acc,
                    cur_prob_offset,
                    num_accepted_tokens,
                    last_accepted_retrive_idx,
                    stop_while,
                ) = state
                draft_index = retrive_index_ref[offset + cur_index]
                draft_token_id = candidates_ref[offset + cur_index]
                sync_copy(
                    target_probs_ref.at[cur_prob_offset, pl.ds(draft_token_id, 128)],
                    target_probs_buffer_ref.at[0],
                )
                target_prob_single = target_probs_buffer_ref[0, 0]
                prob_acc += target_prob_single

                def on_true(state):
                    (
                        cur_index,
                        coin,
                        prob_acc,
                        cur_prob_offset,
                        num_accepted_tokens,
                        last_accepted_retrive_idx,
                        stop_while,
                    ) = state
                    # accept token
                    # reset prob_acc
                    prob_acc = jnp.array(0, dtype=dtype)
                    cur_prob_offset = offset + cur_index
                    coin = uniform_samples_ref[offset + cur_index]
                    o_predicts_ref.at[last_accepted_retrive_idx].set(draft_token_id)
                    num_accepted_tokens += 1
                    o_accept_index_ref.at[bid, num_accepted_tokens].set(draft_index)
                    last_accepted_retrive_idx = draft_index
                    stop_while = 1

                    return (
                        cur_index,
                        coin,
                        prob_acc,
                        cur_prob_offset,
                        num_accepted_tokens,
                        last_accepted_retrive_idx,
                        stop_while,
                    )

                def on_false(state):
                    (
                        cur_index,
                        coin,
                        prob_acc,
                        cur_prob_offset,
                        num_accepted_tokens,
                        last_accepted_retrive_idx,
                        stop_while,
                    ) = state
                    # FIXME: leverage draft probs
                    sync_copy(
                        target_probs_ref.at[cur_prob_offset, pl.ds(draft_token_id, 128)],
                        target_probs_buffer_ref.at[0],
                    )
                    sync_copy(
                        draft_probs_ref.at[cur_prob_offset, pl.ds(draft_token_id, 128)],
                        draft_probs_buffer_ref.at[0],
                    )
                    draft_probs_buffer_ref.at[0, 0].set(target_probs_buffer_ref[0, 0])
                    sync_copy(
                        draft_probs_buffer_ref.at[0],
                        draft_probs_ref.at[cur_prob_offset, pl.ds(draft_token_id, 128)],
                    )

                    cur_index = retrive_next_sibling_ref[offset + cur_index]
                    return (
                        cur_index,
                        coin,
                        prob_acc,
                        cur_prob_offset,
                        num_accepted_tokens,
                        last_accepted_retrive_idx,
                        stop_while,
                    )

                (
                    cur_index,
                    coin,
                    prob_acc,
                    cur_prob_offset,
                    num_accepted_tokens,
                    last_accepted_retrive_idx,
                    stop_while,
                ) = jax.lax.cond(
                    (coin <= prob_acc / threshold_acc) | (target_prob_single >= threshold_single),
                    on_true,
                    on_false,
                    (
                        cur_index,
                        coin,
                        prob_acc,
                        cur_prob_offset,
                        num_accepted_tokens,
                        last_accepted_retrive_idx,
                        stop_while,
                    ),
                )
                return (
                    cur_index,
                    coin,
                    prob_acc,
                    cur_prob_offset,
                    num_accepted_tokens,
                    last_accepted_retrive_idx,
                    stop_while,
                )

            def cond_fn(state):
                cur_index, _, _, _, _, _, stop_while = state
                return (cur_index != -1) & (stop_while != 1)

            (
                cur_index,
                coin,
                prob_acc,
                cur_prob_offset,
                num_accepted_tokens,
                last_accepted_retrive_idx,
                _,
            ) = jax.lax.while_loop(
                cond_fn,
                while_body,
                (
                    cur_index,
                    coin,
                    prob_acc,
                    cur_prob_offset,
                    num_accepted_tokens,
                    last_accepted_retrive_idx,
                    0,
                ),
            )

            stop_fori_loop = jax.lax.select(cur_index == -1, 1, 0)
            return (
                cur_index,
                coin,
                prob_acc,
                cur_prob_offset,
                num_accepted_tokens,
                last_accepted_retrive_idx,
                stop_fori_loop,
            )

        (
            cur_index,
            coin,
            prob_acc,
            cur_prob_offset,
            num_accepted_tokens,
            last_accepted_retrive_idx,
            stop_fori_loop,
        ) = jax.lax.cond(
            stop_fori_loop != 1,
            verify,
            lambda operands: operands,
            state,
        )
        return (
            cur_index,
            coin,
            prob_acc,
            cur_prob_offset,
            num_accepted_tokens,
            last_accepted_retrive_idx,
            stop_fori_loop,
        )

    (
        cur_index,
        coin,
        prob_acc,
        cur_prob_offset,
        num_accepted_tokens,
        last_accepted_retrive_idx,
        _,
    ) = jax.lax.fori_loop(
        0,
        num_spec_tokens,
        body,
        (
            cur_index,
            coin,
            prob_acc,
            cur_prob_offset,
            num_accepted_tokens,
            last_accepted_retrive_idx,
            0,
        ),
    )
    o_accept_token_num_ref.at[bid].set(num_accepted_tokens)

    # we need a different coin for the final sampling
    coin = uniform_samples_for_final_sampling_ref[bid]

    sync_copy(target_probs_ref.at[cur_prob_offset, :], q_vec_ref.at[0])

    def on_true(cur_prob_offset):
        sync_copy(draft_probs_ref.at[cur_prob_offset, :], p_vec_ref.at[0])

    def on_false(cur_prob_offset):
        sync_copy(zeros_ref.at[:], p_vec_ref.at[0])

    jax.lax.cond(
        num_accepted_tokens != num_spec_tokens - 1,
        on_true,
        on_false,
        cur_prob_offset,
    )

    relu_q_minus_p_vec = jnp.maximum(q_vec_ref[0] - p_vec_ref[0], 0)
    print(f"{relu_q_minus_p_vec.shape=}")
    sum_relu_q_minus_p = jnp.sum(relu_q_minus_p_vec)
    u = coin * sum_relu_q_minus_p
    sampled_id = _sampling_from_prob(relu_q_minus_p_vec, u)
    o_predicts_ref.at[last_accepted_retrive_idx].set(sampled_id)


def align_to(x, a):
    return cdiv(x, a) * a


def prepare_for_verify(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    target_probs,
    draft_probs,
):
    bs = retrive_index.shape[0]
    num_draft_tokens = retrive_index.shape[1]
    aligned_num_draft_tokens = align_to(num_draft_tokens, 128)
    candidates = jnp.pad(
        candidates.reshape(bs * num_draft_tokens),
        (0, aligned_num_draft_tokens - num_draft_tokens),
        constant_values=-1,
    )
    retrive_index = jnp.pad(
        retrive_index.reshape(bs * num_draft_tokens),
        (0, aligned_num_draft_tokens - num_draft_tokens),
        constant_values=-1,
    )
    retrive_next_token = jnp.pad(
        retrive_next_token.reshape(bs * num_draft_tokens),
        (0, aligned_num_draft_tokens - num_draft_tokens),
        constant_values=-1,
    )
    retrive_next_sibling = jnp.pad(
        retrive_next_sibling.reshape(bs * num_draft_tokens),
        (0, aligned_num_draft_tokens - num_draft_tokens),
        constant_values=-1,
    )
    uniform_samples = jnp.pad(
        uniform_samples.reshape(bs * num_draft_tokens),
        (0, aligned_num_draft_tokens - num_draft_tokens),
        constant_values=0,
    )

    vocab_size = target_probs.shape[1]
    aligned_vocab_size = align_to(vocab_size, 128)
    target_probs = jnp.pad(
        target_probs,
        (
            (0, 0),
            (0, aligned_vocab_size - vocab_size),
        ),
        constant_values=0,
    )
    draft_probs = jnp.pad(
        draft_probs,
        (
            (0, 0),
            (0, aligned_vocab_size - vocab_size),
        ),
        constant_values=0,
    )

    return (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        target_probs,
        draft_probs,
    )


def tree_speculative_sampling_target_only_pallas_call(
    predicts: jax.Array,  # shape: (bs * draft_token_num,)
    accept_index: jax.Array,  # shape: (bs, num_spec_steps)
    accept_token_num: jax.Array,  # shape: (bs,)
    candidates: jax.Array,  # shape: (bs, draft_token_num)
    retrive_index: jax.Array,  # shape: (bs, draft_token_num)
    retrive_next_token: jax.Array,  # shape: (bs, draft_token_num)
    retrive_next_sibling: jax.Array,  # shape: (bs, draft_token_num)
    uniform_samples: jax.Array,  # shape: (bs, draft_token_num)
    uniform_samples_for_final_sampling: jax.Array,  # shape: (bs,)
    target_probs: jax.Array,  # shape: (bs*draft_token_num, vocab_size)
    draft_probs: jax.Array,  # shape: (bs*draft_token_num, vocab_size)
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
):
    """Verify the tree greedy using a Pallas kernel"""
    bs = candidates.shape[0]
    draft_token_num = retrive_index.shape[1]
    num_spec_tokens = accept_index.shape[1]
    vocab_size = target_probs.shape[1]

    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        target_probs,
        draft_probs,
    ) = prepare_for_verify(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        target_probs,
        draft_probs,
    )

    scalar_prefetches = (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
    )

    in_specs = [
        # target_probs
        pl.BlockSpec(memory_space=pltpu.ANY),
        # draft_probs
        pl.BlockSpec(memory_space=pltpu.ANY),
        # zero probs
        pl.BlockSpec(memory_space=pltpu.ANY),
    ]

    out_specs = [
        # accept token index
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # accept token number
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # predicts
        pl.BlockSpec(memory_space=pltpu.SMEM),
    ]

    target_probs_scratch = pltpu.SMEM(
        (1, 128),
        target_probs.dtype,
    )
    draft_probs_scratch = pltpu.SMEM(
        (1, 128),
        target_probs.dtype,
    )
    q_vec_scratch = pltpu.VMEM(
        (1, vocab_size),
        target_probs.dtype,
    )
    p_vec_scratch = pltpu.VMEM(
        (1, vocab_size),
        target_probs.dtype,
    )
    scratch_shapes = [
        target_probs_scratch,
        draft_probs_scratch,
        q_vec_scratch,
        p_vec_scratch,
    ]

    kernel = pl.pallas_call(
        functools.partial(
            _tree_speculative_sampling_target_only_kernel,
            num_draft_tokens=draft_token_num,
            num_spec_tokens=num_spec_tokens,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=deterministic,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(bs,),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(shape=accept_index.shape, dtype=accept_index.dtype),
            jax.ShapeDtypeStruct(shape=accept_token_num.shape, dtype=accept_token_num.dtype),
            jax.ShapeDtypeStruct(shape=predicts.shape, dtype=predicts.dtype),
        ],
    )
    (
        accept_index,
        accept_token_num,
        predicts,
    ) = kernel(
        *scalar_prefetches,
        target_probs,
        draft_probs,
        jnp.zeros((vocab_size,), dtype=target_probs.dtype),
    )

    return accept_index, accept_token_num, predicts
