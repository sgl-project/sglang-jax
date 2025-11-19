import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.common_utils import cdiv


def _verify_tree_greedy_kernel(
    # Prefetch
    candidates_ref,  # candidate draft tokens  # shape: (bs*draft_token_num)
    retrive_index_ref,  # store the predict array index of token, shape: (bs*draft_token_num)
    retrive_next_token_ref,  # store the first children in tree, shape: (bs*draft_token_num)
    retrive_next_sibling_ref,  # store the next brother node in tree, shape: (bs*draft_token_num)
    target_predict_ref,  # the probs from target model, shape: (bs*draft_token_num,)
    # Output
    o_accept_index_ref,  #  the index of accept token, shape: (bs, num_spec_step).
    o_accept_token_num_ref,  # accept token number, shape: (bs,)
    o_predicts_ref,  # draft probabilities, shape: (bs*draft_token_num,).
    *,
    draft_token_num: int,
    num_spec_tokens: int,
):
    bid = pl.program_id(0)

    def init_accept_index():
        def body(i, _):
            o_accept_index_ref.at[bid, i].set(-1)

        jax.lax.fori_loop(
            0,
            num_spec_tokens,
            body,
            None,
            unroll=num_spec_tokens,
        )

    # init accept_index to -1
    init_accept_index()
    # index-0 token must be accepted
    last_accepted_retrive_idx = retrive_index_ref[bid * draft_token_num]
    o_accept_index_ref.at[bid, 0].set(last_accepted_retrive_idx)
    num_accepted_tokens = 0
    cur_index = 0

    def loop_body(i, state):
        cur_index, last_accepted_retrive_idx, num_accepted_tokens, stop_fori_loop = state

        def verify(state):
            cur_index, last_accepted_retrive_idx, num_accepted_tokens, stop_fori_loop = state
            cur_index = retrive_next_token_ref[bid * draft_token_num + cur_index]

            def while_loop_body(state):
                cur_index, stop_while, last_accepted_retrive_idx, num_accepted_tokens = state
                draft_index = retrive_index_ref[bid * draft_token_num + cur_index]
                draft_token_id = candidates_ref[bid * draft_token_num + cur_index]
                target_token_id = target_predict_ref[last_accepted_retrive_idx]

                def on_true(state):
                    (
                        cur_index,
                        stop_while,
                        last_accepted_retrive_idx,
                        target_token_id,
                        draft_index,
                        num_accepted_tokens,
                    ) = state
                    o_predicts_ref.at[last_accepted_retrive_idx].set(target_token_id)
                    num_accepted_tokens += 1
                    o_accept_index_ref.at[bid, num_accepted_tokens].set(draft_index)
                    last_accepted_retrive_idx = draft_index

                    stop_while = 1
                    return (
                        cur_index,
                        stop_while,
                        last_accepted_retrive_idx,
                        target_token_id,
                        draft_index,
                        num_accepted_tokens,
                    )

                def on_false(state):
                    (
                        cur_index,
                        stop_while,
                        last_accepted_retrive_idx,
                        target_token_id,
                        draft_index,
                        num_accepted_tokens,
                    ) = state
                    cur_index = retrive_next_sibling_ref[bid * draft_token_num + cur_index]

                    return (
                        cur_index,
                        stop_while,
                        last_accepted_retrive_idx,
                        target_token_id,
                        draft_index,
                        num_accepted_tokens,
                    )

                (
                    cur_index,
                    stop_while,
                    last_accepted_retrive_idx,
                    target_token_id,
                    draft_index,
                    num_accepted_tokens,
                ) = jax.lax.cond(
                    draft_token_id == target_token_id,
                    on_true,
                    on_false,
                    (
                        cur_index,
                        stop_while,
                        last_accepted_retrive_idx,
                        target_token_id,
                        draft_index,
                        num_accepted_tokens,
                    ),
                )
                return cur_index, stop_while, last_accepted_retrive_idx, num_accepted_tokens

            def cond_fn(state):
                cur_index, stop_while, _, _ = state
                return (cur_index != -1) & (stop_while != 1)

            cur_index, _, last_accepted_retrive_idx, num_accepted_tokens = jax.lax.while_loop(
                cond_fn,
                while_loop_body,
                (cur_index, 0, last_accepted_retrive_idx, num_accepted_tokens),
            )

            stop_fori_loop = jax.lax.select(cur_index == -1, 1, 0)
            return cur_index, last_accepted_retrive_idx, num_accepted_tokens, stop_fori_loop

        (
            cur_index,
            last_accepted_retrive_idx,
            num_accepted_tokens,
            stop_fori_loop,
        ) = jax.lax.cond(
            stop_fori_loop != 1,
            verify,
            lambda operands: operands,
            (cur_index, last_accepted_retrive_idx, num_accepted_tokens, stop_fori_loop),
        )
        return cur_index, last_accepted_retrive_idx, num_accepted_tokens, stop_fori_loop

    _, last_accepted_retrive_idx, num_accepted_tokens, _ = jax.lax.fori_loop(
        1,
        num_spec_tokens,
        loop_body,
        (cur_index, last_accepted_retrive_idx, num_accepted_tokens, 0),
    )
    o_accept_token_num_ref.at[bid].set(num_accepted_tokens)
    o_predicts_ref.at[last_accepted_retrive_idx].set(target_predict_ref[last_accepted_retrive_idx])


def align_to(x, a):
    return cdiv(x, a) * a


def prepare_for_verify(candidates, retrive_index, retrive_next_token, retrive_next_sibling):
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

    return candidates, retrive_index, retrive_next_token, retrive_next_sibling


@functools.partial(
    jax.jit,
    static_argnames=("draft_token_num", "num_spec_tokens"),
    donate_argnames=("predicts", "accept_index", "accept_token_num"),
)
def verify_tree_greedy_pallas_call(
    predicts: jax.Array,  # shape: (bs*num_draft_tokens,)
    accept_index: jax.Array,  # shape: (bs,num_spec_tokens)
    accept_token_num: jax.Array,
    candidates: jax.Array,
    retrive_index: jax.Array,
    retrive_next_token: jax.Array,
    retrive_next_sibling: jax.Array,
    target_predict: jax.Array,  # shape: (bs*num_draft_tokens,)
    draft_token_num: int,
    num_spec_tokens: int,
):
    """Verify the tree greedy using a Pallas kernel"""
    bs = candidates.shape[0]

    (candidates, retrive_index, retrive_next_token, retrive_next_sibling) = prepare_for_verify(
        candidates, retrive_index, retrive_next_token, retrive_next_sibling
    )

    scalar_prefetches = (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )

    out_specs = [
        # accept token index
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # accept token number
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # predicts
        pl.BlockSpec(memory_space=pltpu.SMEM),
    ]

    kernel = pl.pallas_call(
        functools.partial(
            _verify_tree_greedy_kernel,
            draft_token_num=draft_token_num,
            num_spec_tokens=num_spec_tokens,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            out_specs=out_specs,
            grid=(bs,),
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
    ) = kernel(*scalar_prefetches)

    return accept_index, accept_token_num, predicts


@jax.jit
def verify_tree_greedy(
    predicts: jax.Array,  # shape: (bs*num_draft_tokens,)
    accept_index: jax.Array,  # shape: (bs*num_spec_step,)
    accept_token_num: jax.Array,  # shape: (bs,)
    candidates: jax.Array,  # shape: (bs, num_draft_tokens)
    retrive_index: jax.Array,  # shape: (bs, num_draft_tokens)
    retrive_next_token: jax.Array,  # shape: (bs, num_draft_tokens)
    retrive_next_sibling: jax.Array,  # shape: (bs, num_draft_tokens)
    target_predict: jax.Array,  # shape: (bs*num_draft_tokens,)
):
    in_specs = (
        P(),  # predicts
        P(),  # accept_index
        P(),  # accept_token_num
        P(),  # candidates
        P(),  # retrive_index
        P(),  # retrive_next_token
        P(),  # retrive_next_sibling
        P(),  # target_predict
        None,
        None,
    )
    out_specs = (
        P(),  # accept_index
        P(),  # accept_token_num
        P(),  # predicts
    )

    (accept_index, accept_token_num, predicts) = jax.shard_map(
        verify_tree_greedy_pallas_call,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        retrive_index.shape[1],
        accept_index.shape[1],
    )
    return accept_index, accept_token_num, predicts
