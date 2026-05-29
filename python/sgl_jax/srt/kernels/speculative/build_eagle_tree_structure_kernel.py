from functools import partial

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic.helpers import sync_copy
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P


def _build_eagle_tree_structure_kernel(
    # Prefetch
    parents_ref,  # shape: (bs, topk * (depth-1) + 1)
    selected_index_ref,  # shape: (bs, draft_token_num - 1)
    verified_seq_len_ref,  # shape: (bs,)
    cu_full_mask_len_ref,  # shape: (bs+1,)
    tree_mask_size_ref,  # shape: (1,)
    # Input
    zeros_ref,  # used for init tree mask output, shape: (batched_tree_mask_capacity, 128)
    ones_ref,  # used for init tree mask output, shape: (batched_tree_mask_capacity, 128)
    # Output
    # TODO: how to padding tree mask
    o_tree_mask_ref,  # on HBM, shape: (batched_tree_mask_capacity, 128)
    o_positions_ref,  # on SMEM, shape: (bs*draft_token_num,)
    o_retrive_index_ref,  # on SMEM, shape: (bs, draft_token_num)
    o_retrive_next_token_ref,  # on SMEM, shape: (bs, draft_token_num)
    o_retrive_next_sibling_ref,  # on SMEM, shape: (bs, draft_token_num)
    *,
    draft_token_num: int,
    topk: int,
    tree_mask_mode: int = 0,  # FULL_MASK = 0
):
    bid = pl.program_id(0)
    retrive_output_batch_offset = bid * draft_token_num
    batched_tree_mask_capacity = o_tree_mask_ref.shape[0]
    seq_len = verified_seq_len_ref[bid]
    actual_batched_tree_mask_size = tree_mask_size_ref[0]

    def set_tree_mask(start, offset, val):
        def set_true():
            sync_copy(
                ones_ref.at[pl.ds(0, offset)],
                o_tree_mask_ref.at[pl.ds(start, offset)],
            )

        def set_false():
            sync_copy(
                zeros_ref.at[pl.ds(0, offset)],
                o_tree_mask_ref.at[pl.ds(start, offset)],
            )

        jax.lax.cond(
            val == 0,
            set_false,
            set_true,
        )

    def init_output():
        def init_tree_mask():
            # set invalid tokens to false
            @pl.when(actual_batched_tree_mask_size < batched_tree_mask_capacity)
            def _():
                set_tree_mask(
                    actual_batched_tree_mask_size,
                    batched_tree_mask_capacity - actual_batched_tree_mask_size,
                    0,
                )

            # set valid tokens to true
            set_tree_mask(0, actual_batched_tree_mask_size, 1)

        def init_positions_and_retrive():
            def body(i, _):
                o_positions_ref.at[retrive_output_batch_offset + i].set(0)
                o_retrive_index_ref.at[bid, i].set(-1)
                o_retrive_next_token_ref.at[bid, i].set(-1)
                o_retrive_next_sibling_ref.at[bid, i].set(-1)

            jax.lax.fori_loop(
                0,
                draft_token_num,
                body,
                None,
                unroll=draft_token_num,
            )

        @pl.when(bid == 0)
        def _():
            init_tree_mask()

        init_positions_and_retrive()

    # init output
    init_output()

    # # build tree
    # # Calculate seq_tree_idx for this batch (exactly like CUDA kernel)
    seq_tree_idx = draft_token_num * draft_token_num * bid

    def on_true(seq_tree_idx):
        def body(i, seq_tree_idx):
            return seq_tree_idx + verified_seq_len_ref[i] * draft_token_num

        return jax.lax.fori_loop(
            0,
            bid,
            body,
            seq_tree_idx,
        )

    seq_tree_idx = jax.lax.cond(
        tree_mask_mode == 0,
        on_true,
        lambda x: x,
        seq_tree_idx,
    )

    def build_tree_body(tid, _):
        global_token_idx = bid * draft_token_num + tid
        # Calculate token_tree_idx for tree_mask
        if tree_mask_mode == 0:  # FULL_MASK
            token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1
        else:
            token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1

        # Set tree_mask for first token
        set_tree_mask(token_tree_idx - 1, 1, 1)

        # Clear next draft_token_num - 1 positions
        def loop_body(i, _):
            set_tree_mask(token_tree_idx + i, 1, 0)

        jax.lax.fori_loop(
            0,
            draft_token_num - 1,
            loop_body,
            None,
        )

        def tid_equal_to_0():
            # Verified token (tid == 0)
            o_positions_ref.at[global_token_idx].set(seq_len)
            o_retrive_index_ref.at[bid, tid].set(global_token_idx)

            # Build retrive_next_token and retrive_next_sibling (backwards iteration)
            retrive_index_offset = bid * draft_token_num

            def while_body(state):
                i, while_break = state
                current_token_idx = retrive_index_offset + i
                o_retrive_index_ref.at[bid, i].set(current_token_idx)

                parent_tb_idx = selected_index_ref[bid, i - 1] // topk

                def parent_tb_idx_over_0():
                    parent_position = 0
                    parent_token_idx = parents_ref[bid, parent_tb_idx]

                    def cond(state):
                        parent_pos, loop_upper_bound, _, while_break = state
                        return (parent_pos < loop_upper_bound) & (while_break != 1)

                    def body(state):
                        parent_pos, loop_upper_bound, parent_position, while_break = state

                        def on_true():
                            return parent_pos + 1, 1

                        parent_position, while_break = jax.lax.cond(
                            selected_index_ref[bid, parent_pos] == parent_token_idx,
                            on_true,
                            lambda: (0, 0),
                        )
                        return parent_pos + 1, loop_upper_bound, parent_position, while_break

                    (_, _, parent_position, _) = jax.lax.while_loop(
                        cond,
                        body,
                        (0, draft_token_num - 1, parent_position, 0),
                    )

                    return parent_position

                parent_position = jax.lax.cond(
                    parent_tb_idx > 0,
                    parent_tb_idx_over_0,
                    lambda: 0,
                )

                def parent_position_lt_draft_token_num(parent_position):
                    def on_true():
                        o_retrive_next_token_ref.at[bid, parent_position].set(i)

                    def on_false():
                        origin_next_token = o_retrive_next_token_ref[bid, parent_position]
                        o_retrive_next_token_ref.at[bid, parent_position].set(i)
                        o_retrive_next_sibling_ref.at[bid, i].set(origin_next_token)

                    jax.lax.cond(
                        o_retrive_next_token_ref[bid, parent_position] == -1,
                        on_true,
                        on_false,
                    )

                jax.lax.cond(
                    parent_position >= draft_token_num,
                    lambda x: None,
                    parent_position_lt_draft_token_num,
                    parent_position,
                )

                return i - 1, while_break

            def while_cond(state):
                i, while_break = state
                return (i > 0) & (while_break != 1)

            jax.lax.while_loop(
                while_cond,
                while_body,
                (
                    draft_token_num - 1,
                    0,
                ),
            )

            o_retrive_index_ref.at[bid, 0].set(bid * draft_token_num)

        def tid_not_equal_to_0():
            # Draft token (tid > 0)
            # Calculate position by tracing back to root
            position = 0
            cur_position = tid - 1  # Convert to 0-indexed for selected_index

            def body(state):
                while_break, position, cur_position = state
                position += 1
                mask_idx = token_tree_idx + cur_position

                @pl.when(mask_idx < actual_batched_tree_mask_size)
                def _():
                    set_tree_mask(mask_idx, 1, 1)

                parent_tb_idx = selected_index_ref[bid, cur_position] // topk

                def parent_tb_idx_not_equal_to_0(cur_position):
                    token_idx = parents_ref[bid, parent_tb_idx]

                    def while_body(state):
                        i, upper_bound, _, cur_position = state

                        cur_position, while_break = jax.lax.cond(
                            selected_index_ref[bid, i] == token_idx,
                            lambda: (i, 1),  # return i and break
                            lambda: (cur_position, 0),
                        )
                        return i + 1, upper_bound, while_break, cur_position

                    def while_cond(state):
                        i, upper_bound, while_break, _ = state
                        return (i < upper_bound) & (while_break != 1)

                    _, _, _, cur_position = jax.lax.while_loop(
                        while_cond,
                        while_body,
                        (0, draft_token_num - 1, 0, cur_position),
                    )
                    return cur_position, 0

                (cur_position, while_break) = jax.lax.cond(
                    parent_tb_idx == 0,
                    lambda cur_position: (cur_position, 1),  # Reached root, break
                    parent_tb_idx_not_equal_to_0,
                    cur_position,
                )
                return while_break, position, cur_position

            def cond(state):
                while_break, _, _ = state
                return while_break != 1

            _, position, _ = jax.lax.while_loop(cond, body, (0, position, cur_position))

            o_positions_ref.at[global_token_idx].set(position + seq_len)
            o_retrive_index_ref.at[bid, tid].set(global_token_idx)

        jax.lax.cond(
            tid == 0,
            tid_equal_to_0,
            tid_not_equal_to_0,
        )

    jax.lax.fori_loop(
        0,
        draft_token_num,
        build_tree_body,
        None,
    )


@partial(
    jax.jit,
    static_argnames=[
        "draft_token_num",
        "topk",
        "max_context_len",
        "tree_mask_mode",
    ],
)
def build_eagle_tree_structure_pallas_call(
    parent_list: jax.Array,
    selected_index: jax.Array,
    verified_seq_len: jax.Array,
    seq_lens_sum: jax.Array,
    *,
    draft_token_num: int,
    topk: int,
    max_context_len: int,
    tree_mask_mode: int = 0,  # FULL_MASK = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    bs = parent_list.shape[0]

    if tree_mask_mode == 0:  # FULL_MASK
        tree_mask_size = seq_lens_sum * draft_token_num + draft_token_num * draft_token_num * bs
        tree_mask_capacity = (
            max_context_len * draft_token_num * bs + draft_token_num * draft_token_num * bs
        )
    else:
        tree_mask_size = bs * draft_token_num * draft_token_num
        tree_mask_capacity = bs * draft_token_num * draft_token_num

    cu_full_mask_len = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(verified_seq_len + draft_token_num)]
    )

    tree_mask_size = jnp.asarray(tree_mask_size, dtype=jnp.int32).reshape(1)
    scalar_prefetches = (
        parent_list,
        selected_index,
        verified_seq_len,
        cu_full_mask_len,
        tree_mask_size,
    )

    in_specs = [
        # all zero array
        pl.BlockSpec(memory_space=pltpu.ANY),
        # all one array
        pl.BlockSpec(memory_space=pltpu.ANY),
    ]

    out_specs = [
        # tree mask, shape: (draft_token_num, tree_mask_capacity, 128)
        pl.BlockSpec(memory_space=pltpu.ANY),
        # positions, shape: (bs*draft_token_num,)
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # retrive_index, shape: (bs, draft_token_num)
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # retrive_next_token, shape: (bs, draft_token_num)
        pl.BlockSpec(memory_space=pltpu.SMEM),
        # retrive_next_sibling, shape: (bs, draft_token_num)
        pl.BlockSpec(memory_space=pltpu.SMEM),
    ]

    tree_mask_dtype = jnp.int32
    # expand an extra dim of 128 at the last dim
    tree_mask_last_dim = 128

    out_shape = [
        # tree mask
        jax.ShapeDtypeStruct(shape=(tree_mask_capacity, tree_mask_last_dim), dtype=tree_mask_dtype),
        # positions
        jax.ShapeDtypeStruct(shape=(bs * draft_token_num,), dtype=jnp.int32),
        # retrive_index
        jax.ShapeDtypeStruct(shape=(bs, draft_token_num), dtype=jnp.int32),
        # retrive_next_token
        jax.ShapeDtypeStruct(shape=(bs, draft_token_num), dtype=jnp.int32),
        # retrive_next_sibling
        jax.ShapeDtypeStruct(shape=(bs, draft_token_num), dtype=jnp.int32),
    ]
    kernel = pl.pallas_call(
        partial(
            _build_eagle_tree_structure_kernel,
            draft_token_num=draft_token_num,
            topk=topk,
            tree_mask_mode=tree_mask_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(bs,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        out_shape=out_shape,
    )
    (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling) = kernel(
        *scalar_prefetches,
        jnp.zeros((tree_mask_capacity, tree_mask_last_dim), dtype=tree_mask_dtype),
        jnp.ones((tree_mask_capacity, tree_mask_last_dim), dtype=tree_mask_dtype),
    )

    return (
        tree_mask[:, 0].reshape(-1),
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


@partial(
    jax.jit,
    static_argnames=(
        "draft_token_num",
        "topk",
        "max_context_len",
        "tree_mask_mode",
    ),
)
def build_eagle_tree_structure(
    parent_list: jax.Array,
    selected_index: jax.Array,
    verified_seq_len: jax.Array,
    seq_lens_sum: jax.Array,
    draft_token_num: int,
    topk: int,
    max_context_len: int,
    tree_mask_mode: int = 0,  # FULL_MASK = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Build eagle tree using a Pallas kernel.
    Args:
        parent_list: Parent indices array [bs, topk * (depth-1) + 1]
        selected_index: Selected token indices [bs, draft_token_num - 1]
        verified_seq_len: Sequence lengths [bs]
        draft_token_num: Number of draft tokens (num_verify_tokens)
        topk: Top-k value
        seq_lens_sum: Sum of sequence lengths
        tree_mask_mode: Tree mask mode (0=FULL_MASK)
        mesh: jax mesh used for distributed computation
        max_context_len: The max context length per request.

    Returns:
        tuple of (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling)
    """
    in_specs = (
        P(),  # parent_list
        P(),  # selected_index
        P(),  # verified_seq_len
        P(),  # seq_lens_sum
    )
    out_specs = (
        P(),  # tree_mask
        P(),  # positions
        P(),  # retrive_index
        P(),  # retrive_next_token
        P(),  # retrive_next_sibling
    )
    _kernel = partial(
        build_eagle_tree_structure_pallas_call,
        draft_token_num=draft_token_num,
        topk=topk,
        max_context_len=max_context_len,
        tree_mask_mode=tree_mask_mode,
    )
    (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    ) = jax.shard_map(
        _kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        parent_list,
        selected_index,
        verified_seq_len,
        seq_lens_sum,
    )
    return tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling
