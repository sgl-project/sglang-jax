"""The model-forward JIT shared by serving and offline compilation."""

from functools import partial

import jax
from flax import nnx

from sgl_jax.srt.lora.context_manager import LoraBatchContext


def _maybe_apply_recurrent_cow(forward_batch, memory_pools):
    """Clone matched tree slots before recurrent reads, when requested."""
    src = getattr(forward_batch, "recurrent_cow_src_indices", None)
    if src is None or forward_batch.recurrent_indices is None:
        return memory_pools
    rsp = memory_pools.recurrent_state_pool
    new_recurrent, new_conv = rsp.copy_slots(src, forward_batch.recurrent_indices)
    _, aux = rsp.tree_flatten()
    new_rsp = type(rsp).tree_unflatten(aux, (new_recurrent, new_conv))
    pools = dict(memory_pools._pools)
    pools["recurrent_state_pool"] = new_rsp
    return type(memory_pools)(**pools)


def make_jitted_run_model(attn_backend, compiler_options=None):
    @partial(
        jax.jit,
        donate_argnames=["memory_pools"],
        static_argnames=["model_state_def"],
        compiler_options=compiler_options,
    )
    def jitted_run_model(
        model_def,
        model_state_def,
        model_state_leaves,
        forward_batch,
        memory_pools,
        logits_metadata,
    ):
        prepare_model_state = getattr(attn_backend, "prepare_model_state", None)
        if prepare_model_state is not None:
            model_state_leaves = prepare_model_state(model_state_leaves)
        model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
        model = nnx.merge(model_def, model_state)
        memory_pools = _maybe_apply_recurrent_cow(forward_batch, memory_pools)
        with LoraBatchContext.set_batch(forward_batch):
            return model(forward_batch, memory_pools, logits_metadata)

    return jitted_run_model
