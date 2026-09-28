"""AOT execute_sharded dispatch for large-arg-count jitted functions.

Models whose weights flatten to thousands of jit arguments (GLM-5.x /
DeepSeek-V3-class MoE: ~2487 weight leaves + ~99 KV-pool leaves + metadata
~= 2600 args) pay tens of milliseconds of per-step Python dispatch in pjit's
call path: ``_infer_params``, per-argument ``sharding.is_equivalent_to``
checks, and the O(n_args) ``shard_args`` loop. On single-stream decode this
host cost dominates TPOT (measured ~34ms/step on both v6e-64 tp64 and v7x
tp16 for GLM-5.2 753B — device-independent).

``AotDispatcher`` removes the steady-state cost:

1. The first call for each dynamic-shape key lowers and compiles an AOT
   executable, then runs once through the checked ``Compiled.__call__``
   path — this validates input shardings/layouts against the executable
   exactly like normal pjit dispatch.
2. Steady state dispatches via ``xla_executable.execute_sharded`` directly:
   the stable (weight) leaves' sharded buffers are captured once per key,
   and only the small dynamic tail (forward batch, donated pools, metadata)
   goes through ``shard_args``.
3. Executables with ordered/unordered effects, host callbacks, or mutation
   stay on the checked compiled path for that key.

An executable store uses the same cache and dispatch path, loading binaries
through CompilationManager instead of compiling. Missing binaries are errors.

Donation is unaffected: XLA input-output aliasing is baked into the
executable, and ``ExecuteReplicated`` adds no Python-side donation logic.

Enabling: ``SGLANG_JAX_AOT_DISPATCH`` = ``auto`` (on when the function sees
>= ``_AUTO_MIN_ARGS`` flat args), ``1`` (always), ``0`` (default, off).
"""

from __future__ import annotations

import logging
import os

import jax
from jax._src.lib import xla_client as _xc

from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

logger = logging.getLogger(__name__)

_ENV = os.environ.get("SGLANG_JAX_AOT_DISPATCH", "0")
_AUTO_MIN_ARGS = 512


def aot_dispatch_requested() -> bool:
    """True when SGLANG_JAX_AOT_DISPATCH is set to "auto" or "1".

    Default is off: when this returns False callers should not construct an
    AotDispatcher at all, keeping the stock pjit dispatch path untouched.
    """
    return _ENV in ("auto", "1")


def aot_dispatch_enabled(num_flat_args: int) -> bool:
    if _ENV == "auto":
        return num_flat_args >= _AUTO_MIN_ARGS
    return _ENV == "1"


def decode_no_sc_gather_compiler_options_fn():
    """Temporary XLA workaround for the jax 0.11.1 SparseCore gather-offload
    decode regression on TPU v7x (#1613, jax-ml/jax#40553).

    When SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD is set on TPU, returns a
    ``compiler_options_fn`` that compiles decode-shaped executables with the
    offload pass disabled while prefill keeps the default (the offload is
    profitable for large prefill gathers, and a process-global
    LIBTPU_INIT_ARGS disable costs ~+12% on 110k prefill). Returns ``None``
    when the workaround is not requested. Only effective together with
    SGLANG_JAX_AOT_DISPATCH since it hooks the per-shape AOT compile path.
    Remove once the upstream cost-model fix ships.
    """
    if not (is_tpu_runtime() and get_bool_env_var("SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD")):
        return None
    logger.info(
        "SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD: decode executables "
        "will be compiled with SparseCore gather offload disabled."
    )

    def _compiler_options_fn(dyn_args):
        forward_batch = dyn_args[0]
        if forward_batch.forward_mode.is_decode():
            return {
                "xla_tpu_offload_gather_to_sparsecore": "false",
                "xla_tpu_offload_all_supported_gathers_to_sparsecore": "false",
            }
        return None

    return _compiler_options_fn


class AotDispatcher:
    """Dispatch ``jit_fn(*stable_call_args, *dyn_args)`` via cached AOT executables.

    Args:
      jit_fn: the ``jax.jit``-wrapped function.
      stable_call_args: positional prefix passed to ``lower``/``Compiled.__call__``
        including any static (hashable, non-array) arguments.
      stable_flat_args: the same prefix with static arguments removed — i.e.
        exactly the prefix pytrees that appear in the executable's flat input
        list. Their leaves must be the same arrays every call (weights).
      name: label for logs.
      executable_store: optional trusted offline artifacts; a miss is an error.
      allow_fast_dispatch: allow the environment-controlled execute_sharded path.
        Loading still works through checked compiled calls when disabled.

    If the caller ever replaces the stable containers (LoRA / EPLB reload),
    it must call :meth:`invalidate` (or construct a new dispatcher); the
    cheap ``id()`` guard below also catches replacement of the top-level
    containers between calls.
    """

    def __init__(
        self,
        jit_fn,
        stable_call_args: tuple,
        stable_flat_args: tuple,
        name: str,
        compiler_options_fn=None,
        executable_store=None,
        allow_fast_dispatch=True,
    ):
        self._jit_fn = jit_fn
        self._stable_call_args = stable_call_args
        self._stable_flat_args = stable_flat_args
        self._stable_ids = tuple(id(a) for a in stable_flat_args)
        self._cache = {}
        self._name = name
        self._compiler_options_fn = compiler_options_fn
        self._store = executable_store
        self._allow_fast_dispatch = allow_fast_dispatch
        # Loading is explicit even when the optional dispatch optimization is off.
        self._enabled = True if executable_store is not None else None

    def invalidate(self) -> None:
        self._cache.clear()

    def ensure_stable_args(self, stable_call_args: tuple, stable_flat_args: tuple) -> None:
        """Rebind the stable containers if the caller replaced them.

        Callers that rebind a stable container to a new object (e.g. LoRA
        weight loading re-flattens ``model_state_leaves`` into a new list in
        ``tp_worker.prepare_lora_batch``) must route every call through this
        check: the dispatcher would otherwise keep executing with the buffers
        captured at construction time. No-op (two identity checks) when the
        containers are unchanged.
        """
        if len(stable_flat_args) == len(self._stable_flat_args) and all(
            a is b for a, b in zip(stable_flat_args, self._stable_flat_args)
        ):
            return
        logger.info("[aot-dispatch:%s] stable args rebound; dropping AOT cache", self._name)
        self._stable_call_args = stable_call_args
        self._stable_flat_args = stable_flat_args
        self._stable_ids = tuple(id(a) for a in stable_flat_args)
        self._cache.clear()

    def __call__(self, *dyn_args):
        if self._enabled is False:
            return self._jit_fn(*self._stable_call_args, *dyn_args)
        if tuple(id(a) for a in self._stable_flat_args) != self._stable_ids:
            logger.info(
                "[aot-dispatch:%s] stable args replaced; invalidating executables", self._name
            )
            self._stable_ids = tuple(id(a) for a in self._stable_flat_args)
            self._cache.clear()

        dyn_leaves, dyn_tree = jax.tree_util.tree_flatten(dyn_args)
        # Batch mode and other pytree metadata can change the program even
        # when all array shapes match. Scalar types also affect compilation.
        avals = [jax.typeof(a) for a in dyn_leaves]
        key = (dyn_tree, tuple((a.shape, a.dtype, a.weak_type) for a in avals))
        entry = self._cache.get(key)
        if entry is None:
            return self._acquire_and_first_call(key, dyn_args)
        if callable(entry):
            return entry(*dyn_args)

        (
            xla_exec,
            out_handlers,
            out_tree,
            static_bufs,
            dyn_kept,
            dyn_shardings,
            dyn_layouts,
            dyn_copy,
        ) = entry
        from jax._src.interpreters import pxla

        dyn_bufs = pxla.shard_args(
            dyn_shardings, dyn_layouts, dyn_copy, [dyn_leaves[i] for i in dyn_kept]
        )
        results = xla_exec.execute_sharded(static_bufs + list(dyn_bufs))
        out_flat = results.consume_with_handlers(out_handlers)
        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    def _acquire_and_first_call(self, key, dyn_args):
        from jax._src.interpreters import pxla

        if self._enabled is None:
            n_flat = len(jax.tree_util.tree_leaves((self._stable_flat_args + dyn_args, {})))
            self._enabled = aot_dispatch_enabled(n_flat)
            if not self._enabled:
                logger.info(
                    "[aot-dispatch:%s] disabled (%d flat args, env=%s)",
                    self._name,
                    n_flat,
                    _ENV,
                )
                return self._jit_fn(*self._stable_call_args, *dyn_args)

        compile_opts = self._compiler_options_fn(dyn_args) if self._compiler_options_fn else None
        lowered = self._jit_fn.lower(*self._stable_call_args, *dyn_args)
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager

        compiled = CompilationManager.get_executable(
            lowered, compiler_options=compile_opts, store=self._store
        )
        # Saved binaries have a flat ABI containing only DCE-surviving inputs.
        # Keep original input indices for both checked calls and fast dispatch.
        if self._store is not None:
            from sgl_jax.srt.model_executor.aot_executable import _kept_inputs

            kept = _kept_inputs(lowered)
        else:
            kept = sorted(compiled._executable.unsafe_call.kept_var_idx)

        def checked_call(*args):
            values = self._stable_flat_args + args
            if self._store is not None:
                leaves = jax.tree_util.tree_leaves(values)
                values = tuple(leaves[i] for i in kept)
            return compiled(*values)

        n_flat = len(jax.tree_util.tree_leaves(self._stable_flat_args + dyn_args))
        if not self._allow_fast_dispatch or not aot_dispatch_enabled(n_flat):
            self._cache[key] = checked_call
            return checked_call(*dyn_args)

        unsafe = compiled._executable.unsafe_call
        if (
            unsafe.ordered_effects
            or unsafe.has_unordered_effects
            or unsafe.has_host_callbacks
            or unsafe.mut is not None
        ):
            logger.warning(
                "[aot-dispatch:%s] effects/mutation present; falling back to "
                "checked dispatch for this shape",
                self._name,
            )
            self._cache[key] = checked_call
            return checked_call(*dyn_args)

        # Flat layout: [stable leaves][dyn leaves]; kept_var_idx is the
        # DCE-surviving subset, in order, matching in_handler's shardings.
        n_stable = len(jax.tree_util.tree_leaves(self._stable_flat_args))
        n_static = sum(1 for i in kept if i < n_stable)
        dyn_kept = [i for i in kept if i >= n_stable]
        shardings = unsafe.in_handler.in_shardings
        layouts = unsafe.in_handler.in_layouts

        args_flat, _ = jax.tree_util.tree_flatten((self._stable_flat_args + dyn_args, {}))
        reuse = [_xc.ArrayCopySemantics.REUSE_INPUT] * n_static
        static_bufs = list(
            pxla.shard_args(
                shardings[:n_static],
                layouts[:n_static],
                reuse,
                [args_flat[i] for i in kept[:n_static]],
            )
        )
        self._cache[key] = (
            unsafe.xla_executable,
            unsafe.out_handler.handlers,
            compiled._params.out_tree,
            static_bufs,
            [i - n_stable for i in dyn_kept],
            shardings[n_static:],
            layouts[n_static:],
            [_xc.ArrayCopySemantics.REUSE_INPUT] * len(dyn_kept),
        )
        logger.info(
            "[aot-dispatch:%s] cached shape key (%d stable + %d dyn kept args)",
            self._name,
            n_static,
            len(dyn_kept),
        )
        # First call goes through the checked path: validates that every
        # input's sharding/layout matches what the executable expects.
        return checked_call(*dyn_args)
