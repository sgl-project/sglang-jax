"""ModelRunner runs the forward passes of the models."""

import logging
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import mesh as mesh_lib
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import AttentionArch, MockModelConfig, ModelConfig
from sgl_jax.srt.eplb.expert_location import (
    init_expert_location_metadata,
    set_global_server_args,
)
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.routed_experts_capturer import (
    RoutedExpertsCapturer,
    set_global_experts_capturer,
)
from sgl_jax.srt.layers.sampler import Sampler, compute_logprobs
from sgl_jax.srt.lora.context_manager import LoraBatchContext
from sgl_jax.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_available_device_memory

logger = logging.getLogger(__name__)


def _assert_no_shapedtypestruct(tree, name: str):
    """Fail fast if any input leaf is a ShapeDtypeStruct (pjit refuses them)."""
    try:
        tree_map_with_path = jax.tree_util.tree_map_with_path
    except AttributeError:
        tree_map_with_path = None

    if tree_map_with_path is not None:
        hits = []

        def _check(path, x):
            if isinstance(x, jax.ShapeDtypeStruct):
                hits.append((path, x.shape, x.dtype, getattr(x, "sharding", None)))
            return x

        tree_map_with_path(_check, tree)
        if hits:
            formatted = []
            for path, shape, dtype, sharding in hits:
                path_str = "/".join(str(k) for k in path)
                formatted.append(f"{path_str} shape={shape} dtype={dtype} sharding={sharding}")
            raise TypeError(f"Found ShapeDtypeStruct in {name}: " + "; ".join(formatted))
    else:
        def _check(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                raise TypeError(
                    f"Found ShapeDtypeStruct in {name}: shape={x.shape}, dtype={x.dtype}, "
                    f"sharding={getattr(x, 'sharding', None)}"
                )
            return x

        jax.tree_util.tree_map(_check, tree)


class ModelRunner(BaseModelRunner):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        tp_size: int,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        is_draft_worker: bool = False,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
        rngs: nnx.Rngs = None,
        max_padding: int = 1,
        model_class=None,
    ):
        # Parse args
        self.is_draft_worker = is_draft_worker
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.mesh = mesh
        # model args
        self.num_attn_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(tp_size)
        self.rngs = rngs

        self.tp_size = tp_size
        self.ep_size = server_args.ep_size
        self.server_args = server_args
        self.is_generation = model_config.is_generation
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid = False
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)

        self.forward_pass_id = 0

        # For sampling
        self.use_sort_for_toppk_minp = server_args.use_sort_for_toppk_minp

        self.max_padding = max_padding

        # Global vars
        global_server_args_dict.update(
            {k: getattr(server_args, k) for k in GLOBAL_SERVER_ARGS_KEYS}
        )

        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                load_format=server_args.load_format,
                download_dir=server_args.download_dir,
                model_class=model_class,
            ),
            mesh=self.mesh,
        )

        # Initialize precision tracer enable state
        precision_tracer.set_enable_precision_tracer(server_args.enable_precision_tracer)

        # If it is a draft model, tp_group can be different
        self.initialize()

    def initialize(self):
        server_args = self.server_args

        # Set highest matmul precision only for GPU/CUDA to improve numerical stability.
        # Do this at runtime (not import time) to avoid initializing busy backends.
        try:
            if str(getattr(server_args, "device", "")).lower() in ("gpu", "cuda"):
                from jax import config as _jax_config

                _jax_config.update("jax_default_matmul_precision", "highest")
        except Exception:
            pass

        # Load the model
        self.sampler = Sampler(nnx.Rngs(server_args.random_seed), mesh=self.mesh)
        total_device_memory = self.get_available_device_memory()
        self.init_attention_backend()
        self.load_model()

        # Check if the model is using hybrid SWA
        if (
            not self.server_args.disable_hybrid_swa_memory
            and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        ):
            self.is_hybrid = True

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        if not self.is_draft_worker:
            self.initialize_jit()

        # Init memory pool and attention backends
        self.init_memory_pool(
            server_args.max_running_requests,
            server_args.max_total_tokens,
            total_device_memory,
        )

        # Init routed experts capturer
        self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=self.server_args.enable_return_routed_experts,
                model_config=self.model_config,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_padding=self.max_padding,
                ep_size=self.server_args.ep_size,
                enable_balance_debug=self.server_args.enable_expert_balance_debug,
                balance_segment_counter=self.server_args.expert_balance_segment_counter,
                balance_output_file=self.server_args.expert_balance_output_file,
                enable_dist_recorder=self.server_args.enable_expert_distribution_recorder,
                dist_recorder_buffer_size=self.server_args.expert_distribution_recorder_buffer_size,
                dist_recorder_output_file=self.server_args.expert_distribution_recorder_output_file,
                physical_expert_counts=self.server_args.ep_num_redundant_experts
                + getattr(self.model_config.hf_config, "num_experts", 0),
            )
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        # note export for external modification
        self.model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        sampler_def, sampler_state = nnx.split(self.sampler)
        sampler_state_leaves, sampler_state_def = jax.tree_util.tree_flatten(sampler_state)

        # Catch abstract params early (e.g., missing weights creating ShapeDtypeStruct placeholders).
        _assert_no_shapedtypestruct(model_state, "model_state")

        enable_tpu_log_recorder = jax.default_backend() == "tpu" and (
            get_bool_env_var("SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER")
        )
        jit_compiler_options = (
            {"xla_tpu_enable_log_recorder": "true"} if enable_tpu_log_recorder else None
        )
        if enable_tpu_log_recorder:
            logger.info(
                "Enabling TPU log recorder for JIT compilation "
                "(compiler_options: xla_tpu_enable_log_recorder=true)."
            )

        @partial(
            jax.jit,
            donate_argnames=["token_to_kv_pool"],  # just donate KV cache
            static_argnames=["model_state_def"],
            compiler_options=jit_compiler_options,
        )
        def jitted_run_model(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            token_to_kv_pool,
            logits_metadata,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            with LoraBatchContext.set_batch(forward_batch):
                return model(forward_batch, token_to_kv_pool, logits_metadata)

        @partial(jax.jit, static_argnames=["sampler_state_def", "use_sort_for_toppk_minp"])
        def jitted_sampler(
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            use_sort_for_toppk_minp,
            *args,
        ):

            model_state = jax.tree_util.tree_unflatten(sampler_state_def, sampler_state_leaves)
            sampler = nnx.merge(sampler_def, model_state)
            return sampler(*args, use_sort_for_toppk_minp=use_sort_for_toppk_minp)

        @partial(jax.jit, static_argnames=["mesh"])
        def jitted_compute_logprobs(mesh, logits, next_tokens):
            return compute_logprobs(mesh, logits, next_tokens)

        def run_model_wrapper(forward_batch, logits_metadata):
            token_to_kv_pool = self.token_to_kv_pool
            return jitted_run_model(
                model_def,
                model_state_def,
                self.model_state_leaves,
                forward_batch,
                token_to_kv_pool,
                logits_metadata,
            )

        self.jitted_run_model = run_model_wrapper

        self.jitted_sampler = partial(
            jitted_sampler,
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            self.use_sort_for_toppk_minp,
        )

        self.jitted_compute_logprobs = partial(jitted_compute_logprobs, self.mesh)

    def get_available_device_memory(self):
        distributed = jax.process_count() != 1
        min_available_device_memory = get_available_device_memory(
            self.device, distributed=distributed, device_indexes=self.server_args.device_indexes
        )

        # Check memory for tensor parallelism
        local_device_memory = get_available_device_memory(
            self.device, device_indexes=self.server_args.device_indexes
        )
        if self.tp_size > 1 and min_available_device_memory < local_device_memory * 0.9:
            if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                logger.warning(
                    "The memory capacity is unbalanced. min_available_device_memory=%s, local_device_memory=%s, local_device_memory*0.9=%s",
                    min_available_device_memory,
                    local_device_memory,
                    local_device_memory * 0.9,
                )
            else:
                raise ValueError(
                    f"The memory capacity is unbalanced. min_available_device_memory={min_available_device_memory}, local_device_memory={local_device_memory}, local_device_memory*0.9={local_device_memory * 0.9}"
                )

        return min_available_device_memory

    def load_model(self):
        set_global_server_args(self.server_args)
        self.model_config.validate_tensor_parallel_config(self.tp_size)
        self.model_config.configure_for_tensor_parallel(self.tp_size)
        self.model_config.log_kv_heads_info(self.tp_size)
        self.model_config.hf_config.ep_size = self.ep_size
        self.model_config.hf_config.ep_num_redundant_experts = (
            self.server_args.ep_num_redundant_experts
        )
        self.model_config.hf_config.moe_backend = self.model_config.moe_backend.value

        if self.server_args.ep_dispatch_algorithm:
            with jax.set_mesh(self.mesh):
                init_expert_location_metadata(self.server_args, self.model_config)

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )
        if self.is_draft_worker:
            # if draft model and target model share same safetensor files, we should hack here to avoid create redundant layer kv cache
            self.model_config.num_hidden_layers = getattr(
                self.model_config, "num_nextn_predict_layers", self.model_config.num_hidden_layers
            )

        # Apply quantization if quantization config is set
        if self.model_config.quantization_config is not None:
            is_static = self.model_config.quantization_config.is_static_checkpoint

            from sgl_jax.srt.utils.quantization.quantization_utils import (
                adapt_fused_moe_static_block_quant_for_kernel,
                apply_linear_quantization,
                apply_moe_quantization,
            )

            # Apply MoE quantization first. Static checkpoints already prepare MoE
            # quantized structure in the loader before weight loading; re-running here
            # would clobber loaded scales with placeholders.
            if self.model_config.quantization_config.has_moe_quantization():
                if is_static:
                    logger.info(
                        "Skipping STATIC MoE quantization re-wrap in ModelRunner; loader already prepared and loaded MoE scales."
                    )
                else:
                    self.model = apply_moe_quantization(
                        self.model_config, self.model, is_static_input=is_static
                    )

            # Apply quantization for linear layers
            linear_rules = self.model_config.quantization_config.get_linear_rules()
            if linear_rules:
                if is_static:
                    logger.info("Applying STATIC fp8 wrapping for linear layers...")
                else:
                    logger.info("Applying DYNAMIC (online) quantization for linear layers...")
                self.model = apply_linear_quantization(
                    self.model_config, self.model, is_static_input=is_static
                )
            if (
                is_static
                and self.model_config.quantization_config.has_moe_quantization()
                and self.model_config.moe_backend.value == "fused"
            ):
                self.model = adapt_fused_moe_static_block_quant_for_kernel(
                    self.model, target_subc_quant_wsz=256
                )
            if is_static:
                self._log_static_quant_debug_once()
        # Parse other args
        self.sliding_window_size = self.model_config.sliding_window
        self.dtype = self.model_config.dtype
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", self.model_config.num_hidden_layers)
        self.num_effective_layers = self.end_layer - self.start_layer
        if self.server_args.speculative_algorithm == "EAGLE3" and not self.is_draft_worker:
            try:
                # get the aux layer from draft model config
                eagle_config = getattr(self.model_config.hf_config, "eagle_config", None)
                eagle_aux_hidden_state_layer_ids = eagle_config["eagle_aux_hidden_state_layer_ids"]
            except Exception as e:
                logger.warning("get the aux layer from draft model config %s", e)
                # if there is no aux layer, set to None
                eagle_aux_hidden_state_layer_ids = None
            self.model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)

        if os.environ.get("SGL_FUSED_MOE_DEBUG_LOG_PARAM_SHARDING", "0") == "1":
            self._log_fused_moe_param_sharding_debug_once()

    def _log_static_quant_debug_once(self):
        """Host-side one-time stats for debugging static FP8 scale loading."""
        logger = logging.getLogger(__name__)

        def _scalar(x):
            return jax.device_get(x).item() if hasattr(x, "shape") else x

        def _arr_stats(name, arr):
            try:
                nan_cnt = _scalar(jnp.isnan(arr).sum())
                inf_cnt = _scalar(jnp.isinf(arr).sum())
                min_v = _scalar(jnp.nanmin(arr))
                max_v = _scalar(jnp.nanmax(arr))
                logger.info(
                    "STATIC_FP8_DEBUG %s shape=%s dtype=%s sharding=%s nan=%s inf=%s min=%s max=%s",
                    name,
                    getattr(arr, "shape", None),
                    getattr(arr, "dtype", None),
                    getattr(arr, "sharding", None),
                    nan_cnt,
                    inf_cnt,
                    min_v,
                    max_v,
                )
            except Exception as e:
                logger.warning("STATIC_FP8_DEBUG failed for %s: %s", name, e)

        def _run_moe_gmm_compare(layer_idx, experts):
            """Compare gmm(rhs_scale) vs explicit dequantization on a local MoE shard."""
            try:
                from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
            except Exception as e:
                logger.warning("STATIC_FP8_GMM_COMPARE import failed: %s", e)
                return

            try:
                w0 = experts.wi_0.value
                w0_scale = experts.wi_0_scale.value
            except Exception as e:
                logger.warning("STATIC_FP8_GMM_COMPARE missing MoE tensors: %s", e)
                return

            try:
                # Match the runtime `EPMoE.__call__` scale sharding for gmm1.
                w0_scale_runtime = jax.sharding.reshard(
                    w0_scale,
                    NamedSharding(experts.moe_mesh, P("expert", None, None, "tensor")),
                )
            except Exception as e:
                logger.warning("STATIC_FP8_GMM_COMPARE scale reshard failed: %s", e)
                return

            try:
                w0_shards = list(getattr(w0, "addressable_shards", []))
                s0_shards = list(getattr(w0_scale_runtime, "addressable_shards", []))
                if not w0_shards or not s0_shards:
                    logger.warning("STATIC_FP8_GMM_COMPARE no addressable shards found")
                    return

                target_device = w0_shards[0].device

                def _shard_on_device(shards, dev):
                    for shard in shards:
                        if shard.device == dev:
                            return shard.data
                    return None

                rhs_local = _shard_on_device(w0_shards, target_device)
                rhs_scale_local = _shard_on_device(s0_shards, target_device)
                if rhs_local is None or rhs_scale_local is None:
                    logger.warning(
                        "STATIC_FP8_GMM_COMPARE failed to align local shards on device=%s",
                        target_device,
                    )
                    return

                if rhs_local.ndim != 3 or rhs_scale_local.ndim != 4:
                    logger.warning(
                        "STATIC_FP8_GMM_COMPARE unexpected local shapes rhs=%s rhs_scale=%s",
                        rhs_local.shape,
                        rhs_scale_local.shape,
                    )
                    return

                num_groups, n, k = map(int, rhs_local.shape)
                num_blocks = int(rhs_scale_local.shape[1])
                if num_blocks <= 0 or k % num_blocks != 0:
                    logger.warning(
                        "STATIC_FP8_GMM_COMPARE invalid block config rhs=%s rhs_scale=%s",
                        rhs_local.shape,
                        rhs_scale_local.shape,
                    )
                    return
                block_k = k // num_blocks

                m = min(64, max(8, num_groups * 8))
                sizes = np.full((num_groups,), m // num_groups, dtype=np.int32)
                sizes[: (m % num_groups)] += 1
                group_sizes = jnp.array(sizes, dtype=jnp.int32)

                dev = next(iter(rhs_local.devices()))
                key = jax.random.PRNGKey(1234 + int(layer_idx))
                lhs = jax.random.uniform(
                    key,
                    (m, k),
                    minval=-3.0,
                    maxval=3.0,
                    dtype=jnp.float32,
                )
                lhs = jax.device_put(lhs, dev)
                group_sizes = jax.device_put(group_sizes, dev)
                group_offset = jnp.array(0, dtype=jnp.int32)

                # Expand block scales [G, B, 1, N] -> [G, N, K] to build an explicit dequant rhs.
                scales = jnp.squeeze(rhs_scale_local, axis=2)  # [G, B, N]
                scales = jnp.transpose(scales, (0, 2, 1))  # [G, N, B]
                scales = jnp.repeat(scales, block_k, axis=2)
                scales = scales[..., :k]
                rhs_deq = rhs_local.astype(jnp.float32) * scales.astype(jnp.float32)

                tiling = (min(512, m), min(1024, k), min(1024, n))
                out_scaled = gmm(
                    lhs=lhs,
                    rhs=rhs_local,
                    group_sizes=group_sizes,
                    preferred_element_type=jnp.float32,
                    rhs_scale=rhs_scale_local,
                    tiling=tiling,
                    group_offset=group_offset,
                    interpret=False,
                )
                out_deq_gmm = gmm(
                    lhs=lhs,
                    rhs=rhs_deq,
                    group_sizes=group_sizes,
                    preferred_element_type=jnp.float32,
                    rhs_scale=None,
                    tiling=tiling,
                    group_offset=group_offset,
                    interpret=False,
                )

                # Piecewise dense reference matching gmm's grouped semantics.
                lhs_host = np.asarray(jax.device_get(lhs), dtype=np.float32)
                rhs_deq_host = np.asarray(jax.device_get(rhs_deq), dtype=np.float32)
                sizes_host = np.asarray(jax.device_get(group_sizes), dtype=np.int32)
                outputs = []
                start = 0
                for g, sz in enumerate(sizes_host.tolist()):
                    if sz > 0:
                        end = start + sz
                        outputs.append(lhs_host[start:end] @ rhs_deq_host[g].T)
                        start = end
                out_ref_host = (
                    np.concatenate(outputs, axis=0)
                    if outputs
                    else np.zeros((0, n), dtype=np.float32)
                )
                out_ref = jax.device_put(out_ref_host, dev)

                def _stats(arr):
                    return (
                        _scalar(jnp.isnan(arr).sum()),
                        _scalar(jnp.isinf(arr).sum()),
                        _scalar(jnp.nanmax(jnp.abs(arr.astype(jnp.float32)))),
                    )

                def _diff(a, b):
                    a32 = a.astype(jnp.float32)
                    b32 = b.astype(jnp.float32)
                    finite = jnp.isfinite(a32) & jnp.isfinite(b32)
                    diff = jnp.where(finite, jnp.abs(a32 - b32), 0.0)
                    denom = jnp.where(finite, jnp.maximum(jnp.abs(b32), 1e-6), 1.0)
                    rel = diff / denom
                    finite_cnt = _scalar(finite.sum())
                    return (
                        finite_cnt,
                        _scalar(diff.max()) if finite_cnt else None,
                        _scalar(rel.max()) if finite_cnt else None,
                    )

                out_scaled_stats = _stats(out_scaled)
                out_deq_stats = _stats(out_deq_gmm)
                out_ref_stats = _stats(out_ref)
                diff_scaled_deq = _diff(out_scaled, out_deq_gmm)
                diff_scaled_ref = _diff(out_scaled, out_ref)
                diff_deq_ref = _diff(out_deq_gmm, out_ref)

                logger.info(
                    "STATIC_FP8_GMM_COMPARE layer=%s tensor=wi_0 rhs_local=%s rhs_scale_local=%s "
                    "block_k=%s m=%s tiling=%s "
                    "scaled(nan=%s inf=%s absmax=%s) deq_gmm(nan=%s inf=%s absmax=%s) "
                    "ref(nan=%s inf=%s absmax=%s) "
                    "diff_scaled_deq(finite=%s max_abs=%s max_rel=%s) "
                    "diff_scaled_ref(finite=%s max_abs=%s max_rel=%s) "
                    "diff_deq_ref(finite=%s max_abs=%s max_rel=%s)",
                    layer_idx,
                    rhs_local.shape,
                    rhs_scale_local.shape,
                    block_k,
                    m,
                    tiling,
                    *out_scaled_stats,
                    *out_deq_stats,
                    *out_ref_stats,
                    *diff_scaled_deq,
                    *diff_scaled_ref,
                    *diff_deq_ref,
                )
            except Exception as e:
                logger.warning("STATIC_FP8_GMM_COMPARE failed: %s", e)

        try:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
            if not layers:
                return

            # A representative linear layer that previously hit TP/block misalignment.
            k_proj = getattr(getattr(layers[0], "self_attn", None), "k_proj", None)
            if k_proj is not None and hasattr(k_proj, "weight_scale"):
                logger.info(
                    "STATIC_FP8_DEBUG linear k_proj weight_q_shape=%s weight_scale_shape=%s "
                    "weight_q_sharding=%s weight_scale_sharding=%s block_size=%s",
                    getattr(k_proj.weight_q.value, "shape", None),
                    getattr(k_proj.weight_scale.value, "shape", None),
                    getattr(k_proj.weight_q.value, "sharding", None),
                    getattr(k_proj.weight_scale.value, "sharding", None),
                    getattr(k_proj, "weight_block_size", None),
                )
                _arr_stats("linear.k_proj.weight_scale", k_proj.weight_scale.value)

            # First MoE layer is typically layer 1 in MiMo-V2-Flash.
            for i, layer in enumerate(layers):
                experts = getattr(getattr(layer, "mlp", None), "experts", None)
                if experts is None or not hasattr(experts, "wi_0_scale"):
                    continue
                if getattr(experts, "wi_0_scale", None) is None:
                    continue
                logger.info("STATIC_FP8_DEBUG first_moe_layer=%s", i)
                _arr_stats(f"moe.layer{i}.wi_0_scale", experts.wi_0_scale.value)
                _arr_stats(f"moe.layer{i}.wi_1_scale", experts.wi_1_scale.value)
                _arr_stats(f"moe.layer{i}.wo_scale", experts.wo_scale.value)
                _run_moe_gmm_compare(i, experts)
                break
        except Exception as e:
            logger.warning("STATIC_FP8_DEBUG summary failed: %s", e)

    def _log_fused_moe_param_sharding_debug_once(self):
        """Host-side concrete sharding audit for FusedEPMoE parameters."""
        logger = logging.getLogger(__name__)

        try:
            from sgl_jax.srt.layers.moe import FusedEPMoE
        except Exception as e:
            logger.warning("FUSED_MOE_PARAM_AUDIT import failed: %s", e)
            return

        max_layers = int(os.environ.get("SGL_FUSED_MOE_DEBUG_LOG_PARAM_SHARDING_MAX_LAYERS", "2"))
        mesh_shape = dict(getattr(self.mesh, "shape", {}))
        mesh_ep_size = mesh_shape.get("data", 1) * mesh_shape.get("tensor", 1)

        audited = 0

        def _arr_meta(x):
            if x is None:
                return "None"
            try:
                shape = tuple(x.shape)
            except Exception:
                shape = None
            dtype = getattr(x, "dtype", None)
            sharding = getattr(x, "sharding", None)
            sharding_spec = getattr(sharding, "spec", sharding)
            local_shapes = []
            shard_count = 0
            try:
                shards = list(getattr(x, "addressable_shards", []))
                shard_count = len(shards)
                for shard in shards[: min(4, shard_count)]:
                    local_shapes.append(tuple(shard.data.shape))
            except Exception as e:
                local_shapes = [f"<err:{type(e).__name__}>"]
            unique_local_shapes = []
            for s in local_shapes:
                if s not in unique_local_shapes:
                    unique_local_shapes.append(s)
            return (
                f"shape={shape} dtype={dtype} sharding={sharding_spec} "
                f"addr_shards={shard_count} local_shapes={unique_local_shapes}"
            )

        def _walk(obj, path="", visited=None):
            nonlocal audited
            if visited is None:
                visited = set()
            oid = id(obj)
            if oid in visited or audited >= max_layers:
                return
            visited.add(oid)

            if isinstance(obj, FusedEPMoE):
                audited += 1
                logger.info(
                    "FUSED_MOE_PARAM_AUDIT path=%s layer=%s self.ep_size=%s mesh_shape=%s mesh_ep_size=%s "
                    "num_experts=%s hidden=%s inter=%s subc=%s",
                    path or getattr(obj, "name", type(obj).__name__),
                    getattr(obj, "layer_id", None),
                    getattr(obj, "ep_size", None),
                    mesh_shape,
                    mesh_ep_size,
                    getattr(obj, "num_experts", None),
                    getattr(obj, "hidden_size", None),
                    getattr(obj, "intermediate_dim", None),
                    getattr(obj, "subc_quant_wsz", None),
                )
                for name in ("w1", "w2", "w3", "w1_scale", "w2_scale", "w3_scale"):
                    var = getattr(obj, name, None)
                    arr = None
                    try:
                        arr = None if var is None else var.value
                    except Exception:
                        arr = var
                    logger.info("FUSED_MOE_PARAM_AUDIT %s %s", name, _arr_meta(arr))

                try:
                    w1 = obj.w1.value
                    local_shapes = [tuple(s.data.shape) for s in w1.addressable_shards]
                    if local_shapes:
                        local_e = local_shapes[0][0]
                        expected_local_e = w1.shape[0] // max(mesh_ep_size, 1)
                        logger.info(
                            "FUSED_MOE_PARAM_AUDIT local_experts_check local_e=%s expected_local_e=%s",
                            local_e,
                            expected_local_e,
                        )
                except Exception as e:
                    logger.warning("FUSED_MOE_PARAM_AUDIT local_experts_check failed: %s", e)
                return

            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    child_path = f"{path}/{attr_name}" if path else attr_name
                    if isinstance(attr_value, nnx.Module):
                        _walk(attr_value, child_path, visited)
                    elif isinstance(attr_value, list):
                        for idx, item in enumerate(attr_value):
                            if isinstance(item, nnx.Module):
                                _walk(item, f"{child_path}[{idx}]", visited)

        _walk(self.model)
        if audited == 0:
            logger.info("FUSED_MOE_PARAM_AUDIT no FusedEPMoE modules found")

    def profile_max_num_token(self, total_device_memory: int):
        """
        Profile the maximum number of tokens that can fit in memory.
        Uses tpu_info to get accurate TPU memory information.
        """
        # Get accurate memory information using TPU-specific methods
        # Use tpu_info for memory information
        available_device_memory = self.get_available_device_memory()
        available_kv_cache_bytes = available_device_memory - total_device_memory * (
            1 - self.mem_fraction_static
        )

        if available_kv_cache_bytes <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")
        
        # head_dim/v_head_dim handling
        head_dim = self.model_config.head_dim
        v_head_dim = getattr(self.model_config, "v_head_dim", head_dim)
        
        head_dim_aligned = (head_dim + 127) // 128 * 128
        v_head_dim_aligned = (v_head_dim + 127) // 128 * 128
        
        # If head dims differ, they are stored separately and each aligned to 128
        if head_dim != v_head_dim:
             per_token_dim = head_dim_aligned + v_head_dim_aligned
        else:
             # Fused case
             per_token_dim = head_dim_aligned * 2
             
        cell_size = (
            self.model_config.get_num_kv_heads(self.tp_size)
            * per_token_dim
            * self.model_config.num_hidden_layers
            * jnp.dtype(self.kv_cache_dtype).itemsize
        )

        # Calculate max tokens that can fit in available memory
        max_tokens = max(1, int(available_kv_cache_bytes // cell_size))

        logger.info(
            "TPU Memory profiling: available_device_memory=%.1fGB, available_kv_cache=%.1fGB, max_tokens=%d, cell_size=%dbytes",
            available_device_memory / (1024**3),
            available_kv_cache_bytes / (1024**3),
            max_tokens,
            cell_size,
        )

        return max_tokens

    @property
    def is_hybrid_gdn(self):
        return self.model_config.hf_config.architectures[0] in [
            "Qwen3NextForCausalLM",
            "Qwen3NextForCausalLMMTP",
        ]

    def init_memory_pool(
        self,
        max_num_reqs: int | None = None,
        max_total_tokens: int | None = None,
        total_device_memory: int | None = None,
    ):
        """Initialize memory pool for KV cache."""
        # Set KV cache data type
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = jnp.bfloat16
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}.")
        logger.info("ModelRunner kv_cache_dtype: %s", self.kv_cache_dtype)
        # Profile maximum number of tokens
        self.max_total_num_tokens = self.profile_max_num_token(total_device_memory)

        # Calculate max number of requests if not provided
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / self.model_config.context_len * 512),
                    2048,
                ),
                4096,
            )

        # Handle CI environment variable for testing
        SGLANG_CI_SMALL_KV_SIZE = os.environ.get("SGLANG_CI_SMALL_KV_SIZE")
        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
                # can be concurrently allocated, so we should give a headroom for it.
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    # draft
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    # verify
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    # buffer
                    + 100
                )
                # Target worker and draft worker shares the same indices for the
                # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        # Handle max_total_tokens override
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logger.warning(
                    "max_total_tokens=%s is larger than the profiled value %s. Use the profiled value instead.",
                    max_total_tokens,
                    self.max_total_num_tokens,
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        # Align to page size
        self.max_total_num_tokens = (
            self.max_total_num_tokens // self.server_args.page_size * self.server_args.page_size
        )

        # create token size for hybrid cache
        if self.is_hybrid:
            self.set_num_token_hybrid()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

        logger.info("ModelRunner max_total_num_tokens: %s", self.max_total_num_tokens)

        # Create request to token pool if not already created
        if self.req_to_token_pool is None:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.model_config.context_len + 4,
                dtype=np.int32,
            )

        # Create KV cache pool
        if self.is_hybrid:
            self.token_to_kv_pool = SWAKVPool(
                size=self.full_max_total_num_tokens,
                size_swa=self.swa_max_total_num_tokens,
                swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
                head_dim=self.model_config.head_dim,
                v_head_dim=getattr(self.model_config, "v_head_dim", self.model_config.head_dim),
                swa_head_dim=getattr(self.model_config, "swa_head_dim", self.model_config.head_dim),
                swa_v_head_dim=getattr(
                    self.model_config,
                    "swa_v_head_dim",
                    getattr(
                        self.model_config,
                        "v_head_dim",
                        getattr(self.model_config, "swa_head_dim", self.model_config.head_dim),
                    ),
                ),
                mesh=self.mesh,
            )
        else:
            head_dim = self.model_config.head_dim
            v_head_dim = getattr(self.model_config, "v_head_dim", head_dim)
            
            self.token_to_kv_pool = MHATokenToKVPool(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
                head_dim=head_dim,
                layer_num=self.model_config.num_hidden_layers,
                mesh=self.mesh,
                v_head_dim=v_head_dim,
            )

        # Create KV pool allocator
        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                if self.is_hybrid:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        kvcache=self.token_to_kv_pool,
                    )
                else:
                    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                        size=self.max_total_num_tokens,
                        kvcache=self.token_to_kv_pool,
                    )
            else:
                assert not self.is_hybrid
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    kvcache=self.token_to_kv_pool,
                    debug_mode=False,
                )

    def init_attention_backend(self):
        """Init attention kernel backend."""
        self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self):
        # Fallback on CPU: FlashAttention (Pallas/Triton) does not support CPU compilation and execution
        backend = self.server_args.attention_backend
        if self.server_args.device == "cpu" and backend == "fa":
            logger.warning(
                "FlashAttention backend is not supported on CPU; falling back to native."
            )
            backend = "native"
        if backend == "native":
            from sgl_jax.srt.layers.attention.native_backend import NativeAttention

            return NativeAttention(self.num_attn_heads, self.num_kv_heads, self.mesh)
        elif backend == "fa":
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )

            return FlashAttention(
                self.num_attn_heads,
                self.num_kv_heads,
                self.model_config.head_dim,
                page_size=self.page_size,
                mesh=self.mesh,
                v_head_dim=getattr(self.model_config, "v_head_dim", None),
            )
        else:
            raise ValueError(f"Invalid attention backend: {self.server_args.attention_backend}")

    def _forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        for key, value in forward_batch.__dict__.items():
            if isinstance(value, jax.Array):
                logger.debug(
                    "forward_batch %s: shape=%s, sharding=%s, dtype=%s",
                    key,
                    value.shape,
                    value.sharding,
                    value.dtype,
                )
            else:
                logger.debug("forward_batch %s: %s", key, value)

        for key, value in logits_metadata.__dict__.items():
            if isinstance(value, jax.Array):
                logger.debug(
                    "logits_metadata %s: shape=%s, sharding=%s, dtype=%s",
                    key,
                    value.shape,
                    value.sharding,
                    value.dtype,
                )
            else:
                logger.debug("logits_metadata %s: %s", key, value)

        with jtu.count_pjit_cpp_cache_miss() as count:
            output, layers_kv_fused, _, layers_topk_ids = self.jitted_run_model(
                forward_batch, logits_metadata
            )
            cache_miss_count = count()
        self._set_kv_cache_after_forward(layers_kv_fused)

        # layers_topk_ids required real_bs and original_input_len which could not be stored in ForwardBatch
        return output, cache_miss_count, layers_topk_ids

    def _set_kv_cache_after_forward(self, layers_kv_fused):
        # Note: For tp_size == 1, we need to put the layers_kv_fused on the device with the target_sharding
        # because sharding P(None, 'tensor') constraint has lost and this results in cache_miss for first prefill phase.
        # Issue: https://github.com/sgl-project/sglang-jax/issues/233
        # Q: Why does not call device_put in every layer?
        # A: Because it does not work and cache_miss still happens. According to benchmark(https://github.com/sgl-project/sglang-jax/pull/234), the performance is not influenced.

        if self.tp_size == 1:
            target_sharding = NamedSharding(
                self.token_to_kv_pool.mesh,
                P(None, self.token_to_kv_pool.kv_partition_axis, None),
            )
            layers_kv_fused = [
                jax.device_put(layer_kv_fused, target_sharding)
                for layer_kv_fused in layers_kv_fused
            ]

        self.token_to_kv_pool.replace_kv_buffer(layers_kv_fused)

        # Diagnostic: verify cache was actually updated by checking L1 norm
        # L1 norm (sum of abs) should increase as new tokens are written to the cache,
        # unlike absmax which can stay constant if new values are smaller than existing max.
        self._cache_step = getattr(self, '_cache_step', 0) + 1
        if hasattr(self.token_to_kv_pool, 'full_kv_pool'):
            pool = self.token_to_kv_pool.full_kv_pool
            if pool.is_split and pool.k_buffer:
                k0 = pool.k_buffer[0]
                v0 = pool.v_buffer[0]
                k0_l1 = float(jnp.sum(jnp.abs(k0[:100].astype(jnp.float32))))
                v0_l1 = float(jnp.sum(jnp.abs(v0[:100].astype(jnp.float32))))
                logger.info(
                    "[CACHE_VERIFY] step=%d full_layer0 k_l1_100=%.2f v_l1_100=%.2f k_id=%s",
                    self._cache_step, k0_l1, v0_l1, id(k0),
                )
            swa_pool = self.token_to_kv_pool.swa_kv_pool
            if swa_pool.is_split and swa_pool.k_buffer:
                k1 = swa_pool.k_buffer[0]
                v1 = swa_pool.v_buffer[0]
                k1_l1 = float(jnp.sum(jnp.abs(k1[:100].astype(jnp.float32))))
                v1_l1 = float(jnp.sum(jnp.abs(v1[:100].astype(jnp.float32))))
                logger.info(
                    "[CACHE_VERIFY] step=%d swa_layer0 k_l1_100=%.2f v_l1_100=%.2f k_id=%s",
                    self._cache_step, k1_l1, v1_l1, id(k1),
                )

    def forward_idle(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        raise NotImplementedError("forward_idle is not implemented")

    def forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        self.forward_pass_id += 1
        precision_tracer.start_batch_trace(forward_batch.bid)
        precision_tracer.set_current_forward_pass_id(self.forward_pass_id)
        with jax.profiler.TraceAnnotation("_forward_raw"):
            ret = self._forward_raw(forward_batch, logits_metadata)
        return ret

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> tuple[LogitsProcessorOutput, int]:
        # for compatibility, 0.6.3 need to use use_mesh. set_mesh is not have __entry__ attribute.
        # on jax >=0.7.1, we need to use set_mesh.
        try:
            ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                ctx = self.mesh
        with ctx:
            if forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_extend():
                ret = self._forward(forward_batch, logits_metadata)
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(forward_batch, logits_metadata)
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        return ret

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_metadata: SamplingMetadata,
    ) -> jax.Array:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
            positions: The positions of the tokens in the sequence.
        Returns:
            A list of next_token_ids
        """
        debug_sample_sync = os.getenv("SGL_DEBUG_SAMPLE_SYNC_CHECKPOINTS") == "1"
        disable_logits_stats = os.getenv("SGL_DEBUG_DISABLE_LOGITS_STATS") == "1"

        if debug_sample_sync:
            logits = logits_output.next_token_logits
            try:
                jax.block_until_ready(logits)
                logger.info(
                    "SAMPLE_SYNC pre-sampler logits ready shape=%s sharding=%s",
                    getattr(logits, "shape", None),
                    getattr(logits, "sharding", None),
                )
            except Exception:
                logger.exception("SAMPLE_SYNC failed before sampler (logits_output.next_token_logits)")
                raise

        # Penalty application has been moved to the Sampler for better JIT performance
        next_token_ids, logprobs, new_logits_output = self.jitted_sampler(
            logits_output,
            sampling_metadata,
        )

        if debug_sample_sync:
            try:
                jax.block_until_ready(next_token_ids)
                logger.info(
                    "SAMPLE_SYNC post-sampler next_token_ids ready shape=%s sharding=%s",
                    getattr(next_token_ids, "shape", None),
                    getattr(next_token_ids, "sharding", None),
                )
            except Exception:
                logger.exception("SAMPLE_SYNC failed on next_token_ids after sampler")
                raise
            try:
                if logprobs is not None:
                    jax.block_until_ready(logprobs)
                    logger.info(
                        "SAMPLE_SYNC post-sampler logprobs ready shape=%s sharding=%s",
                        getattr(logprobs, "shape", None),
                        getattr(logprobs, "sharding", None),
                    )
            except Exception:
                logger.exception("SAMPLE_SYNC failed on logprobs after sampler")
                raise
            try:
                if new_logits_output is not None:
                    jax.block_until_ready(new_logits_output)
                    logger.info("SAMPLE_SYNC post-sampler new_logits_output ready")
            except Exception:
                logger.exception("SAMPLE_SYNC failed on new_logits_output after sampler")
                raise

        if not disable_logits_stats and not hasattr(self, "_logged_logits_stats"):
            logits = logits_output.next_token_logits
            logits_min = float(jax.device_get(jnp.nanmin(logits)))
            logits_max = float(jax.device_get(jnp.nanmax(logits)))
            logits_nan = int(jax.device_get(jnp.isnan(logits).sum()))
            logits_inf = int(jax.device_get(jnp.isinf(logits).sum()))
            logger.info(
                "Logits stats min=%s max=%s nan=%s inf=%s",
                logits_min,
                logits_max,
                logits_nan,
                logits_inf,
            )
            self._logged_logits_stats = True

        return next_token_ids, logprobs, new_logits_output

    def compute_logprobs(self, logits, token_ids: jax.Array) -> jax.Array:
        return self.jitted_compute_logprobs(logits, token_ids)

    def set_num_token_hybrid(self):
        assert self.sliding_window_size is not None and self.sliding_window_size > 0
        full_attention_layer_ids = []
        swa_attention_layer_ids = []

        # Try different attribute paths to access model layers
        layers = None
        layer_access_attempts = [
            lambda: self.model.model.layers,
            lambda: self.model.language_model.model.layers,
            lambda: self.model.language_model.layers,
            lambda: self.model.transformer.layers,
        ]
        for get_layers in layer_access_attempts:
            try:
                layers = get_layers()
                break
            except AttributeError:
                continue

        if layers is None:
            self.is_hybrid = False
            return

        for layer in layers:
            if (
                layer.self_attn.attn.sliding_window_size is None
                or layer.self_attn.attn.sliding_window_size == -1
            ):
                full_attention_layer_ids.append(layer.layer_id)
            else:
                swa_attention_layer_ids.append(layer.layer_id)

        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

        # Algorithm:
        # Existing max_total_num_tokens is per layer and assume all layers have the same number of tokens.
        # - Find total # of tokens available across layers.
        # - Calculate full_max_total_num_tokens and swa_max_total_num_tokens based on the given swa_full_tokens_ratio.
        total_tokens = self.max_total_num_tokens * self.model_config.num_hidden_layers
        full_layers_num = len(full_attention_layer_ids)
        swa_layers_num = len(swa_attention_layer_ids)
        swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

        # Solve the equations:
        # 1. swa_max_total_num_tokens * swa_layers_num + full_max_total_num_tokens * full_layers_num == total_tokens
        # 2. full_max_total_num_tokens * swa_full_tokens_ratio == swa_max_total_num_tokens
        denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
        self.full_max_total_num_tokens = int(total_tokens / denominator)
        self.swa_max_total_num_tokens = int(self.full_max_total_num_tokens * swa_full_tokens_ratio)
        self.max_total_num_tokens = self.full_max_total_num_tokens

        logger.info(
            "Use Sliding window memory pool. full_layer_tokens=%s, swa_layer_tokens=%s",
            self.full_max_total_num_tokens,
            self.swa_max_total_num_tokens,
        )

    def init_lora_manager(self):
        """Initialize LoRA manager for LoRA adapter support."""
        from sgl_jax.srt.lora.lora_manager import LoRAManager

        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            dtype=self.dtype,
            mesh=self.mesh,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
            server_args=self.server_args,
            model_config=self.model_config,
        )


class MockModelRunner(ModelRunner):
    def __init__(
        self,
        model_config: ModelConfig | MockModelConfig,
        rngs: nnx.Rngs = None,
        mesh: mesh_lib.Mesh = None,
        server_args: ServerArgs = None,
    ):
        self.server_args = server_args
        self.tp_size = server_args.tp_size

        if isinstance(model_config, MockModelConfig):
            self.num_kv_heads = model_config.num_kv_heads
            self.num_attn_heads = model_config.num_heads
            self.rngs = rngs
        else:
            self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(self.tp_size)
            self.num_attn_heads = model_config.num_attention_heads
            self.rngs = rngs

        self.dtype = jnp.float32
        self.mem_fraction_static = 0.8
        self.model_config = model_config
        self.max_total_num_tokens = 1 << 15
        self.kv_cache_dtype = jnp.bfloat16
        self.page_size = 1
        self.mesh = mesh

        # Validate tensor parallel configuration for MockModelRunner too
        if not isinstance(model_config, MockModelConfig):
            self.model_config.validate_tensor_parallel_config(self.tp_size)

        # If it is a draft model, tp_group can be different
        max_num_reqs = min(
            max(
                int(self.max_total_num_tokens / self.model_config.context_len * 512),
                2048,
            ),
            4096,
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            dtype=np.int32,
        )

        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_total_num_kv_heads_with_replication(self.tp_size),
            head_dim=(self.model_config.head_dim + 127) // 128 * 128,
            layer_num=self.model_config.num_hidden_layers,
            mesh=mesh,
        )
