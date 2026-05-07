"""ModelRunner runs the forward passes of the models."""

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import mesh as mesh_lib

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
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _build_non_hybrid_memory_pools,
)
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_available_device_memory

logger = logging.getLogger(__name__)


class ModelRunner(ModelRunnerKVCacheMixin, BaseModelRunner):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        tp_size: int,
        dp_size: int,
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
        self.rngs = rngs

        self.tp_size = tp_size
        self.dp_size = dp_size
        self.attention_tp_size = self.tp_size // self.dp_size
        self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
            self.attention_tp_size
        )
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
            self._sampler_base_rng = jax.random.PRNGKey(server_args.random_seed)
            self._sampler_step = 0
            self.initialize_jit()

        # Init memory pool and attention backends
        self.init_memory_pool(
            server_args.max_running_requests,
            server_args.max_total_tokens,
            total_device_memory,
            dp_size=server_args.dp_size,
        )

        # Init routed experts capturer
        self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                mesh=self.mesh,
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
            donate_argnames=["memory_pools"],
            static_argnames=["model_state_def"],
            compiler_options=jit_compiler_options,
        )
        def jitted_run_model(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            memory_pools,
            logits_metadata,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            with LoraBatchContext.set_batch(forward_batch):
                return model(forward_batch, memory_pools.token_to_kv_pool, logits_metadata)

        # Capture base RNG key as a constant in the JIT closure.
        # fold_in(constant, dynamic_step) is computed inside JIT, avoiding
        # the eager jax.random.split that would serialize the host-device pipeline.
        base_rng_key = self._sampler_base_rng

        @partial(jax.jit, static_argnames=["sampler_state_def", "use_sort_for_toppk_minp"])
        def jitted_sampler(
            sampler_def,
            sampler_state_def,
            sampler_state_leaves,
            use_sort_for_toppk_minp,
            rng_step,
            *args,
        ):

            model_state = jax.tree_util.tree_unflatten(sampler_state_def, sampler_state_leaves)
            sampler = nnx.merge(sampler_def, model_state)
            rng_key = jax.random.fold_in(base_rng_key, rng_step)
            return sampler(
                *args, use_sort_for_toppk_minp=use_sort_for_toppk_minp, rng_override=rng_key
            )

        @partial(jax.jit, static_argnames=["mesh"])
        def jitted_compute_logprobs(mesh, logits, next_tokens):
            return compute_logprobs(mesh, logits, next_tokens)

        def run_model_wrapper(forward_batch, logits_metadata):
            memory_pools = self.memory_pools
            return jitted_run_model(
                model_def,
                model_state_def,
                self.model_state_leaves,
                forward_batch,
                memory_pools,
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

        self.model_config.validate_tensor_parallel_config(self.attention_tp_size)
        self.model_config.configure_for_tensor_parallel(self.attention_tp_size)
        self.model_config.log_kv_heads_info(self.attention_tp_size)
        self.model_config.hf_config.ep_size = self.ep_size
        self.model_config.hf_config.ep_num_redundant_experts = (
            self.server_args.ep_num_redundant_experts
        )
        self.model_config.hf_config.moe_backend = self.model_config.moe_backend.value
        # Pick MLA forward path at server start. Only `fa` selects absorbed
        # (the MLA Pallas kernel); `fa_mha` and `native` both decompress latent
        # KV via kv_b_proj and run standard attention. Read by
        # DeepseekV3DecoderLayer to construct DeepseekV3Attention; harmless on
        # non-MLA models that ignore the attribute.
        self.model_config.hf_config.use_absorbed_mla = self.server_args.attention_backend == "fa"

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
            # Block-wise quant relies on a TPU Pallas kernel; reject early.
            wbs = self.model_config.quantization_config.weight_block_size
            if wbs is not None and jax.default_backend() != "tpu":
                raise RuntimeError(
                    f"Block-wise quantization (weight_block_size={wbs}) "
                    f"requires TPU backend, but got {jax.default_backend()!r}."
                )
            is_static = self.model_config.quantization_config.is_static_checkpoint

            if not is_static:
                logger.info("Applying DYNAMIC (online) quantization...")
                from sgl_jax.srt.utils.quantization.quantization_utils import (
                    apply_linear_quantization,
                    apply_moe_quantization,
                )

                # Apply MoE quantization first
                if self.model_config.quantization_config.has_moe_quantization():
                    self.model = apply_moe_quantization(
                        self.model_config, self.model, is_static_input=False
                    )

                # Apply quantization for linear layers
                linear_rules = self.model_config.quantization_config.get_linear_rules()
                if linear_rules:
                    self.model = apply_linear_quantization(
                        self.model_config, self.model, is_static_input=False
                    )
            else:
                logger.info("Static quantization detected. Skipping online requantization.")
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

    def adjust_layer_num(self):
        """For hybrid models, compute effective layer count accounting for
        SWA layers having potentially different KV head counts."""
        if not self.is_hybrid:
            return self.model_config.num_hidden_layers

        swa_num_kv_heads = getattr(self.model_config.hf_config, "swa_num_key_value_heads", None)
        if swa_num_kv_heads is None:
            return self.model_config.num_hidden_layers

        # Compute SWA vs full layer counts from hybrid_layer_pattern
        pattern = getattr(self.model_config.hf_config, "hybrid_layer_pattern", None)
        if pattern is None:
            return self.model_config.num_hidden_layers
        swa_layers = sum(1 for p in pattern if p == 1)
        full_layers = sum(1 for p in pattern if p == 0)

        from sgl_jax.srt.utils.jax_utils import get_num_kv_heads_by_tp

        full_heads_per_device = self.model_config.get_num_kv_heads(self.attention_tp_size)
        swa_heads_per_device = get_num_kv_heads_by_tp(swa_num_kv_heads, self.attention_tp_size)
        ratio = swa_heads_per_device / full_heads_per_device
        effective = ratio * swa_layers + full_layers
        return effective

    @property
    def is_hybrid_gdn(self):
        return self.model_config.hf_config.architectures[0] in [
            "Qwen3NextForCausalLM",
            "Qwen3NextForCausalLMMTP",
        ]

    def init_attention_backend(self):
        """Init attention kernel backend."""
        self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self):
        backend = self.server_args.attention_backend
        if self.server_args.device == "cpu" and backend in ("fa", "fa_mha"):
            logger.warning(
                "FlashAttention backend is not supported on CPU; falling back to native."
            )
            backend = "native"

        if backend == "native":
            from sgl_jax.srt.layers.attention.native_backend import NativeAttention

            return NativeAttention(self.num_attn_heads, self.num_kv_heads, self.mesh)

        # Absorbed MLA is the only branch that does not use FlashAttention.
        if backend == "fa" and self.use_mla_backend:
            from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend

            cfg = self.model_config.hf_text_config
            return MLAAttentionBackend(
                num_attn_heads=self.num_attn_heads,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                page_size=self.page_size,
                mesh=self.mesh,
                attention_data_partition_axis="data",
            )

        if backend not in ("fa", "fa_mha"):
            raise ValueError(f"Unsupported attention backend: {self.server_args.attention_backend}")

        # fa_mha on an MLA model: decompress latent KV via kv_b_proj per-forward
        # and run standard FlashAttention with per-head K/V (num_kv_heads ==
        # num_attn_heads). All other (backend, model) combinations use the
        # model's native MHA/GQA dims.
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

        if backend == "fa_mha" and self.use_mla_backend:
            cfg = self.model_config.hf_text_config
            head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
            num_kv_heads = self.num_attn_heads
        else:
            head_dim = self.model_config.head_dim
            num_kv_heads = self.num_kv_heads

        return FlashAttention(
            self.num_attn_heads,
            num_kv_heads,
            head_dim,
            page_size=self.page_size,
            mesh=self.mesh,
        )

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
            output, pool_updates, _, layers_topk_ids = self.jitted_run_model(
                forward_batch, logits_metadata
            )
            cache_miss_count = count()

        # tp_size==1: sharding constraint is lost after JIT; re-place explicitly.
        # See https://github.com/sgl-project/sglang-jax/issues/233
        if self.tp_size == 1 and isinstance(pool_updates, list):
            target_sharding = self.token_to_kv_pool.kv_sharding
            pool_updates = [jax.device_put(kv, target_sharding) for kv in pool_updates]
        self.memory_pools.replace_all(pool_updates)

        # layers_topk_ids required real_bs and original_input_len which could not be stored in ForwardBatch
        return output, cache_miss_count, layers_topk_ids

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
        # Advance step counter (pure Python, zero device overhead).
        # fold_in(base_key, step) inside JIT produces a unique RNG per step.
        self._sampler_step += 1
        # Penalty application has been moved to the Sampler for better JIT performance
        return self.jitted_sampler(
            self._sampler_step,
            logits_output,
            sampling_metadata,
        )

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
        # total_tokens is in "full-layer-equivalent token-layer" units,
        # when swa kv_head_num is not same as full kv_head_num, the constraint is:
        #   swa_tokens * swa_layers * ratio + full_tokens * full_layers = total_tokens
        #   swa_tokens = full_tokens * swa_full_tokens_ratio
        total_tokens = self.max_total_num_tokens * self.adjust_layer_num()
        full_layers_num = len(full_attention_layer_ids)
        swa_layers_num = len(swa_attention_layer_ids)
        swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

        swa_num_kv_heads = getattr(self.model_config.hf_config, "swa_num_key_value_heads", None)
        if swa_num_kv_heads is not None:
            from sgl_jax.srt.utils.jax_utils import get_num_kv_heads_by_tp

            full_heads = self.model_config.get_num_kv_heads(self.attention_tp_size)
            swa_heads = get_num_kv_heads_by_tp(swa_num_kv_heads, self.attention_tp_size)
            ratio = swa_heads / full_heads
        else:
            ratio = 1.0

        denominator = swa_full_tokens_ratio * swa_layers_num * ratio + full_layers_num
        self.full_max_total_num_tokens = int(total_tokens / denominator)
        self.swa_max_total_num_tokens = int(self.full_max_total_num_tokens * swa_full_tokens_ratio)

        # Align pool sizes to page_size and dp_size for sharding compatibility
        dp_size = self.server_args.dp_size
        alignment = self.page_size * dp_size
        self.full_max_total_num_tokens -= self.full_max_total_num_tokens % alignment
        self.swa_max_total_num_tokens -= self.swa_max_total_num_tokens % alignment

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
        self.dp_size = server_args.dp_size
        self.attention_tp_size = self.tp_size // self.dp_size

        if isinstance(model_config, MockModelConfig):
            self.num_kv_heads = model_config.num_kv_heads
            self.num_attn_heads = model_config.num_heads
            self.rngs = rngs
        else:
            self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
                self.attention_tp_size
            )
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
            self.model_config.validate_tensor_parallel_config(self.attention_tp_size)

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
            head_num=self.model_config.get_total_num_kv_heads_with_replication(
                self.attention_tp_size
            ),
            head_dim=(self.model_config.head_dim + 127) // 128 * 128,
            layer_num=self.model_config.num_hidden_layers,
            mesh=mesh,
        )
        self.memory_pools = _build_non_hybrid_memory_pools(self.token_to_kv_pool)
