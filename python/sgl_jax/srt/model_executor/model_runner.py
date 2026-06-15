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
from sgl_jax.srt.model_executor.vision_bucketing import bucket_pad_images
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_available_device_memory

logger = logging.getLogger(__name__)

# JAX's pjit C++ cache-miss counter lives in jax._src.test_util -- a PRIVATE/test-only module that
# can be renamed or removed across JAX versions (review M-8). It was previously imported INSIDE the
# per-call forward (_forward) and per-request mm encode (encode_mm_reqs) loops, so a JAX upgrade
# that moved it would break both hot paths at runtime. Import it once here, guarded, and expose a
# small context manager that degrades to a no-op probe (count 0) when the private API is absent --
# the recompile probes are observability only, never functional, so losing them must not crash.
import contextlib  # noqa: E402

try:
    from jax._src.test_util import (
        count_pjit_cpp_cache_miss as _count_pjit_cpp_cache_miss,
    )
except Exception:  # pragma: no cover - depends on JAX internals
    _count_pjit_cpp_cache_miss = None


@contextlib.contextmanager
def _count_jit_compiles():
    """Yield a callable returning #pjit C++ cache misses inside the block (0 if JAX's private
    counter API is unavailable). Centralizes the one fragile jax._src.test_util dependency."""
    if _count_pjit_cpp_cache_miss is None:
        yield lambda: 0
        return
    with _count_pjit_cpp_cache_miss() as count:
        yield count


# V-2 prompt-length bucket edge (review M-8): when vision bucketing is on, the encode jit is keyed
# by BOTH the vision grid AND the input_ids length, so the prompt is padded up to a multiple of this
# edge to bound seq-dimension recompiles. Named here rather than inlined as a bare 256 in
# encode_mm_reqs. (Follow-up: make this a server_args field so it is same-source with
# --vision-bucket-size instead of a separate constant -- recorded as backlog.)
_VISION_SEQ_BUCKET = 256


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

        self._sampler_base_rng = jax.random.PRNGKey(server_args.random_seed)
        self._sampler_step = 0
        if not self.is_draft_worker:
            self.initialize_jit()

        # G1 AOT-auto (design §5.7 option 3): when a vision max-patches bound is set, AOT-measure
        # the encode jit's peak scratch (temp_size) for a tight HBM reserve. Best-effort -> 0 on
        # failure, and _vision_activation_reserve_bytes falls back to the conservative closed form.
        # Must run before init_memory_pool (it sizes the KV pool minus this reserve).
        self._aot_vision_reserve = 0
        if (
            not self.is_draft_worker
            and getattr(self, "jitted_embed_mm", None) is not None
            and (getattr(server_args, "vision_max_patches", 0) or 0) > 0
        ):
            self._aot_vision_reserve = self._aot_vision_reserve_bytes()

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
                return model(forward_batch, memory_pools, logits_metadata)

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
            return jitted_run_model(
                model_def,
                model_state_def,
                self.model_state_leaves,
                forward_batch,
                self.memory_pools,
                logits_metadata,
            )

        self.jitted_run_model = run_model_wrapper

        # C-1 (design §5.2): standalone full-sequence multimodal encode+merge, invoked once per
        # req on the host BEFORE chunked prefill (model_runner.encode_mm_reqs). Its [seq, hidden]
        # result is held on req.multimodal_embedding and sliced per chunk by
        # ScheduleBatch._merge_multimodal -> no per-chunk re-encode (B8) and no chunk-boundary
        # merge misalignment (B1/B2). grid_thw fixes the ViT shapes -> static (recompiles per
        # distinct geometry; patch bucketing is the deferred V-2 optimization). Only built for
        # models that expose embed_mm (the in-model VLMs); None otherwise.
        @partial(
            jax.jit,
            static_argnames=[
                "model_state_def",
                "mm_grid_thw",
                "mm_video_grid_thw",
                "mm_audio_feature_lengths",
            ],
        )
        def jitted_embed_mm(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            mm_pixel_values,
            mm_grid_thw,
            mm_pixel_values_videos,
            mm_video_grid_thw,
            mm_audio_features,
            mm_audio_feature_lengths,
            mm_real_llm_dims=None,
            mm_real_video_llm_dims=None,
            mm_audio_codes=None,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.embed_mm(
                input_ids,
                mm_pixel_values,
                mm_grid_thw,
                mm_pixel_values_videos,
                mm_video_grid_thw,
                mm_audio_features,
                mm_audio_feature_lengths,
                mm_real_llm_dims=mm_real_llm_dims,
                mm_real_video_llm_dims=mm_real_video_llm_dims,
                mm_audio_codes=mm_audio_codes,
            )

        if hasattr(self.model, "embed_mm"):

            def embed_mm_wrapper(
                input_ids,
                mm_pixel_values,
                mm_grid_thw,
                mm_pixel_values_videos,
                mm_video_grid_thw,
                mm_audio_features,
                mm_audio_feature_lengths,
                mm_real_llm_dims=None,
                mm_real_video_llm_dims=None,
                mm_audio_codes=None,
            ):
                return jitted_embed_mm(
                    model_def,
                    model_state_def,
                    self.model_state_leaves,
                    input_ids,
                    mm_pixel_values,
                    mm_grid_thw,
                    mm_pixel_values_videos,
                    mm_video_grid_thw,
                    mm_audio_features,
                    mm_audio_feature_lengths,
                    mm_real_llm_dims=mm_real_llm_dims,
                    mm_real_video_llm_dims=mm_real_video_llm_dims,
                    mm_audio_codes=mm_audio_codes,
                )

            self.jitted_embed_mm = embed_mm_wrapper
            # G1 AOT (design §5.7 option 3): handle to AOT-lower the encode jit at the max vision
            # shape and read its XLA temp_size for a tight HBM reserve (best-effort).
            self._embed_mm_aot = (jitted_embed_mm, model_def, model_state_def)

            # L-k (design §7): batched vision encode. encode_mm runs ONE tower pass over all images
            # of all reqs in the batch (patches concatenated; the ViT segments per-image so this is
            # equivalent to one req with more images), then merge_mm merges per req. Built only for
            # models exposing the split (no-deepstack VLMs: MiMo, Qwen2.5-VL). encode_mm_reqs uses
            # this batched path when V-2 bucketing is off; else falls back to per-req embed_mm.
            if hasattr(self.model, "encode_mm") and hasattr(self.model, "merge_mm"):

                @partial(
                    jax.jit,
                    static_argnames=[
                        "model_state_def",
                        "mm_grid_thw",
                        "mm_video_grid_thw",
                        "mm_audio_feature_lengths",
                    ],
                )
                def jitted_encode_mm(
                    model_def,
                    model_state_def,
                    model_state_leaves,
                    mm_pixel_values=None,
                    mm_grid_thw=None,
                    mm_pixel_values_videos=None,
                    mm_video_grid_thw=None,
                    mm_audio_features=None,
                    mm_audio_feature_lengths=None,
                    mm_audio_codes=None,
                ):
                    model = nnx.merge(
                        model_def,
                        jax.tree_util.tree_unflatten(model_state_def, model_state_leaves),
                    )
                    return model.encode_mm(
                        mm_pixel_values=mm_pixel_values,
                        mm_grid_thw=mm_grid_thw,
                        mm_pixel_values_videos=mm_pixel_values_videos,
                        mm_video_grid_thw=mm_video_grid_thw,
                        mm_audio_features=mm_audio_features,
                        mm_audio_feature_lengths=mm_audio_feature_lengths,
                        mm_audio_codes=mm_audio_codes,
                    )

                @partial(jax.jit, static_argnames=["model_state_def"])
                def jitted_merge_mm(
                    model_def,
                    model_state_def,
                    model_state_leaves,
                    input_ids,
                    image=None,
                    video=None,
                    audio=None,
                ):
                    model = nnx.merge(
                        model_def,
                        jax.tree_util.tree_unflatten(model_state_def, model_state_leaves),
                    )
                    return model.merge_mm(input_ids, image=image, video=video, audio=audio)

                self.jitted_encode_mm = lambda **kw: jitted_encode_mm(
                    model_def, model_state_def, self.model_state_leaves, **kw
                )
                self.jitted_merge_mm = lambda input_ids, **kw: jitted_merge_mm(
                    model_def, model_state_def, self.model_state_leaves, input_ids, **kw
                )
            else:
                self.jitted_encode_mm = None
                self.jitted_merge_mm = None
        else:
            self.jitted_embed_mm = None
            self._embed_mm_aot = None
            self.jitted_encode_mm = None
            self.jitted_merge_mm = None

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
        self.model_config.hf_config.use_jax_allreduce_metadata = (
            not self.server_args.disable_jax_allreduce_metadata
        )
        # Pick MLA forward path at server start. Only `fa` selects absorbed
        # (the MLA Pallas kernel); `fa_mha` and `native` both decompress latent
        # KV via kv_b_proj and run standard attention. Read by
        # DeepseekV3DecoderLayer to construct DeepseekV3Attention; harmless on
        # non-MLA models that ignore the attribute.
        self.model_config.hf_config.use_absorbed_mla = self.server_args.attention_backend == "fa"
        self.model_config.hf_config.enable_sequence_parallel = (
            self.server_args.enable_sequence_parallel
        )

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

            full_attn_backend = NativeAttention(self.num_attn_heads, self.num_kv_heads, self.mesh)

        elif backend == "fa" and self.use_mla_backend:
            from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend

            cfg = self.model_config.hf_text_config
            full_attn_backend = MLAAttentionBackend(
                num_attn_heads=self.num_attn_heads,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                page_size=self.page_size,
                mesh=self.mesh,
                attention_data_partition_axis="data",
            )

        elif backend in ("fa", "fa_mha"):
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )

            if backend == "fa_mha" and self.use_mla_backend:
                cfg = self.model_config.hf_text_config
                head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
                num_kv_heads = self.num_attn_heads
            else:
                head_dim = self.model_config.head_dim
                num_kv_heads = self.num_kv_heads

            full_attn_backend = FlashAttention(
                self.num_attn_heads,
                num_kv_heads,
                head_dim,
                page_size=self.page_size,
                mesh=self.mesh,
            )

        else:
            raise ValueError(f"Unsupported attention backend: {self.server_args.attention_backend}")

        # Always go through the wrapper — it's a no-op when no hybrid config is set.
        from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
            attn_backend_wrapper,
        )

        return attn_backend_wrapper(self, full_attn_backend)

    def _forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ):
        cache_miss_count = 0
        with _count_jit_compiles() as count:
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

    def encode_mm_reqs(self, reqs):
        """C-1 (design §5.2): for each req carrying raw multimodal input and NO precomputed
        embedding, run the FULL-sequence encode+merge once and attach the ``[seq, hidden]``
        fused embedding (host ``np.ndarray``) to ``req.multimodal_embedding`` (+ sparse deepstack
        for Omni). The scheduler's ``_merge_multimodal`` then slices it per chunk into
        ``input_embedding``; the model ``__call__`` does NO in-forward encode (it just reads the
        sliced embedding) -> no per-chunk re-encode (B8) and no chunk-boundary merge misalignment
        (B1/B2). No-op for non-mm models (``jitted_embed_mm is None``) and for reqs without raw
        image/video/audio."""
        if self.jitted_embed_mm is None or not reqs:
            return
        # mm_assembly now lives in the neutral mm_core layer (M6-S1), which srt may import directly
        # -- no more importlib workaround to dodge the (now-removed) srt->multimodal reverse import.
        from sgl_jax.srt.mm_core.mm_assembly import assemble_mm_inputs as assemble

        repl = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        def _put(x, bf16=False):
            if x is None:
                return None
            y = jax.device_put(np.asarray(x), repl)
            return y.astype(jnp.bfloat16) if bf16 else y

        def _thw(rows):
            return tuple(tuple(int(v) for v in row) for row in rows) if rows else None

        # V-2 bucketing: pad each image's LLM-grid to a multiple of the bucket edge so the encode
        # jit sees a canonical (bounded) set of grid geometries. Returns (padded_px, padded_grids,
        # real_llm_dims[num,2]); the ViT masks the bucket padding (traced real dims -> no compile
        # on the real size) and the model compacts it out before merge. Only for Qwen2.5-VL (its
        # ViT exposes encode_bucketed); 0/absent -> off (validated path untouched).
        bucket_s = int(getattr(self.server_args, "vision_bucket_size", 0) or 0)
        merge_m = int(getattr(self.model, "spatial_merge_size", 0) or 0)
        bucketing_on = (
            bucket_s > 0
            and merge_m > 0
            and hasattr(getattr(self.model, "visual", None), "encode_bucketed")
        )

        try:
            ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                ctx = self.mesh

        enc_reqs = [
            r
            for r in reqs
            if getattr(r, "mm_inputs", None) and getattr(r, "multimodal_embedding", None) is None
        ]
        if not enc_reqs:
            return

        with ctx:
            # L-k (design §7): batched vision encode -- ONE tower pass over all reqs' images (patches
            # concatenated; the ViT segments per-image via cu_seqlens, so cross-req concat == one req
            # with more images), then per-req merge. Used when the model exposes encode_mm/merge_mm
            # AND V-2 bucketing is off (bucketing's per-image valid masks key recompiles on the
            # canonical grid and don't compose with a concatenated cross-req encode -> per-req
            # fallback). MiMo / Qwen2.5-VL only; Qwen3-Omni (deepstack) lacks encode_mm -> per-req.
            if self.jitted_encode_mm is not None and not bucketing_on:
                self._encode_mm_batched(enc_reqs, assemble, _put, _thw)
                return
            for r in enc_reqs:
                a = assemble(r.mm_inputs)
                img_px, vid_px = a.get("pixel_values_images"), a.get("pixel_values_videos")
                aud_feats = a.get("audio_features")
                aud_codes = a.get("audio_codes")  # MiMo-V2.5 RVQ discrete codes (int)
                if img_px is None and vid_px is None and aud_feats is None and aud_codes is None:
                    continue
                # Audio: continuous-mel features (traced) + per-audio length (static; the tower
                # chunks by it). Derive lengths from the attention mask (per-audio mel length).
                aud_len = None
                if aud_feats is not None:
                    mask = a.get("audio_feature_attention_mask")
                    if mask is not None:
                        m = np.asarray(mask)
                        aud_len = (
                            (int(m.sum()),)
                            if m.ndim <= 1
                            else tuple(int(x) for x in m.sum(axis=-1))
                        )
                    else:
                        aud_len = (int(np.asarray(aud_feats).shape[-1]),)
                # V-2 bucketing: the encode jit is keyed by BOTH the vision grid AND the
                # input_ids length (text-embed + merge shapes). Grid bucketing alone leaves the
                # seq dimension unbounded (two images of different real size -> different #image
                # placeholder tokens -> different prompt length -> recompile). So also pad the
                # prompt up to a seq bucket; the extra tail rows are sliced off the fused output
                # below (consumer sees the real length). Together this bounds the encode recompiles
                # to (#grid buckets x #seq buckets). Off (default): raw length, validated path.
                ids_np = np.asarray(r.origin_input_ids, dtype=np.int32)
                real_len = int(ids_np.shape[0])
                if bucketing_on:
                    seq_bucket = _VISION_SEQ_BUCKET
                    padded_len = ((real_len + seq_bucket - 1) // seq_bucket) * seq_bucket
                    if padded_len > real_len:
                        ids_np = np.pad(ids_np, (0, padded_len - real_len))
                input_ids = _put(ids_np)
                # V-2 bucketing (images only): pad to a canonical grid + carry traced real dims.
                img_grid = a.get("image_grid_thw")
                img_px_dev = _put(img_px, bf16=True)
                real_llm_dims = None
                if bucketing_on and img_px is not None and img_grid is not None:
                    img_px_b, img_grid, real_dims_np = bucket_pad_images(
                        img_px, img_grid, merge_m, bucket_s
                    )
                    img_px_dev = _put(img_px_b, bf16=True)
                    real_llm_dims = _put(real_dims_np)
                # V-2 probe (design §5.3): count vision-encode jit (re)compiles. A miss = a
                # distinct grid geometry that triggered an XLA compile. First-seen resolutions
                # miss once (expected); a sustained stream of misses = unbounded vision shapes
                # (the recompile storm bucketing bounds -- with --vision-bucket-size the padded
                # grids collapse to the bucket multiples, so the probe goes quiet after warmup).
                # Surfacing it makes the V-3 "compile count <= bucket count" gate observable.
                # _count_jit_compiles centralizes the (private, version-fragile) JAX counter API
                # and degrades to a no-op probe when it is unavailable (review M-8).
                with _count_jit_compiles() as _vit_compiles:
                    fused, deepstack, pos_mask = self.jitted_embed_mm(
                        input_ids,
                        img_px_dev,
                        _thw(img_grid),
                        _put(vid_px, bf16=True),
                        _thw(a.get("video_grid_thw")),
                        _put(aud_feats, bf16=True),
                        aud_len,
                        mm_real_llm_dims=real_llm_dims,
                        mm_audio_codes=_put(aud_codes),  # int RVQ codes (MiMo); no bf16 cast
                    )
                if _vit_compiles() > 0:
                    logger.info(
                        "V-2 probe: vision encode jit compiled for new geometry "
                        "(image_grid=%s, video_grid=%s, audio_len=%s, bucketing=%s)",
                        _thw(img_grid),
                        a.get("video_grid_thw"),
                        aud_len,
                        bucketing_on,
                    )
                # Slice off any seq-bucket padding tail so the held embedding matches the real
                # prompt length (the scheduler slices it per chunk over [0, real_len)).
                r.multimodal_embedding = np.asarray(jax.device_get(fused))[:real_len]
                # Deepstack (Qwen3-Omni): attach the SPARSE per-level visual features + the
                # full-prompt visual mask; _merge_multimodal densifies them per chunk.
                if deepstack is not None and pos_mask is not None:
                    r.deepstack_visual_embedding = np.asarray(jax.device_get(deepstack))
                    r.deepstack_visual_pos_mask = np.asarray(jax.device_get(pos_mask)).astype(bool)
                    r.apply_for_deepstack = True

    def _encode_mm_batched(self, enc_reqs, assemble, _put, _thw):
        """L-k (design §7): batch the VISION tower over ALL reqs needing encode (one ViT call for
        every image/video across the batch), then merge per req. Correctness rests on (a) the ViT
        segmenting per-image via cu_seqlens -- cross-req patch concat == one req with more images,
        no leakage (design §7.4) -- and (b) merge being 1:1 placeholder<->feature, so each req's
        per-modality feature-row count equals its placeholder count in the clean input_ids (the K-2
        invariant, exact under non-bucketing). Audio is encoded PER-REQ (the RVQ/mel tower's
        per-clip segmentation is not yet asserted, so cross-req audio concat is conservatively
        avoided); audio batching is a follow-up. Caller guarantees jitted_encode_mm + bucketing-off.
        """
        img_tok = getattr(self.model, "image_token_id", None)
        vid_tok = getattr(self.model, "video_token_id", None)
        aud_tok = getattr(self.model, "audio_token_id", None)

        img_px, img_grids, vid_px, vid_grids = [], [], [], []
        plan = []  # per req: (r, input_ids_np, {modality: nrows}, audio_codes_or_None)
        for r in enc_reqs:
            a = assemble(r.mm_inputs)
            ids = np.asarray(r.origin_input_ids, dtype=np.int32)
            rows, aud = {}, None
            ip = a.get("pixel_values_images")
            if ip is not None and img_tok is not None:
                img_px.append(np.asarray(ip))
                img_grids.extend(a.get("image_grid_thw") or [])
                rows["image"] = int(np.count_nonzero(ids == img_tok))
            vp = a.get("pixel_values_videos")
            if vp is not None and vid_tok is not None:
                vid_px.append(np.asarray(vp))
                vid_grids.extend(a.get("video_grid_thw") or [])
                rows["video"] = int(np.count_nonzero(ids == vid_tok))
            ac = a.get("audio_codes")
            if ac is not None and aud_tok is not None:
                aud = np.asarray(ac)
            plan.append((r, ids, rows, aud))

        # One batched ViT pass over all reqs' image+video patches.
        vfeats = {}
        if img_px or vid_px:
            vfeats = self.jitted_encode_mm(
                mm_pixel_values=_put(np.concatenate(img_px, 0), bf16=True) if img_px else None,
                mm_grid_thw=_thw(img_grids) if img_grids else None,
                mm_pixel_values_videos=(
                    _put(np.concatenate(vid_px, 0), bf16=True) if vid_px else None
                ),
                mm_video_grid_thw=_thw(vid_grids) if vid_grids else None,
            )

        cum = {"image": 0, "video": 0}
        pending = []  # (r, fused_device) -- merges dispatched async, gathered in ONE device_get
        for r, ids, rows, aud in plan:
            kw = {}
            for m in ("image", "video"):
                n = rows.get(m)
                if n:
                    kw[m] = vfeats[m][cum[m] : cum[m] + n]
                    cum[m] += n
            if aud is not None:
                # Audio per-req (batch of 1) -- conservatively not concatenated across reqs.
                kw["audio"] = self.jitted_encode_mm(mm_audio_codes=_put(aud))["audio"]
            if not kw:
                continue  # mm_inputs but no encodable feature -> treated as text (no embedding)
            fused, _, _ = self.jitted_merge_mm(_put(ids), **kw)
            pending.append((r, fused))
        # Partial 甲->乙 (design §5.2.2): gather all reqs' fused embeddings in ONE device_get instead
        # of a blocking per-req device_get inside the loop -> the N merge jits dispatch async and
        # their D2H copies overlap, removing the serial sync that stalled the encode loop. (The full
        # device-resident handoff -- no D2H at all -- needs a multi-host-SPMD-aware device-side
        # assembly in _merge_multimodal/forward_batch_info and is a scoped follow-up; design §7.)
        if pending:
            host_embeds = jax.device_get([f for _, f in pending])
            for (r, _), emb in zip(pending, host_embeds):
                r.multimodal_embedding = np.asarray(emb)

    def _aot_vision_reserve_bytes(self) -> int:
        """G1 (design §5.7 option 3): AOT-lower the encode jit at the max vision shape and read
        its XLA temp_size -> a tight HBM reserve. Best-effort: returns 0 on any failure so the
        caller falls back to the conservative closed form. Runs once at startup."""
        handle = getattr(self, "_embed_mm_aot", None)
        max_patches = getattr(self.server_args, "vision_max_patches", 0) or 0
        if handle is None or max_patches <= 0:
            return 0
        try:
            jit_fn, model_def, model_state_def = handle
            hf = self.model_config.hf_config
            vcfg = getattr(hf, "vision_config", None) or getattr(
                getattr(hf, "thinker_config", None), "vision_config", None
            )
            patch = int(getattr(vcfg, "patch_size", 14))
            tpatch = int(getattr(vcfg, "temporal_patch_size", 2))
            chans = int(getattr(vcfg, "in_channels", None) or getattr(vcfg, "num_channels", 3))
            merge = int(getattr(vcfg, "spatial_merge_size", 2))
            patch_dim = chans * tpatch * patch * patch
            # Round the probe side UP so side^2 >= max_patches (a sqrt-floor would under-measure,
            # and T^2 attention amplifies even a small under-count past the 10% headroom -- MED-1).
            s = int(max_patches**0.5)
            if s * s < max_patches:
                s += 1
            side = max(merge, ((s + merge - 1) // merge) * merge)  # multiple of merge
            # When V-2 bucketing is on, the runtime worst compile shape is the largest canonical
            # bucket grid -> round the LLM-grid side up to a bucket multiple to match it.
            bucket_s = int(getattr(self.server_args, "vision_bucket_size", 0) or 0)
            if bucket_s > 0:
                llm = ((side // merge + bucket_s - 1) // bucket_s) * bucket_s
                side = llm * merge
            patches = side * side
            grid = ((1, side, side),)
            seq = patches // (merge * merge) + 16
            repl = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            ids = jax.ShapeDtypeStruct((seq,), jnp.int32, sharding=repl)
            px = jax.ShapeDtypeStruct((patches, patch_dim), jnp.bfloat16, sharding=repl)
            compiled = jit_fn.lower(
                model_def,
                model_state_def,
                self.model_state_leaves,
                ids,
                px,
                grid,
                None,
                None,
                None,
                None,
            ).compile()
            temp = int(compiled.memory_analysis().temp_size_in_bytes)
            logger.info(
                "G1 AOT: vision encode temp_size=%.2f GiB @ %d patches", temp / (1024**3), patches
            )
            return int(temp * 1.1)  # 10% headroom
        except Exception as e:  # noqa: BLE001 - best-effort probe; closed-form is the fallback
            logger.warning("G1 AOT vision reserve probe failed (%s); using closed-form estimate", e)
            return 0

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
        if self.is_draft_worker:
            if full_layers_num == 0:
                self.full_max_total_num_tokens = 0
                self.swa_max_total_num_tokens = self.max_total_num_tokens
            else:
                self.full_max_total_num_tokens = self.max_total_num_tokens
                self.swa_max_total_num_tokens = int(
                    self.max_total_num_tokens * swa_full_tokens_ratio
                )
        else:
            self.full_max_total_num_tokens = int(total_tokens / denominator)
            self.swa_max_total_num_tokens = int(
                self.full_max_total_num_tokens * swa_full_tokens_ratio
            )

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
