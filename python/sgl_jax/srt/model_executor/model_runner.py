"""ModelRunner runs the forward passes of the models."""

import logging
import os
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import mesh as mesh_lib
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import AttentionArch, MockModelConfig, ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import Sampler
from sgl_jax.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    ModelWorkerBatch,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_loader.loader import JAXModelLoader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo, SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import get_available_device_memory

logger = logging.getLogger(__name__)


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        tp_size: int,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        rngs: nnx.Rngs = None,
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.mesh = mesh
        # model args
        self.num_attn_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
            tp_size
        )
        self.rngs = rngs

        self.tp_size = tp_size
        self.server_args = server_args
        self.is_generation = model_config.is_generation
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA

        self.forward_pass_id = 0

        # Global vars
        global_server_args_dict.update(
            {k: getattr(server_args, k) for k in GLOBAL_SERVER_ARGS_KEYS}
        )

        self.model_loader = JAXModelLoader(
            load_config=LoadConfig(
                load_format=LoadFormat.JAX, download_dir=server_args.download_dir
            ),
            rngs=rngs,
            mesh=self.mesh,
        )

        # Initialize precision tracer enable state
        precision_tracer.set_enable_precision_tracer(
            server_args.enable_precision_tracer
        )

        # If it is a draft model, tp_group can be different
        self.initialize()

    def initialize(self):
        server_args = self.server_args

        # Load the model
        self.sampler = Sampler(nnx.Rngs(server_args.random_seed))
        total_device_memory = self.get_available_device_memory()
        self.load_model()

        self.initialize_jit()

        # Init memory pool and attention backends
        self.init_memory_pool(
            server_args.max_running_requests,
            server_args.max_total_tokens,
            total_device_memory,
        )

        self.init_attention_backend()

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        sampler_def, sampler_state = nnx.split(self.sampler)
        sampler_state_leaves, sampler_state_def = jax.tree_util.tree_flatten(
            sampler_state
        )

        @partial(
            jax.jit,
            donate_argnames=["forward_batch"],
            static_argnames=["model_state_def"],
        )
        def jitted_run_model(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            logits_metadata,
        ):
            model_state = jax.tree_util.tree_unflatten(
                model_state_def, model_state_leaves
            )
            model = nnx.merge(model_def, model_state)
            return model(forward_batch, logits_metadata)

        @partial(jax.jit, static_argnames=["sampler_state_def"])
        def jitted_sampler(sampler_def, sampler_state_def, sampler_state_leaves, *args):
            model_state = jax.tree_util.tree_unflatten(
                sampler_state_def, sampler_state_leaves
            )
            sampler = nnx.merge(sampler_def, model_state)
            return sampler(*args)

        self.jitted_run_model = partial(
            jitted_run_model, model_def, model_state_def, model_state_leaves
        )
        self.jitted_sampler = partial(
            jitted_sampler, sampler_def, sampler_state_def, sampler_state_leaves
        )

    def get_available_device_memory(self):
        min_available_device_memory = get_available_device_memory(
            self.device, distributed=False
        )

        # Check memory for tensor parallelism
        local_device_memory = get_available_device_memory(self.device)
        if self.tp_size > 1:
            if min_available_device_memory < local_device_memory * 0.9:
                if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                    logger.warning(
                        "The memory capacity is unbalanced. "
                        f"{min_available_device_memory=}, {local_device_memory=}, {local_device_memory * 0.9=}"
                    )
                else:
                    raise ValueError(
                        "The memory capacity is unbalanced. "
                        f"{min_available_device_memory=}, {local_device_memory=}, {local_device_memory * 0.9=}"
                    )

        return min_available_device_memory

    def load_model(self):
        self.model_config.validate_tensor_parallel_config(self.tp_size)
        self.model_config.configure_for_tensor_parallel(self.tp_size)
        self.model_config.log_kv_heads_info(self.tp_size)

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )

        self.dtype = self.model_config.dtype
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(
            self.model, "end_layer", self.model_config.num_hidden_layers
        )
        self.num_effective_layers = self.end_layer - self.start_layer

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
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        cell_size = (
            self.model_config.get_num_kv_heads(self.tp_size)
            * self.model_config.head_dim
            * self.model_config.num_hidden_layers
            * 2
            * jnp.dtype(self.kv_cache_dtype).itemsize
        )

        # Calculate max tokens that can fit in available memory
        max_tokens = max(1, int(available_kv_cache_bytes // cell_size))

        logger.info(
            f"TPU Memory profiling: "
            f"available_device_memory={available_device_memory / (1024**3):.1f}GB, "
            f"available_kv_cache={available_kv_cache_bytes / (1024**3):.1f}GB, "
            f"max_tokens={max_tokens}, "
            f"cell_size={cell_size}bytes"
        )

        return max_tokens

    def init_memory_pool(
        self,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        total_device_memory: Optional[int] = None,
    ):
        """Initialize memory pool for KV cache."""
        # Set KV cache data type
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = jnp.bfloat16
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )
        logger.info(f"ModelRunner kv_cache_dtype: {self.kv_cache_dtype}")
        # Profile maximum number of tokens
        self.max_total_num_tokens = self.profile_max_num_token(total_device_memory)

        # Calculate max number of requests if not provided
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        # Handle CI environment variable for testing
        SGLANG_CI_SMALL_KV_SIZE = os.environ.get("SGLANG_CI_SMALL_KV_SIZE")
        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        # Handle max_total_tokens override
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        # Align to page size
        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        logger.info(f"ModelRunner max_total_num_tokens: {self.max_total_num_tokens}")

        # Create request to token pool if not already created
        if self.req_to_token_pool is None:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.model_config.context_len + 4,
                dtype=np.int32,
            )

        # Create KV cache pool
        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_total_num_kv_heads_with_replication(
                self.tp_size
            ),
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            mesh=self.mesh,
        )

        # Create KV pool allocator
        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    # dtype=self.kv_cache_dtype,
                    kvcache=self.token_to_kv_pool,
                )
            else:
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    # dtype=self.kv_cache_dtype,
                    kvcache=self.token_to_kv_pool,
                    debug_mode=True,
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

            return NativeAttention(self.num_attn_heads, self.num_kv_heads)
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
            )
        else:
            raise ValueError(
                f"Unsupported attention backend: {self.server_args.attention_backend}"
            )

    def _forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count:
            output, layers_kv_fused, _ = self.jitted_run_model(
                forward_batch, logits_metadata
            )
            cache_miss_count = count()
        self._set_kv_cache_after_forward(layers_kv_fused, forward_batch)

        return output, cache_miss_count

    def _set_kv_cache_after_forward(self, layers_kv_fused, forward_batch: ForwardBatch):
        start_idx = forward_batch.token_to_kv_pool.start_layer
        end_idx = start_idx + len(layers_kv_fused)
        forward_batch.token_to_kv_pool.kv_buffer[start_idx:end_idx] = layers_kv_fused

    def forward_idle(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[LogitsProcessorOutput, int]:
        raise NotImplementedError("forward_idle is not implemented")

    def forward(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[LogitsProcessorOutput, int]:
        self.forward_pass_id += 1
        precision_tracer.start_batch_trace(forward_batch.bid)
        precision_tracer.set_current_forward_pass_id(self.forward_pass_id)
        return self._forward_raw(forward_batch, logits_metadata)

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[LogitsProcessorOutput, int]:
        # for compatibility, 0.6.3 need to use use_mesh. set_mesh is not have __entry__ attribute.
        # on jax 0.7.1, we need to use set_mesh.
        # with jax.sharding.set_mesh(self.mesh):
        with jax.sharding.use_mesh(self.mesh):
            if (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_extend()
            ):
                ret = self._forward(forward_batch, logits_metadata)
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(forward_batch, logits_metadata)
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        return ret

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        logits_output.next_token_logits = sampling_info.apply_logits_bias(
            logits_output.next_token_logits
        )

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_metadata: SamplingMetadata,
        positions: jax.Array,
    ) -> jax.Array:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
            positions: The positions of the tokens in the sequence.
        Returns:
            A list of next_token_ids
        """
        return self.jitted_sampler(
            logits_output,
            sampling_metadata,
            positions,
        )


class MockModelRunner(ModelRunner):
    def __init__(
        self,
        model_config: Union[ModelConfig, MockModelConfig],
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
            self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
                self.tp_size
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
            head_num=self.model_config.get_total_num_kv_heads_with_replication(
                self.tp_size
            ),
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            mesh=mesh,
        )
