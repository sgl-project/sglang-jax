"""Serving backend and abstract cache factories for offline forwards."""

from types import SimpleNamespace

import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import AttentionArch
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
    _linear_state_params_from_config,
)
from sgl_jax.srt.server_args import ServerArgs


class AbstractResources(ModelRunnerKVCacheMixin):
    """Only the context consumed by serving's backend/cache factories.

    No ModelRunner initialization, device-memory query, allocator or scheduler.
    Capacities are explicit CLI inputs instead of a hardware memory budget.
    """

    def __init__(self, model_config, model, options, mesh, server_args=None):
        self.model_config = model_config
        self.model = model
        self.mesh = mesh
        self.server_args = server_args or SimpleNamespace(
            attention_backend=options.attention_backend,
            device=options.target,
            gdn_prefill_impl=ServerArgs.gdn_prefill_impl,
        )
        self.attention_tp_size = options.tp_size // options.dp_size
        self.dp_size = options.dp_size
        self.dtype = model_config.dtype
        self.is_draft_worker = False
        self.spec_algorithm = None
        self.sliding_window_size = model_config.sliding_window
        self.num_attn_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
            self.attention_tp_size
        )
        self.use_mla_backend = model_config.attention_arch == AttentionArch.MLA
        self.page_size = options.page_size
        self.kv_cache_dtype = jnp.bfloat16
        self.max_total_num_tokens = options.kv_capacity
        self.full_max_total_num_tokens = options.kv_capacity
        self.swa_max_total_num_tokens = options.kv_capacity
        # Inspect instantiated attention layers: this works for targets and draft
        # blocks, including configurations whose target layer pattern is longer.
        attention = [
            module for _, module in nnx.iter_graph(model) if isinstance(module, RadixAttention)
        ]
        swa = sorted(
            {layer.layer_id for layer in attention if (layer.sliding_window_size or 0) > 0}
        )
        full = sorted(
            {layer.layer_id for layer in attention if (layer.sliding_window_size or 0) <= 0}
        )
        self.is_hybrid = bool(swa) and self.linear_recurrent_config is None
        if self.is_hybrid:
            model_config.swa_attention_layer_ids = swa
            model_config.full_attention_layer_ids = full
        self._validate_kv_pool_compatibility()
        self.attn_backend = ModelRunner._get_attention_backend(self)
        if server_args is not None:
            self._init_kv_cache_dtype()
            if self.linear_recurrent_config is not None:
                _enforce_recurrent_state_server_constraints(
                    server_args,
                    is_lightning=(
                        self.lightning_config is not None
                        and not getattr(self.linear_recurrent_config, "use_kda", False)
                    ),
                )
                if server_args.max_recurrent_state_size is None:
                    if server_args.disable_radix_cache and server_args.max_running_requests:
                        server_args.max_recurrent_state_size = server_args.max_running_requests
                    else:
                        raise ValueError(
                            "CPU AOT needs --max-recurrent-state-size; available TPU HBM "
                            "cannot be queried on the compilation host"
                        )
                options.recurrent_capacity = server_args.max_recurrent_state_size
                if options.recurrent_capacity <= 0 or options.recurrent_capacity % self.dp_size:
                    raise ValueError(
                        "max_recurrent_state_size must be positive and divisible by dp_size"
                    )
            # Match serving's per-DP cap, page alignment, request limit and SWA split.
            self.max_total_num_tokens = self._apply_token_constraints(
                server_args.max_total_tokens, server_args.max_total_tokens, self.dp_size
            )
            self.max_num_reqs = self._resolve_max_num_reqs(server_args.max_running_requests)
            if self.is_hybrid:
                ModelRunner.set_num_token_hybrid(self)

    def adjust_layer_num(self):
        return ModelRunner.adjust_layer_num(self)

    def create_pools(self, options):
        pool = self._create_token_to_kv_pool(options.dp_size, abstract=True)
        recurrent_pool = None
        if self.linear_recurrent_config is not None:
            params = _linear_state_params_from_config(self.linear_recurrent_config)
            recurrent_pool = RecurrentStatePool(
                linear_recurrent_layer_ids=params.layers,
                size=options.recurrent_capacity or options.batch_size,
                num_heads=params.num_heads,
                head_dim=params.head_dim,
                conv_kernel_size=params.conv_kernel_size,
                mesh=self.mesh,
                dp_size=options.dp_size,
                temporal_dtype=params.dtype.temporal,
                conv_dtype=params.dtype.conv,
                num_k_heads=params.num_k_heads,
                head_k_dim=params.head_k_dim,
                abstract=True,
            )
        elif options.recurrent_capacity is not None:
            raise ValueError("recurrent_capacity requires a recurrent-state cache")
        return MemoryPools(token_to_kv_pool=pool, recurrent_state_pool=recurrent_pool)
