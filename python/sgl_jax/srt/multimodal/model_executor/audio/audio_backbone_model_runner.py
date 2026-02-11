"""Audio Backbone Model Runner for MiMo Audio.

Uses RadixAttention with proper mesh context handling.
"""

import json
import logging
import os
from functools import partial
from typing import Optional

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.configs.config_registry import get_audio_backbone_config
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class AudioBackboneModelRunner(BaseModelRunner):
    """Model runner for MiMo Audio Backbone (LLM with audio generation).

    Uses RadixAttention with proper mesh context handling following
    the pattern from standard ModelRunner.
    """

    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=model_class,
                sub_dir=None,
            ),
            mesh=self.mesh,
        )
        self.model_class = model_class
        self.server_args = server_args
        self.page_size = server_args.page_size if server_args else 1
        self.tp_size = server_args.tp_size if server_args else 1
        self.initialize()

    def initialize(self):
        self.load_model()
        self.init_attention_backend()
        self.init_memory_pool()
        self.initialize_jit()

    def _load_hf_config(self, model_path: str) -> dict:
        """Load config.json from HuggingFace model path."""
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        else:
            config_path = huggingface_hub.hf_hub_download(
                model_path,
                "config.json",
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        with open(config_path, "r") as f:
            return json.load(f)

    def load_model(self):
        hf_config = self._load_hf_config(self.server_args.model_path)
        self.model_config = get_audio_backbone_config(self.server_args.model_path)

        # Update config with values from HF config
        for key, value in hf_config.items():
            if hasattr(self.model_config, key):
                if key in ("speech_vocab_sizes", "speech_empty_ids"):
                    if isinstance(value, str):
                        import ast
                        value = list(ast.literal_eval(value))
                elif key == "delay_pattern":
                    if isinstance(value, str) and "-" in value:
                        value = [int(x) for x in value.split("-")]
                    elif isinstance(value, str):
                        import ast
                        value = list(ast.literal_eval(value))
                setattr(self.model_config, key, value)

        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class

        # Create audio arguments from HF config
        # Default values match MiMo Audio tokenizer special tokens:
        # <|sosp|>: 151665, <|eosp|>: 151666, <|empty|>: 151667
        # <|sostm|>: 151670, <|eostm|>: 151671, <|eot|>: 151672
        self.audio_args = MiMoAudioArguments(
            model_name_or_path=self.server_args.model_path,
            sosp_idx=hf_config.get("sosp_idx", 151665),
            eosp_idx=hf_config.get("eosp_idx", 151666),
            sostm_idx=hf_config.get("sostm_idx", 151670),
            eostm_idx=hf_config.get("eostm_idx", 151671),
            eot_idx=hf_config.get("eot_idx", 151672),
            empty_idx=hf_config.get("empty_idx", 151667),
        )

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )

        # Parse model config
        self.dtype = self.model_config.dtype if hasattr(self.model_config, 'dtype') else jnp.bfloat16
        self.kv_cache_dtype = jnp.bfloat16

    def init_attention_backend(self):
        """Initialize attention backend.

        Note: For audio backbone, we force native attention because FlashAttention
        requires additional metadata (cu_q_lens, cu_kv_lens, distribution, etc.)
        that are computed by the standard scheduler flow.
        """
        num_attn_heads = self.model_config.num_attention_heads
        num_kv_heads = self.model_config.num_key_value_heads

        # Force native attention for audio backbone
        # FlashAttention requires forward_metadata with cu_q_lens, cu_kv_lens, etc.
        # which are computed in the standard scheduler flow
        from sgl_jax.srt.layers.attention.native_backend import NativeAttention
        self.attn_backend = NativeAttention(num_attn_heads, num_kv_heads, self.mesh)
        logger.info("AudioBackboneModelRunner using NativeAttention backend")

    def init_memory_pool(self):
        """Initialize KV cache memory pool."""
        # Calculate max tokens based on available memory
        max_total_num_tokens = 8192  # Default value for audio backbone
        max_num_reqs = 16

        self.max_total_num_tokens = max_total_num_tokens

        # Create request to token pool
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.max_position_embeddings + 4,
            dtype=np.int32,
        )

        # Create KV cache pool
        head_dim_aligned = (self.model_config.head_dim + 127) // 128 * 128
        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.num_key_value_heads,
            head_dim=head_dim_aligned,
            layer_num=self.model_config.num_hidden_layers,
            mesh=self.mesh,
        )

        # Create allocator
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            kvcache=self.token_to_kv_pool,
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            donate_argnames=["token_to_kv_pool"],
            static_argnames=["model_state_def"],
        )
        def forward(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            forward_batch,
            token_to_kv_pool,
            logits_metadata,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.forward(input_ids, forward_batch, token_to_kv_pool, logits_metadata)

        @partial(
            jax.jit,
            static_argnames=["model_state_def", "do_sample", "temperature"],
        )
        def patch_decode(
            model_def,
            model_state_def,
            model_state_leaves,
            local_embeds,
            key,
            do_sample,
            temperature,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            sampler_config = MiMoSamplerConfig(do_sample=do_sample, temperature=temperature)
            return model.patch_decode(local_embeds, key, sampler_config)

        def forward_wrapper(
            input_ids: jax.Array,
            forward_batch: ForwardBatch,
            logits_metadata: LogitsMetadata,
        ):
            token_to_kv_pool = self.token_to_kv_pool
            return forward(
                model_def,
                model_state_def,
                model_state_leaves,
                input_ids,
                forward_batch,
                token_to_kv_pool,
                logits_metadata,
            )

        def patch_decode_wrapper(
            local_embeds: jax.Array,
            key: jax.Array,
            sampler_config: Optional[MiMoSamplerConfig] = None,
        ):
            if sampler_config is None:
                sampler_config = MiMoSamplerConfig()
            return patch_decode(
                model_def,
                model_state_def,
                model_state_leaves,
                local_embeds,
                key,
                sampler_config.do_sample,
                sampler_config.temperature,
            )

        self.jitted_forward = forward_wrapper
        self.jitted_patch_decode = patch_decode_wrapper

    def _get_mesh_context(self):
        """Get mesh context manager for JAX operations.

        Following the pattern from standard ModelRunner._forward_raw.
        """
        try:
            return jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                return jax.set_mesh(self.mesh)
            except AttributeError:
                return self.mesh

    def forward(
        self,
        input_ids: jax.Array,
        forward_batch: ForwardBatch,
        logits_metadata: LogitsMetadata,
        **kwargs,
    ):
        """Forward pass through main transformer using RadixAttention.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            forward_batch: Batch metadata for RadixAttention
            logits_metadata: Metadata for logits processing

        Returns:
            (text_logits, local_hidden_states, None, layers_kv_fused, layers_callback_flag), cache_miss_count
        """
        cache_miss_count = 0
        import jax._src.test_util as jtu

        # Use mesh context for RadixAttention operations
        with self._get_mesh_context():
            with jtu.count_pjit_cpp_cache_miss() as count:
                result = self.jitted_forward(input_ids, forward_batch, logits_metadata)
                cache_miss_count = count()

            # Handle KV cache update after forward
            if len(result) == 5:
                text_logits, local_hidden_states, cache, layers_kv_fused, layers_callback_flag = result
                self._set_kv_cache_after_forward(layers_kv_fused)
                return (text_logits, local_hidden_states, cache), cache_miss_count
            else:
                return result, cache_miss_count

    def _set_kv_cache_after_forward(self, layers_kv_fused):
        """Update KV cache after forward pass."""
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

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Generate audio tokens for one group using patch decoder.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels], cache_miss_count
        """
        cache_miss_count = 0
        import jax._src.test_util as jtu

        # Use mesh context for operations
        with self._get_mesh_context():
            with jtu.count_pjit_cpp_cache_miss() as count:
                output = self.jitted_patch_decode(local_embeds, key, sampler_config)
                cache_miss_count = count()

        return output, cache_miss_count
