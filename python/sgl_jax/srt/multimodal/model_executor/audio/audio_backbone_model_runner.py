"""Audio Backbone Model Runner for MiMo Audio."""

import json
import os
from functools import partial
from typing import Optional

import huggingface_hub
import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.configs.config_registry import get_audio_backbone_config
from sgl_jax.srt.server_args import ServerArgs


class AudioBackboneModelRunner(BaseModelRunner):
    """Model runner for MiMo Audio Backbone (LLM with audio generation)."""

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
        self.initialize()

    def initialize(self):
        self.load_model()
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
                        value = tuple(ast.literal_eval(value))
                    elif isinstance(value, list):
                        value = tuple(value)
                elif key == "delay_pattern":
                    if isinstance(value, str) and "-" in value:
                        value = tuple(int(x) for x in value.split("-"))
                    elif isinstance(value, str):
                        import ast
                        value = tuple(ast.literal_eval(value))
                    elif isinstance(value, list):
                        value = tuple(value)
                setattr(self.model_config, key, value)

        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class

        # Create audio arguments from HF config
        self.audio_args = MiMoAudioArguments(
            model_name_or_path=self.server_args.model_path,
            sosp_idx=hf_config.get("sosp_idx", 0),
            eosp_idx=hf_config.get("eosp_idx", 0),
            sostm_idx=hf_config.get("sostm_idx", 0),
            eostm_idx=hf_config.get("eostm_idx", 0),
            eot_idx=hf_config.get("eot_idx", 0),
            empty_idx=hf_config.get("empty_idx", 0),
        )

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def forward(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            cache,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.forward(input_ids, cache)

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
            cache: Optional[list] = None,
        ):
            return forward(model_def, model_state_def, model_state_leaves, input_ids, cache)

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

    def forward(
        self,
        input_ids: jax.Array,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            cache: Optional KV cache

        Returns:
            (text_logits, local_hidden_states, cache), cache_miss_count
        """
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count:
            output = self.jitted_forward(input_ids, cache)
            cache_miss_count = count()
        return output, cache_miss_count

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

        with jtu.count_pjit_cpp_cache_miss() as count:
            output = self.jitted_patch_decode(local_embeds, key, sampler_config)
            cache_miss_count = count()
        return output, cache_miss_count

    def init_cache(self, batch_size: int) -> list:
        """Initialize KV cache for main transformer."""
        return self.model.model.init_cache(
            batch_size,
            self.model_config.max_position_embeddings,
            jnp.bfloat16,
        )
