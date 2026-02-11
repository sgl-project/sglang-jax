import json
import os
from functools import partial

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.config_registry import get_audio_config
from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_tokenizer import EncoderOutput
from sgl_jax.srt.server_args import ServerArgs


class AudioModelRunner(BaseModelRunner):
    def __init__(
        self, server_args: ServerArgs = None, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(model_class=model_class, sub_dir=None),
            mesh=self.mesh,
        )
        self.model_class = model_class
        self.server_args = server_args
        self.initialize()

    def initialize(self):
        self.load_model()
        self.initialize_jit()

    def _load_hf_config(self, model_path: str) -> dict:
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
        self.model_config = get_audio_config(self.server_args.model_path)
        for key, value in hf_config.items():
            if hasattr(self.model_config, key):
                if key in ("encoder_attn_window_size", "decoder_attn_window_size", "vocoder_attn_window_size"):
                    value = tuple(value) if isinstance(value, list) else value
                setattr(self.model_config, key, value)
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        self.model = self.model_loader.load_model(model_config=self.model_config)

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(jax.jit, static_argnames=["model_state_def", "use_quantizer", "n_q"])
        def encode(model_def, model_state_def, model_state_leaves, mels, input_lens, use_quantizer, n_q):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            result = model.encode(mels, input_lens, use_quantizer=use_quantizer, n_q=n_q)
            return result.hidden_states, result.packed_states, result.output_lengths, result.codes

        @partial(jax.jit, static_argnames=["model_state_def"])
        def decode(model_def, model_state_def, model_state_leaves, codes):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model.decode(codes)

        def encode_wrapper(mels: jax.Array, input_lens: jax.Array, use_quantizer: bool = True, n_q: int | None = None):
            hidden_states, packed_states, output_lengths, codes = encode(
                model_def, model_state_def, model_state_leaves, mels, input_lens, use_quantizer, n_q
            )
            return EncoderOutput(
                hidden_states=hidden_states,
                packed_states=packed_states,
                output_lengths=output_lengths,
                codes=codes,
            )

        def decode_wrapper(codes: jax.Array):
            codes_np = np.asarray(jax.device_get(codes))
            codes_clean = jnp.array(codes_np)
            return self.model.decode(codes_clean)

        self.jitted_encode = encode_wrapper
        self.jitted_decode = decode_wrapper

    def forward(self, x: jax.Array, input_lens: jax.Array | None, mode: str, **kwargs):
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count:
            if mode == "encode":
                use_quantizer = kwargs.get("use_quantizer", True)
                n_q = kwargs.get("n_q", None)
                output = self.jitted_encode(x, input_lens, use_quantizer=use_quantizer, n_q=n_q)
            elif mode == "decode":
                output = self.jitted_decode(x)
            cache_miss_count = count()
        return output, cache_miss_count
