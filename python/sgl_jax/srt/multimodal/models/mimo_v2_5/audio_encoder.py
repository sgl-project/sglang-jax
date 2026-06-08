"""MiMo-V2.5 audio understanding tower (host RVQ codes -> [N, hidden]).

Split out of ``embedding.py`` (design doc §5). Consumes discrete RVQ
``audio_codes`` (the host-side codec produces them) and runs the frozen
per-channel ``speech_embeddings`` -> ``input_local_transformer`` (6-layer
full-attention) -> 2-layer ``projection`` path, emitting one ``[hidden]`` row per
LLM audio token. Image/video towers are intentionally out of scope this round.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone import (
    MiMoAudioTransformer,
)


class MiMoV25AudioUnderstandingEncoder(nnx.Module):
    @staticmethod
    def _require_config_value(config: PretrainedConfig, name: str, expected) -> None:
        if not hasattr(config, name):
            return
        value = getattr(config, name)
        if value != expected:
            raise ValueError(
                "MiMoV25AudioUnderstandingEncoder only supports the upstream "
                f"MiMo-V2.5 audio tower config with {name}={expected!r}; got {value!r}."
            )

    @staticmethod
    def _require_zero_config_value(config: PretrainedConfig, name: str) -> None:
        if not hasattr(config, name):
            return
        value = float(getattr(config, name))
        if value != 0.0:
            raise ValueError(
                "MiMoV25AudioUnderstandingEncoder is an inference-only audio tower and "
                f"does not support {name}={value}; expected 0.0."
            )

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        # NOTE: the real audio_config stores speech_vocab_size / speech_zeroemb_idx
        # as strings, so every numeric field is coerced with int()/float() here.
        #
        # The HF input_local_transformer is a stock Qwen2Model with
        # partial_rotary_factor passed in. We reuse mimo_audio's MiMoAudioTransformer
        # (MiMoAudioDecoderLayer) because it is the only layer here that supports the
        # stateless, no-KV-cache forward this encoder needs, and it is structurally
        # equivalent to Qwen2 for the real V2.5 config (q/k/v bias, SwiGLU, neox rope,
        # no qk-norm). That layer hardcodes full rotary (rotary_dim=head_dim), which
        # matches V2.5 (partial_rotary_factor=1.0). Guard here — NOT in the shared
        # MiMoAudioAttention — so mimo-audio's patch encoder/decoder are unaffected:
        # refuse a partial-rotary checkpoint loudly instead of silently using full.
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
        if partial_rotary_factor != 1.0:
            raise ValueError(
                "MiMoV25AudioUnderstandingEncoder reuses MiMoAudioTransformer, which only "
                f"supports full rotary (rotary_dim=head_dim); got partial_rotary_factor="
                f"{partial_rotary_factor}. Wire partial rotary before serving such a checkpoint."
            )
        self._require_config_value(config, "input_full_attention", True)
        self._require_config_value(config, "add_post_norm", True)
        self._require_config_value(config, "projection_layers", 2)
        self._require_zero_config_value(config, "input_local_hidden_dropout")
        self.audio_channels = int(getattr(config, "audio_channels", 20))
        raw_vocab_size = getattr(config, "speech_vocab_size", 1280)
        raw_zeroemb_idx = getattr(
            config, "speech_zeroemb_idx", getattr(config, "zeroemb_idx", 1024)
        )
        self.speech_vocab_size = int(raw_vocab_size)
        self.zeroemb_idx = int(raw_zeroemb_idx)
        self.input_local_dim = int(getattr(config, "input_local_dim", 1024))
        self.group_size = int(getattr(config, "group_size", 4))
        self.hidden_size = int(
            getattr(config, "out_hidden_size", getattr(config, "hidden_size", 4096))
        )
        projection_hidden_size = self.input_local_dim * self.group_size * 4

        self.speech_embeddings = nnx.List(
            [
                Embed(
                    num_embeddings=self.speech_vocab_size,
                    features=self.input_local_dim,
                    dtype=dtype,
                    param_dtype=dtype,
                    kernel_axes=(None, None),
                    mesh=mesh,
                )
                for _ in range(self.audio_channels)
            ]
        )

        self.input_local_transformer = MiMoAudioTransformer(
            hidden_size=self.input_local_dim,
            num_layers=int(getattr(config, "input_local_layers", 6)),
            num_heads=int(getattr(config, "input_local_attn_heads", 16)),
            num_kv_heads=int(getattr(config, "input_local_attn_heads", 16)),
            head_dim=int(getattr(config, "input_local_head_dim", 64)),
            intermediate_size=int(
                getattr(
                    config,
                    "input_local_intermediate_size",
                    getattr(config, "input_local_ffn_dim", 4096),
                )
            ),
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 8192)),
            rope_theta=float(getattr(config, "rope_theta", 640000.0)),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            vocab_size=self.speech_vocab_size,
            mesh=mesh,
            use_bias=bool(getattr(config, "attention_bias", True)),
            use_causal_mask=False,
            has_embedder=False,
            dtype=dtype,
        )

        self.proj_fc1 = LinearBase(
            input_size=self.input_local_dim * self.group_size,
            output_size=projection_hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
        )
        self.proj_fc2 = LinearBase(
            input_size=projection_hidden_size,
            output_size=self.hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
        )

    def _group_audio_codes(self, codes: jax.Array) -> jax.Array:
        batch, channels, steps = codes.shape
        pad_len = (-steps) % self.group_size
        if pad_len:
            tail = codes[:, :, -1:]
            codes = jnp.concatenate([codes, jnp.repeat(tail, pad_len, axis=2)], axis=2)
        groups = codes.shape[2] // self.group_size
        return codes.reshape(batch, channels, groups, self.group_size).transpose(0, 2, 3, 1)

    def _require_non_empty_audio_codes(self, codes: jax.Array) -> jax.Array:
        if codes.shape[0] <= 0 or codes.shape[2] <= 0:
            raise ValueError(f"MiMo-V2.5 audio_codes cannot be empty, got shape={codes.shape}")
        return codes

    def _ensure_channel_first_audio_codes(self, codes: jax.Array) -> jax.Array:
        # Contract: host-side MiMoV25AudioCodecProcessor always emits time-major
        # codes (last axis == num_channels, see MiMoV25AudioPayload.codes_layout).
        # We therefore resolve the last axis as channels first, so the ambiguous
        # square [C, C] case is interpreted consistently with the host contract.
        if codes.ndim == 2:
            if codes.shape[-1] == self.audio_channels:
                return self._require_non_empty_audio_codes(jnp.swapaxes(codes[None, ...], 1, 2))
            if codes.shape[0] == self.audio_channels:
                return self._require_non_empty_audio_codes(codes[None, ...])
        elif codes.ndim == 3:
            if codes.shape[-1] == self.audio_channels:
                return self._require_non_empty_audio_codes(jnp.swapaxes(codes, 1, 2))
            if codes.shape[1] == self.audio_channels:
                return self._require_non_empty_audio_codes(codes)
        raise ValueError(
            "MiMo-V2.5 audio_codes must be shaped [T, C], [C, T], [B, T, C], "
            f"or [B, C, T] with C={self.audio_channels}, got shape={codes.shape}"
        )

    def __call__(
        self,
        input_features: jax.Array | None = None,
        audio_feature_lengths: jax.Array | None = None,
        audio_codes: jax.Array | None = None,
    ) -> jax.Array | None:
        if audio_codes is None:
            if input_features is not None:
                raise ValueError(
                    "MiMo-V2.5 embed stage expects host-side RVQ audio_codes; "
                    "mel/input_features codec encoding is not supported in the JAX embed stage."
                )
            return None

        codes = self._ensure_channel_first_audio_codes(audio_codes).astype(jnp.int32)
        # The audio understanding tower is tiny and runs replicated on the embed device.
        # Under the embed stage's explicit-sharding mesh the incoming codes may carry a
        # data-axis sharding, which makes the group/proj reshapes (e.g. [N,4,1024] ->
        # [1,N,4096]) unsupported without an out_sharding. Pin codes to fully replicated
        # so every downstream reshape is a local op.
        codes = jax.lax.with_sharding_constraint(
            codes, jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        )

        grouped_codes = self._group_audio_codes(codes)
        batch, groups, group_size, _ = grouped_codes.shape
        hidden = jnp.zeros((batch, groups, group_size, self.input_local_dim), dtype=self.dtype)
        for idx in range(self.audio_channels):
            channel_ids = grouped_codes[..., idx]
            embeds = self.speech_embeddings[idx](channel_ids)
            embeds = jnp.where(channel_ids[..., None] == self.zeroemb_idx, 0, embeds)
            hidden = hidden + embeds

        positions = jnp.arange(group_size, dtype=jnp.int32)
        hidden = hidden.reshape(batch * groups, group_size, self.input_local_dim)
        hidden, _, _ = self.input_local_transformer(hidden, positions)
        hidden = hidden.reshape(batch, groups, self.input_local_dim * group_size)
        hidden, _ = self.proj_fc1(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden, _ = self.proj_fc2(hidden)
        return hidden.reshape(-1, self.hidden_size)
