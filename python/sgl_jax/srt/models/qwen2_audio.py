"""Qwen2-Audio (``Qwen2AudioForConditionalGeneration``, e.g. Qwen/Qwen2-Audio-7B-Instruct).

Whisper-style audio encoder (conv1d stem -> learned positions -> pre-LN
blocks with bidirectional attention over the valid mel frames -> 2x average
pool -> LayerNorm) -> linear projector -> Qwen2 language model.

Every clip is a fixed 30 s / 3000-frame log-mel window (the HF feature
extractor pads to it); the number of valid frames is carried per item and
masks padded frames as attention keys, exactly like HF. A clip of ``L`` valid
frames yields ``((L - 1) // 2 + 1 - 2) // 2 + 1`` LM tokens.
"""

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    MelInputSpec,
)

logger = logging.getLogger(__name__)


def audio_output_length(num_frames: int) -> int:
    """HF ``Qwen2AudioEncoder._get_feat_extract_output_lengths``."""
    conv_len = (num_frames - 1) // 2 + 1
    return (conv_len - 2) // 2 + 1


def _gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=False)


class _Linear(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, mesh, dtype, use_bias: bool = True):
        self.inner = LinearBase(
            input_size=in_dim,
            output_size=out_dim,
            mesh=mesh,
            use_bias=use_bias,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.replicated = NamedSharding(mesh, PartitionSpec())

    def __call__(self, x: jax.Array) -> jax.Array:
        out, _ = self.inner(x, out_sharding=self.replicated)
        return out


class _LayerNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        orig = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x * self.weight.value.astype(jnp.float32) + self.bias.value.astype(jnp.float32)
        return x.astype(orig)


class _Conv1d(nnx.Module):
    """PyTorch-layout Conv1d (kernel 3, padding 1) on ``[batch, time, channels]``."""

    def __init__(self, in_dim: int, out_dim: int, stride: int, dtype):
        self.weight = nnx.Param(jnp.zeros((out_dim, in_dim, 3), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype))
        self.stride = stride

    def __call__(self, x: jax.Array) -> jax.Array:
        out = jax.lax.conv_general_dilated(
            x,
            self.weight.value,
            window_strides=(self.stride,),
            padding=((1, 1),),
            dimension_numbers=("NWC", "OIW", "NWC"),
        )
        return out + self.bias.value


class Qwen2AudioEncoderLayer(nnx.Module):
    def __init__(self, cfg, mesh, dtype):
        d = cfg.d_model
        self.num_heads = cfg.encoder_attention_heads
        self.head_dim = d // self.num_heads
        self.q_proj = _Linear(d, d, mesh, dtype)
        self.k_proj = _Linear(d, d, mesh, dtype, use_bias=False)
        self.v_proj = _Linear(d, d, mesh, dtype)
        self.out_proj = _Linear(d, d, mesh, dtype)
        self.fc1 = _Linear(d, cfg.encoder_ffn_dim, mesh, dtype)
        self.fc2 = _Linear(cfg.encoder_ffn_dim, d, mesh, dtype)
        self.self_attn_layer_norm = _LayerNorm(d, 1e-5, dtype)
        self.final_layer_norm = _LayerNorm(d, 1e-5, dtype)

    def __call__(self, x: jax.Array, key_bias: jax.Array) -> jax.Array:
        b, s, d = x.shape
        h = self.self_attn_layer_norm(x).reshape(b * s, d)
        q = self.q_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        attn = jax.nn.dot_product_attention(q, k, v, bias=key_bias, scale=self.head_dim**-0.5)
        x = x + self.out_proj(attn.reshape(b * s, d)).reshape(b, s, d)
        h = self.final_layer_norm(x).reshape(b * s, d)
        return x + self.fc2(_gelu(self.fc1(h))).reshape(b, s, d)


class Qwen2AudioTower(nnx.Module):
    def __init__(self, config, text_hidden: int, mesh, dtype):
        ac = config.audio_config
        if getattr(ac, "scale_embedding", False):
            raise NotImplementedError("scale_embedding=True is not supported")
        self.mesh = mesh
        self.dtype = dtype
        self.num_mel_bins = ac.num_mel_bins
        self.max_positions = ac.max_source_positions
        self.num_frames = self.max_positions * 2  # conv2 stride
        self.conv1 = _Conv1d(ac.num_mel_bins, ac.d_model, 1, dtype)
        self.conv2 = _Conv1d(ac.d_model, ac.d_model, 2, dtype)
        self.embed_positions = nnx.Param(jnp.zeros((self.max_positions, ac.d_model), dtype=dtype))
        self.layers = nnx.List(
            [Qwen2AudioEncoderLayer(ac, mesh, dtype) for _ in range(ac.encoder_layers)]
        )
        self.layer_norm = _LayerNorm(ac.d_model, 1e-5, dtype)
        self.projector = _Linear(ac.d_model, text_hidden, mesh, dtype)

    @jax.jit
    def encode(self, mel: jax.Array, num_frames: jax.Array) -> jax.Array:
        """mel ``[clips, 3000, n_mels]``, num_frames ``[clips]`` -> ``[clips, 750, text_hidden]``."""
        replicated = NamedSharding(self.mesh, PartitionSpec())
        x = jax.sharding.reshard(mel, replicated).astype(self.dtype)
        x = _gelu(self.conv1(x))
        x = _gelu(self.conv2(x))
        x = x + self.embed_positions.value[None]

        conv_len = (num_frames - 1) // 2 + 1
        valid = jnp.arange(self.max_positions)[None, :] < conv_len[:, None]
        key_bias = jnp.where(valid, 0.0, jnp.finfo(jnp.float32).min).astype(x.dtype)
        key_bias = key_bias[:, None, None, :]
        for layer in self.layers:
            x = layer(x, key_bias)

        b, s, d = x.shape
        x = x.reshape(b, s // 2, 2, d).mean(axis=2)
        x = self.layer_norm(x)
        return self.projector(x.reshape(-1, d)).reshape(b, s // 2, -1)


class Qwen2AudioForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config.text_config
        self.dtype = dtype or jnp.bfloat16
        self.model = Qwen2Model(self.text_config, mesh=mesh, dtype=self.dtype)
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        if not self.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            self.text_config.vocab_size,
            mesh=self.mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.audio = Qwen2AudioTower(config, self.text_config.hidden_size, mesh, self.dtype)
        self.audio_input_spec = MelInputSpec(
            num_mel_bins=self.audio.num_mel_bins,
            time_alignment=self.audio.num_frames,
            dtype=np.dtype("float32"),
        )
        self.tokens_per_clip = self.audio.max_positions // 2

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def get_audio_feature(self, items_per_lane: list[list[MultimodalDataItem]]) -> jax.Array:
        """Encode clips per lane; each lane packs its items' valid tokens back to back."""
        num_lanes = len(items_per_lane)
        max_items = max(1, max(len(lane) for lane in items_per_lane))
        bucket = 1 << (max_items - 1).bit_length()  # bound recompiles
        capacity = bucket * self.tokens_per_clip
        mel = np.zeros(
            (num_lanes * bucket, self.audio.num_frames, self.audio.num_mel_bins), np.float32
        )
        frames = np.ones((num_lanes * bucket,), np.int32)
        gather = np.zeros((num_lanes * capacity,), np.int32)
        keep = np.zeros((num_lanes * capacity,), np.bool_)
        for lane_index, lane in enumerate(items_per_lane):
            offset = lane_index * capacity
            for slot, item in enumerate(lane):
                clip = lane_index * bucket + slot
                feature = np.asarray(item.feature, np.float32)
                mel[clip, : feature.shape[0]] = feature
                frames[clip] = int(np.asarray(item.get("num_frames")).reshape(()))
                length = sum(end - start for start, end in item.placeholder_ranges)
                if length != audio_output_length(int(frames[clip])):
                    raise ValueError(
                        f"audio item has {length} placeholders but encodes to "
                        f"{audio_output_length(int(frames[clip]))} tokens"
                    )
                rows = clip * self.tokens_per_clip + np.arange(length)
                gather[offset : offset + length] = rows
                keep[offset : offset + length] = True
                offset += length
        with jax.set_mesh(self.mesh):
            encoded = self.audio.encode(jnp.asarray(mel), jnp.asarray(frames))
            flat = encoded.reshape(-1, encoded.shape[-1])
            out = jnp.where(jnp.asarray(keep)[:, None], flat[jnp.asarray(gather)], 0)
        return jax.sharding.reshard(out, NamedSharding(self.mesh, PartitionSpec()))

    def get_multimodal_encode_funcs(self):
        return {Modality.AUDIO: self.get_audio_feature}

    def load_weights(self, model_config: ModelConfig) -> None:
        mappings = {}
        for key, spec in self._language_weight_mappings().items():
            # hub: "language_model.model.*"; transformers>=5 save_pretrained:
            # "language_model.model.model.*" (lm_head stays "language_model.lm_head").
            mappings[f"language_model.{key}"] = spec
            mappings[f"language_model.model.{key}"] = spec
            mappings[key.replace("model.", "model.language_model.", 1)] = spec
        for key, spec in self._audio_weight_mappings().items():
            mappings[key] = spec
            mappings[f"model.{key}"] = spec
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        if not loader.dummy_mode:
            present = loader.metadata
            mappings = {k: v for k, v in mappings.items() if k in present}
        loader.load(mappings)
        logger.info("Qwen2-Audio weights loaded.")

    def _language_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not self.tie_word_embeddings:
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")
        for layer_idx in range(self.text_config.num_hidden_layers):
            mappings.update(
                Qwen2_5_VLForConditionalGeneration._language_layer_mappings(self, layer_idx)
            )
        return mappings

    def _audio_weight_mappings(self) -> dict:
        rep2, rep1 = (None, None), (None,)

        def lin(src, dst, bias=True):
            out = {f"{src}.weight": WeightSpec(f"{dst}.inner.weight", sharding=rep2, transpose=True)}
            if bias:
                out[f"{src}.bias"] = WeightSpec(f"{dst}.inner.bias", sharding=rep1)
            return out

        def norm(src, dst):
            return {
                f"{src}.weight": WeightSpec(f"{dst}.weight", sharding=rep1),
                f"{src}.bias": WeightSpec(f"{dst}.bias", sharding=rep1),
            }

        at = "audio_tower"
        mappings = {
            f"{at}.conv1.weight": WeightSpec("audio.conv1.weight", sharding=(None, None, None)),
            f"{at}.conv1.bias": WeightSpec("audio.conv1.bias", sharding=rep1),
            f"{at}.conv2.weight": WeightSpec("audio.conv2.weight", sharding=(None, None, None)),
            f"{at}.conv2.bias": WeightSpec("audio.conv2.bias", sharding=rep1),
            f"{at}.embed_positions.weight": WeightSpec("audio.embed_positions", sharding=rep2),
            **norm(f"{at}.layer_norm", "audio.layer_norm"),
            **lin("multi_modal_projector.linear", "audio.projector"),
        }
        for i in range(len(self.audio.layers)):
            src, dst = f"{at}.layers.{i}", f"audio.layers.{i}"
            mappings.update(lin(f"{src}.self_attn.q_proj", f"{dst}.q_proj"))
            mappings.update(lin(f"{src}.self_attn.k_proj", f"{dst}.k_proj", bias=False))
            mappings.update(lin(f"{src}.self_attn.v_proj", f"{dst}.v_proj"))
            mappings.update(lin(f"{src}.self_attn.out_proj", f"{dst}.out_proj"))
            mappings.update(lin(f"{src}.fc1", f"{dst}.fc1"))
            mappings.update(lin(f"{src}.fc2", f"{dst}.fc2"))
            mappings.update(norm(f"{src}.self_attn_layer_norm", f"{dst}.self_attn_layer_norm"))
            mappings.update(norm(f"{src}.final_layer_norm", f"{dst}.final_layer_norm"))
        return mappings

    def get_embed_and_head(self):
        w = self.model.embed_tokens.embedding.value
        return (w, w if self.tie_word_embeddings else self.lm_head.embedding.value)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        head = self.model.embed_tokens if self.tie_word_embeddings else self.lm_head
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2AudioForConditionalGeneration
