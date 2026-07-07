import logging
import math
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model, create_qwen2_weight_mappings

# Import the Qwen2.5-VL vision-metadata module so model import triggers builder
# registration; the encode body consumes only the opaque ``meta`` pytree.
from sgl_jax.srt.models.vision_metadata import (  # noqa: F401
    qwen2_5_vl as _qwen25vl_vision_metadata,
)
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

init_fn = nnx.initializers.uniform()


def _apply_data_sharding(x: jax.Array, mesh: Mesh, spec: PartitionSpec) -> jax.Array:
    sharding = NamedSharding(mesh, spec)
    if "data" in mesh.abstract_mesh.explicit_axes:
        return jax.sharding.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def _apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
    # x: [dp, T, N, H]; rotary_pos_emb: [dp, T, rot] (per-image, dp-leading).
    _, _, _, H = x.shape
    half_dim = H // 2

    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # rope already carries the dp (batch) axis -> only insert the heads (N) axis.
    cos_emb = cos_emb[:, :, None, :]  # [dp, T, 1, rot]
    sin_emb = sin_emb[:, :, None, :]

    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    return jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)


def _vision_attention(
    backend,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seg: jax.Array,
) -> jax.Array:
    """Run DP-leading block-diagonal vision attention."""
    dp, T, N, H = q.shape

    # [dp, T, N, H] -> [dp, N, T, H] for the kernel.
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Pad T to the kernel block size; padding rows use a masked sentinel segment.
    T_aligned = ((T + 127) // 128) * 128
    pad = T_aligned - T
    if pad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        seg = jnp.pad(seg, ((0, 0), (0, pad)), constant_values=-1)  # [dp, T_aligned]

    segment_ids = SegmentIds(q=seg, kv=seg)

    output = backend(q, k, v, segment_ids)  # [dp, N, T_aligned, H]

    output = output[:, :, :T, :]  # slice back to real seq
    # [dp, N, T, H] -> [dp, T, N, H]
    return jnp.transpose(output, (0, 2, 1, 3))


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh = None,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.mesh = mesh
        kernel_size = (temporal_patch_size, patch_size, patch_size)

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),  # Use dummy rngs if None (for eval_shape)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (dp, seq_len, C * T * H * W) -- dp-leading batched;
        # seq_len == the per-image patch count (== patch_k in the plan).
        dp, seq_len, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size * self.patch_size)
        x = x.reshape(
            dp,
            seq_len,
            C,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        if self.mesh is not None:
            x = _apply_data_sharding(
                x,
                self.mesh,
                PartitionSpec("data", None, None, None, None, None),
            )
        # [dp, seq, C, T, H, W] -> [dp, seq, T, H, W, C]
        x = jnp.transpose(x, (0, 1, 3, 4, 5, 2))
        flat_sharding = out_sharding = None
        if self.mesh is not None and "data" in self.mesh.abstract_mesh.explicit_axes:
            flat_sharding = NamedSharding(self.mesh, PartitionSpec("data", None, None, None, None))
            out_sharding = NamedSharding(
                self.mesh,
                PartitionSpec("data", None, None, None, None, None),
            )
        x = x.reshape(
            dp * seq_len,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            C,
            out_sharding=flat_sharding,
        )
        x = self.proj(x, out_sharding=flat_sharding)
        x = x.reshape(
            dp,
            seq_len,
            1,
            1,
            1,
            self.hidden_size,
            out_sharding=out_sharding,
        )
        # After conv: [dp, seq, 1, 1, 1, hidden_size].
        x = jnp.squeeze(x, axis=(2, 3, 4))
        if self.mesh is not None:
            x = _apply_data_sharding(
                x,
                self.mesh,
                PartitionSpec("data", None, None),
            )
        return x


class Qwen2_5_VLMLP(nnx.Module):
    def __init__(self, config: QwenVLModelVitConfig, dtype: jnp.dtype, rngs: nnx.Rngs = None):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.gate_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.up_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.down_proj = nnx.Linear(
            hidden_features,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.act_fn = act_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        return self.down_proj(fuse)


class Qwen2_5_VisionAttention(nnx.Module):
    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mesh = mesh

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        # DP-only vision attention backend reused across all
        # in-model VLMs. Lazy import avoids a module-level import cycle
        # (flash_attention_backend -> schedule_batch -> models). ``mesh`` is None
        # only during eval_shape (which never calls __call__), so guard it.
        if mesh is not None:
            from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
                VisionFlashAttentionBackend,
            )

            self.attn_backend = VisionFlashAttentionBackend(mesh, sm_scale=self.scale, causal=False)
        else:
            self.attn_backend = None

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        """Run one dp-leading ViT attention block."""
        dp, T, D = x.shape

        positions = jnp.arange(T, dtype=cu_window_seqlens.dtype)
        seg = jnp.sum(
            cu_window_seqlens[:, :, None] <= positions[None, None, :],
            axis=1,
        ).astype(jnp.int32)
        if self.mesh is not None:
            seg = _apply_data_sharding(seg, self.mesh, PartitionSpec("data", None))
        if valid is not None:
            is_real = positions[None, :] < jnp.reshape(valid, (dp, 1))
            if self.mesh is not None:
                is_real = _apply_data_sharding(is_real, self.mesh, PartitionSpec("data", None))
            seg = jnp.where(is_real, seg, jnp.full_like(seg, -1))

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [dp, T, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [dp, T, D] each

        # Wrapper uses dp-leading [dp, T, N, H]; _vision_attention adapts this
        # layout to the backend's kernel contract.
        q = q.reshape(dp, T, self.num_heads, self.head_dim)
        k = k.reshape(dp, T, self.num_heads, self.head_dim)
        v = v.reshape(dp, T, self.num_heads, self.head_dim)

        # Apply rotary embeddings (rope is per-image [dp, T, rot])
        q = _apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = _apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Block-diagonal segment flash attention via the DP-only backend.
        output = _vision_attention(self.attn_backend, q, k, v, seg)  # [dp, T, N, H]

        # [dp, T, N, H] -> [dp, T, D]
        output = output.reshape(dp, T, D)

        return self.proj(output)


class Qwen2_5_VisionBlock(nnx.Module):
    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        dim = config.hidden_size
        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=norm_eps,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self.mlp = Qwen2_5_VLMLP(config=config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens, valid)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.mesh = mesh

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = norm_layer(
            context_dim, dtype=dtype, rngs=_rngs, scale_init=nnx.with_partitioning(init_fn, (None,))
        )
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [dp, T, ctx] (dp-leading).
        x = self.ln_q(x)
        dp = x.shape[0]
        # Keep dp on axis 0: the sms² spatial-merge stays WITHIN each image.
        # ``reshape(-1, ...)`` here would interleave T and dp and silently mix
        # across images.
        out_sharding = None
        if self.mesh is not None and "data" in self.mesh.abstract_mesh.explicit_axes:
            out_sharding = NamedSharding(self.mesh, PartitionSpec("data", None, None))
        x = x.reshape(
            dp,
            -1,
            self.hidden_size,
            out_sharding=out_sharding,
        )  # [dp, T/sms², ctx*sms²]
        if self.mesh is not None and out_sharding is None:
            x = _apply_data_sharding(x, self.mesh, PartitionSpec("data", None, None))
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x  # [dp, T/sms², d_model]


class Qwen2_5_VisionTransformer(nnx.Module):

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
    ):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.blocks = nnx.List(
            [
                Qwen2_5_VisionBlock(
                    config=config,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                    norm_eps=norm_eps,
                )
                for _ in range(config.depth)
            ]
        )

        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.spatial_merge_size = config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

    def __call__(
        self,
        pixels: jax.Array,
        meta,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        return self.compute_hidden_states(
            pixels,
            meta.window_index,
            meta.cu_window_seqlens,
            meta.rotary_pos_emb,
            valid,
        )

    def compute_hidden_states(
        self,
        pixels: jax.Array,
        window_index: jax.Array,
        cu_window_seqlens: jax.Array,
        rotary_pos_emb: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        """Run the dp-leading ViT encode body."""
        # pixels: [dp, seq, dim_in] (dp-leading batched).
        hidden_states = self.patch_embed(pixels)  # [dp, seq, D]
        dp = pixels.shape[0]
        seq_len = pixels.shape[1]
        u = self.spatial_merge_unit

        hidden_states = hidden_states.reshape(dp, seq_len // u, u, -1)  # [dp, seq//u, u, D]
        # Reorder spatial-merge units into window order per image.
        gather_idx = jnp.broadcast_to(window_index[:, :, None, None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, gather_idx, axis=1)
        hidden_states = hidden_states.reshape(dp, seq_len, -1)  # [dp, T, D]

        # Full-att blocks use one segment per image; windowed blocks use window boundaries.
        if valid is None:
            full_cu_window_seqlens = jnp.full((dp, 1), seq_len, dtype=cu_window_seqlens.dtype)
        else:
            full_cu_window_seqlens = jnp.reshape(valid, (dp, 1)).astype(cu_window_seqlens.dtype)

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb,
                (
                    full_cu_window_seqlens
                    if layer_num in self.fullatt_block_indexes
                    else cu_window_seqlens
                ),
                valid,
            )

        # adapter (merger): [dp, T, D] -> [dp, T/sms², d_model]
        hidden_states = self.merger(hidden_states)
        # Restore raster order per image.
        reverse_indices = jnp.argsort(window_index, axis=1)  # [dp, seq//u]
        rev_idx = jnp.broadcast_to(reverse_indices[:, :, None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, rev_idx, axis=1)
        return hidden_states  # [dp, out_rows, H]

    def encode(self, pixels: jax.Array, meta, valid: jax.Array | None = None) -> jax.Array:
        if self.mesh is None:
            return self.encode_jit(pixels, meta, valid)
        try:
            ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                ctx = self.mesh
        with ctx:
            return self.encode_jit(pixels, meta, valid)

    @jax.jit
    def encode_jit(self, pixels: jax.Array, meta, valid: jax.Array | None = None) -> jax.Array:
        features = self(pixels, meta, valid)  # [dp, out_rows, H]
        if self.mesh is None:
            return features
        return _apply_data_sharding(
            features,
            self.mesh,
            PartitionSpec("data", None, None),
        )


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """In-model Qwen2.5-VL (single-file): vision tower + Qwen2 backbone (+ MRoPE)
    + lm_head. The visual encode/merge surfaces stay outside the backbone JIT;
    MRoPE is handled transparently by the plain ``Qwen2Model`` (mrope-aware
    ``get_rope`` + 3-D ``forward_batch.mrope_positions``), so no backbone subclass.
    """

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Language backbone (Qwen2 + MRoPE) + lm_head + logits.
        self.model = Qwen2Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)
        self.image_token_id = getattr(self.config, "image_token_id", None)
        self.video_token_id = getattr(self.config, "video_token_id", None)

        # Vision tower. `self.visual` IS the ViT; the in-model embedder is
        # `get_image_feature` (resolved by `embed_mm_inputs` via
        # `getattr(model, "get_image_feature")`, no `mm_embedders` dict).
        self.visual_config = getattr(config, "vision_config", None)
        if self.visual_config is None:
            raise ValueError("Qwen2.5-VL requires config.vision_config.")
        self.visual = Qwen2_5_VisionTransformer(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(self.visual_config, "rms_norm_eps", 1e-6),
        )

    def get_image_feature(self, enc):
        """Encode one DP round of images.

        ``enc.meta`` carries scheduler-built ViT aux
        (``window_index`` / ``cu_window_seqlens`` / ``rotary_pos_emb``).
        Returns dp-leading image features with shape ``[dp, out_rows, H]``.
        """
        return self.visual.encode(enc.pixels, enc.meta, enc.valid)

    def load_weights(self, model_config: ModelConfig):
        # Load text backbone and lm_head weights.
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(create_qwen2_weight_mappings(self.text_config))
        logger.info("Qwen2.5-VL (LLM) weights loaded successfully!")
        # Vision (ViT) weights -- second WeightLoader pass mapping only visual.*.
        # The vision loader reads ONLY `model_path` (WeightLoader's safetensors
        # glob); the head/kv block in weight_utils is skipped for the
        # string-target vision mappings, so no other config field is needed.
        visual_loader_config = SimpleNamespace(model_path=model_config.model_path)
        self._load_vision_weights(visual_loader_config)

    def _load_vision_weights(self, model_config) -> None:
        """Load the ViT (``self.visual``) weights from safetensors.

        This pass maps only ``visual.*`` keys; text embeddings are owned by the
        language backbone.
        """
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen2_5_vl_vision_weight_mappings()
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen2.5-VL ViT weights loaded successfully!")

    def _create_qwen2_5_vl_vision_weight_mappings(self) -> dict:
        # Vision layers use replicated weights (no tensor parallelism). Targets are
        # relative to `self` (the wrapper), whose `self.visual` IS the ViT tower.
        mappings = {
            # Patch embed Conv3D: PyTorch [out,in,kd,kh,kw] -> JAX [kd,kh,kw,in,out]
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale",
                sharding=(None,),
                transpose=False,
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias",
                sharding=(None,),
                transpose=False,
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        num_vision_layers = getattr(self.visual_config, "depth", 0)
        for layer_idx in range(num_vision_layers):
            mappings.update(self._create_vision_layer_mappings(layer_idx))
        return mappings

    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        # Qwen2.5-VL uses visual.blocks.{i}.* for vision layers (replicated).
        prefix = f"visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"
        return {
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
        }

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            weight = self.model.embed_tokens.embedding.value
            return (weight, weight)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2_5_VLForConditionalGeneration
