import logging
import math

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import apply_rotary_emb
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.dits.wan_model_config import WanModelConfig
from sgl_jax.srt.multimodal.layers.attention.layer import USPAttention
from sgl_jax.srt.multimodal.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from sgl_jax.srt.multimodal.layers.mlp import MLP
from sgl_jax.srt.multimodal.layers.rotary_embedding import NDRotaryEmbedding
from sgl_jax.srt.multimodal.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit_weights_mapping import (
    to_i2v_mappings,
    to_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)


class WanImageEmbedding(nnx.Module):
    def __init__(self, in_features: int, out_features: int, mesh: jax.sharding.Mesh | None = None):
        super().__init__()

        self.norm1 = FP32LayerNorm(num_features=in_features, rngs=nnx.Rngs(0))
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu", mesh=mesh)
        self.norm2 = FP32LayerNorm(num_features=out_features, rngs=nnx.Rngs(0))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape (B, L, C) -> Channel Last
        """
        origin_dtype = x.dtype
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x).astype(origin_dtype)
        return x


class WanTransformerBlock(nnx.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        epsilon: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(
            num_features=dim, epsilon=epsilon, use_scale=False, use_bias=False, rngs=nnx.Rngs(0)
        )

        self.to_q = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_k = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_v = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_out = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

        # 1. Self-attention

        self.attn1 = USPAttention(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, epsilon=epsilon)
            self.norm_k = RMSNorm(dim_head, epsilon=epsilon)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, epsilon=epsilon)
            self.norm_k = RMSNorm(dim, epsilon=epsilon)
        else:
            raise RuntimeError("QK Norm type not supported")
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            epsilon=epsilon,
            elementwise_affine=True,
            dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                epsilon=epsilon,
                mesh=mesh,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                epsilon=epsilon,
                mesh=mesh,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            epsilon=epsilon,
            elementwise_affine=False,
            dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(
            input_dim=dim, mlp_hidden_dim=ffn_dim, act_type="gelu_pytorch_tanh", mesh=mesh
        )
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nnx.Param(
            jax.random.normal(jax.random.key(0), (1, 6, dim)) / (dim**0.5)
        )
        self.mesh = mesh

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        freqs_cis: tuple[jax.Array, jax.Array],
        req=None,
    ) -> jax.Array:
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_len, _ = hidden_states.shape
        origin_dtype = hidden_states.dtype
        # Ensure scale_shift_table is not sharded to avoid issues with split
        scale_shift_table = self.scale_shift_table.value
        if self.mesh is not None:
            no_shard = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            scale_shift_table = jax.lax.with_sharding_constraint(scale_shift_table, no_shard)
            # Keep temb unsharded as well; slicing small dims on sharded arrays fails.
            temb = jax.lax.with_sharding_constraint(temb, no_shard)

        if temb.ndim == 4:
            # temb: [batch, seq_len, 6, inner_dim]
            e = scale_shift_table[None, None, :, :] + temb.astype(jnp.float32)
            if self.mesh is not None:
                e = jax.lax.with_sharding_constraint(e, no_shard)

            # Flatten the last two dims to avoid slicing on a sharded axis.
            # e shape: [batch, seq_len, 6, inner_dim] -> [batch, seq_len, 6 * inner_dim]
            inner_dim = e.shape[-1]
            e_flat = e.reshape(e.shape[0], e.shape[1], -1)
            shift_msa = e_flat[:, :, 0:inner_dim]
            scale_msa = e_flat[:, :, inner_dim : 2 * inner_dim]
            gate_msa = e_flat[:, :, 2 * inner_dim : 3 * inner_dim]
            c_shift_msa = e_flat[:, :, 3 * inner_dim : 4 * inner_dim]
            c_scale_msa = e_flat[:, :, 4 * inner_dim : 5 * inner_dim]
            c_gate_msa = e_flat[:, :, 5 * inner_dim : 6 * inner_dim]

        else:
            # temb: [batch, 6, inner_dim]
            e = scale_shift_table + temb.astype(jnp.float32)
            if self.mesh is not None:
                e = jax.lax.with_sharding_constraint(e, no_shard)

            # Flatten the last two dims to avoid slicing on a sharded axis.
            # e shape: [batch, 6, inner_dim] -> [batch, 6 * inner_dim]
            inner_dim = e.shape[-1]
            e_flat = e.reshape(e.shape[0], -1)
            shift_msa = e_flat[:, 0:inner_dim]
            scale_msa = e_flat[:, inner_dim : 2 * inner_dim]
            gate_msa = e_flat[:, 2 * inner_dim : 3 * inner_dim]
            c_shift_msa = e_flat[:, 3 * inner_dim : 4 * inner_dim]
            c_scale_msa = e_flat[:, 4 * inner_dim : 5 * inner_dim]
            c_gate_msa = e_flat[:, 5 * inner_dim : 6 * inner_dim]

            # Broadcast per-batch modulation across sequence length.
            shift_msa = shift_msa[:, None, :]
            scale_msa = scale_msa[:, None, :]
            gate_msa = gate_msa[:, None, :]
            c_shift_msa = c_shift_msa[:, None, :]
            c_scale_msa = c_scale_msa[:, None, :]
            c_gate_msa = c_gate_msa[:, None, :]

        assert shift_msa.dtype == jnp.float32
        # 1. Self-attention
        norm1 = self.norm1(hidden_states.astype(jnp.float32))
        norm_hidden_states = (norm1 * (1 + scale_msa) + shift_msa).astype(origin_dtype)
        q, _ = self.to_q(norm_hidden_states)
        k, _ = self.to_k(norm_hidden_states)
        v, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None and self.norm_q.num_features == self.hidden_dim:
            q = self.norm_q(q)
        if self.norm_k is not None and self.norm_k.num_features == self.hidden_dim:
            k = self.norm_k(k)

        q = q.reshape(bs, seq_len, self.num_attention_heads, -1)
        k = k.reshape(bs, seq_len, self.num_attention_heads, -1)
        v = v.reshape(bs, seq_len, self.num_attention_heads, -1)

        if self.norm_q is not None and self.norm_q.num_features != self.hidden_dim:
            q = self.norm_q(q)
        if self.norm_k is not None and self.norm_k.num_features != self.hidden_dim:
            k = self.norm_k(k)
        # Apply rotary embeddings
        cos, sin = freqs_cis
        q, k = apply_rotary_emb(q, cos, sin, is_neox_style=False), apply_rotary_emb(
            k, cos, sin, is_neox_style=False
        )
        attn_output = self.attn1(q, k, v, req)
        # Flatten last two dims: (B, seq, heads, head_dim) -> (B, seq, heads*head_dim)
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
        attn_output, _ = self.to_out(attn_output)
        # Only squeeze if dim 1 has size 1 (PyTorch squeeze is no-op otherwise, JAX raises error)
        if attn_output.shape[1] == 1:
            attn_output = attn_output.squeeze(1)

        null_shift = null_scale = jnp.zeros((1,), dtype=origin_dtype)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.astype(
            origin_dtype
        ), hidden_states.astype(origin_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.astype(
            origin_dtype
        ), hidden_states.astype(origin_dtype)

        # 3. Feed-forward
        ffn_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ffn_output, c_gate_msa)
        hidden_states = hidden_states.astype(origin_dtype)
        return hidden_states


class WanSelfAttention(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        epsilon=1e-6,
        parallel_attention=False,
        mesh: jax.sharding.Mesh | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.epsilon = epsilon
        self.parallel_attention = parallel_attention

        # layers
        self.to_q = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_k = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_v = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_out = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.norm_q = RMSNorm(dim, epsilon=self.epsilon)
        self.norm_k = RMSNorm(dim, epsilon=self.epsilon)

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def __call__(self, x: jax.Array, context: jax.Array, context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def __call__(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).reshape(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.to_k(context)[0]).reshape(b, -1, n, d)
                v = self.to_v(context)[0].reshape(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.to_k(context)[0]).reshape(b, -1, n, d)
            v = self.to_v(context)[0].reshape(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output - flatten last two dims: (B, seq, heads, head_dim) -> (B, seq, heads*head_dim)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        epsilon=1e-6,
        mesh: jax.sharding.Mesh | None = None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            epsilon,
        )

        self.add_k_proj = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.add_v_proj = LinearBase(
            input_size=dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.norm_added_k = RMSNorm(dim, epsilon=epsilon)
        self.norm_added_q = RMSNorm(dim, epsilon=epsilon)

    def __call__(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).reshape(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).reshape(b, -1, n, d)
        v = self.to_v(context)[0].reshape(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).reshape(b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].reshape(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v)

        # output - flatten last two dims: (B, seq, heads, head_dim) -> (B, seq, heads*head_dim)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        img_x = img_x.reshape(img_x.shape[0], img_x.shape[1], -1)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTimeTextImageEmbedding(nnx.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu", mesh=mesh
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu", mesh=mesh)
        self.text_embedder = MLP(
            input_dim=text_embed_dim,
            mlp_hidden_dim=dim,
            output_dim=dim,
            bias=True,
            act_type="gelu_pytorch_tanh",
            mesh=mesh,
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = nnx.data(
                WanImageEmbedding(in_features=image_embed_dim, out_features=dim, mesh=mesh)
            )

    def __call__(
        self,
        timestep: jax.Array,
        encoder_hidden_states: jax.Array,
        encoder_hidden_states_image: jax.Array | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTransformer3DModel(nnx.Module):
    def __init__(
        self,
        config: WanModelConfig,
        *,
        dtype=jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.patch_size = config.patch_size
        self.hidden_size = config.num_attention_heads * config.attention_head_dim
        self.dtype = dtype
        rngs = nnx.Rngs(0)
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.sp_size = 1
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.mesh = mesh
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
            rngs=rngs,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
            mesh=mesh,
        )

        # 3. Transformer blocks
        # attn_backend = get_global_server_args().attention_backend
        transformer_block = WanTransformerBlock
        self.blocks = nnx.List(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.epsilon,
                    config.added_kv_proj_dim,
                    mesh=mesh,
                )
                for i in range(config.num_layers)
            ]
        )

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=jnp.float32,
        )

        # 4. Output norm & projection
        from sgl_jax.srt.multimodal.layers.layernorm import LayerNormScaleShift

        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            epsilon=config.epsilon,
            elementwise_affine=False,
            dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        out_channels = getattr(config, "out_channels", config.in_channels)
        self.proj_out = nnx.Linear(
            in_features=inner_dim,
            out_features=out_channels * math.prod(config.patch_size),
            use_bias=True,
            rngs=rngs,
        )
        self.scale_shift_table = nnx.Param(
            jax.random.normal(jax.random.key(0), (1, 2, inner_dim)) / (inner_dim**0.5)
        )
        self.model_config = config

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | list[jax.Array],
        timesteps: jax.Array,
        encoder_hidden_states_image: jax.Array | list[jax.Array] | None,
        guidance_scale=1.0,
        req=None,
        **kwargs,
    ):
        # origin_dtype = hidden_states.dtype
        if isinstance(encoder_hidden_states, list):
            # assert len(encoder_hidden_states) > 1, "encoder_hidden_states list is empty"
            # FIXME(pc)
            if len(encoder_hidden_states) == 0:
                # Mock encoder hidden states for testing when no text encoder is used
                # 4096 is the typical hidden dimension for T5 encoder used in Wan2.1
                # Use a sequence length of 512 to better match typical T5 output
                encoder_hidden_states = jax.random.normal(
                    jax.random.key(0), (hidden_states.shape[0], 512, 4096)
                )
            else:
                encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
            (
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            ),
            shard_dim=0,
            start_frame=0,
        )
        assert freqs_cos.dtype == jnp.float32

        # Convert from channel-first (B, C, F, H, W) to channel-last (B, F, H, W, C) for nnx.Conv
        hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = self.patch_embedding(hidden_states)
        # Flatten spatial dimensions: (B, F', H', W', C) -> (B, F'*H'*W', C)
        batch_size = hidden_states.shape[0]
        embed_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(batch_size, -1, embed_dim)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timesteps.ndim == 2:
            # ti2v
            ts_seq_len = timesteps.shape[1]
            timesteps = timesteps.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timesteps,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.reshape(timestep_proj.shape[:2] + (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.reshape(timestep_proj.shape[:1] + (6, -1))
        # Remove tensor sharding before passing to blocks to avoid "out dim not divisible by mesh axes" error in split
        if self.mesh is not None:
            sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            timestep_proj = jax.lax.with_sharding_constraint(timestep_proj, sharding)

        # Concatenate image and text embeddings if image embeddings exist
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = jnp.concatenate(
                [encoder_hidden_states_image, encoder_hidden_states], axis=1
            )

        # 4. Transformer blocks
        freqs_cis = (freqs_cos, freqs_sin)
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, req
            )

        # 5. Output norm, projection & unpatchify
        # Ensure scale_shift_table is not sharded to avoid issues with split
        scale_shift_table = self.scale_shift_table.value
        if self.mesh is not None:
            no_shard = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
            scale_shift_table = jax.lax.with_sharding_constraint(scale_shift_table, no_shard)

        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            # combined shape: [batch, seq_len, 2, inner_dim]
            combined = scale_shift_table[None, :, :, :] + temb[:, :, None, :]
            # Use manual slicing instead of jnp.split to avoid sharding issues
            shift = combined[:, :, 0, :]
            scale = combined[:, :, 1, :]
        else:
            # batch_size, inner_dim
            # combined shape: [batch, 2, inner_dim]
            combined = scale_shift_table + temb[:, None, :]
            # Use manual slicing instead of jnp.split to avoid sharding issues
            shift = combined[:, 0, :]
            scale = combined[:, 1, :]
            # Broadcast per-batch modulation across sequence length.
            shift = shift[:, None, :]
            scale = scale[:, None, :]

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify: reshape from patches back to image space
        # hidden_states shape: [batch_size, num_patches, out_channels * patch_volume]
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,  # out_channels
        )
        # Permute to rearrange patches: [B, out_channels, F, p_t, H, p_h, W, p_w]
        hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))

        # Flatten patch dimensions to get final output: [B, C, F*p_t, H*p_h, W*p_w]
        output = hidden_states.reshape(
            batch_size,
            -1,  # out_channels
            post_patch_num_frames * p_t,
            post_patch_height * p_h,
            post_patch_width * p_w,
        )

        return output

    def load_weights(self, model_path: str | WanModelConfig | None = None) -> None:
        """
        Load weights from HuggingFace safetensors files.

        Args:
            model_path: Path to the transformer directory containing safetensors files,
                       or a config object with a model_path field. If None, uses
                       self.model_config.model_path.
        """
        resolved_model_path = None
        if model_path is not None and not isinstance(model_path, str):
            resolved_model_path = getattr(model_path, "model_path", None)
        else:
            resolved_model_path = model_path

        if resolved_model_path is None:
            resolved_model_path = self.model_config.model_path

        original_path = None
        if resolved_model_path is not None:
            original_path = self.model_config.model_path
            self.model_config.model_path = resolved_model_path

        try:
            weight_mappings = (
                to_i2v_mappings()
                if self.model_config.added_kv_proj_dim is not None
                else to_mappings()
            )
            loader = WeightLoader(
                model=self,
                model_config=self.model_config,
                mesh=self.mesh,
            )
            loader.load_weights_from_safetensors(weight_mappings)
            logger.info("Weights loaded successfully for WanTransformer3DModel")
        finally:
            if original_path is not None:
                self.model_config.model_path = original_path

    @staticmethod
    def get_config_class() -> WanModelConfig:
        return WanModelConfig


class WanDualTransformer3DModel(nnx.Module):
    """
    Wan2.2 MoE (Mixture of Experts) model with dual transformers.

    This model contains two WanTransformer3DModel instances:
    - transformer (high-noise expert): Used for early denoising steps (overall layout)
    - transformer_2 (low-noise expert): Used for later denoising steps (refining details)

    The switching between transformers is controlled by boundary_ratio:
    - When timestep >= boundary_timestep: use transformer (high-noise expert)
    - When timestep < boundary_timestep: use transformer_2 (low-noise expert)
    - boundary_timestep = boundary_ratio * num_train_timesteps

    For Wan2.2 T2V A14B: boundary_ratio = 0.875 (switch at 87.5% of timesteps)
    For Wan2.2 I2V A14B: boundary_ratio = 0.900 (switch at 90% of timesteps)
    """

    # Default number of training timesteps for flow matching scheduler
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        config: WanModelConfig,
        *,
        dtype=jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        """
        Initialize dual transformer model.

        Args:
            config: Model configuration containing boundary_ratio
            dtype: Data type for computations
            mesh: JAX device mesh for sharding
        """
        self.model_config = config
        self.dtype = dtype
        self.mesh = mesh

        # Compute boundary_timestep from boundary_ratio
        if config.boundary_ratio is None:
            raise ValueError(
                "boundary_ratio must be set for WanDualTransformer3DModel. "
                "Use WanTransformer3DModel for single transformer models."
            )

        self.boundary_timestep = config.boundary_ratio * self.NUM_TRAIN_TIMESTEPS
        logger.info(
            "WanDualTransformer3DModel initialized with boundary_ratio=%.3f, boundary_timestep=%.1f",
            config.boundary_ratio,
            self.boundary_timestep,
        )

        # Create two transformer instances with the same config
        # transformer: high-noise expert (used when timestep >= boundary_timestep)
        self.transformer = WanTransformer3DModel(config, dtype=dtype, mesh=mesh)

        # transformer_2: low-noise expert (used when timestep < boundary_timestep)
        self.transformer_2 = WanTransformer3DModel(config, dtype=dtype, mesh=mesh)

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | list[jax.Array],
        timesteps: jax.Array,
        encoder_hidden_states_image: jax.Array | list[jax.Array] | None,
        guidance_scale=1.0,
        req=None,
        **kwargs,
    ):
        """
        Forward pass that selects the appropriate transformer based on timestep.

        Note: This method requires the timestep to be a scalar or the first element
        of the batch for determining which transformer to use. In typical diffusion
        inference, all samples in a batch use the same timestep.

        Args:
            hidden_states: Latent tensor of shape [B, C, T, H, W]
            encoder_hidden_states: Text embeddings
            timesteps: Timestep tensor (scalar or batch)
            encoder_hidden_states_image: Optional image embeddings for I2V
            guidance_scale: CFG guidance scale
            req: Optional request object

        Returns:
            Predicted noise tensor of shape [B, C, T, H, W]
        """
        # Get the timestep value for transformer selection
        # Use the first timestep in the batch (all should be the same in typical use)

        t_value = jnp.ravel(timesteps)[0].astype(jnp.float32)
        boundary = jnp.asarray(self.boundary_timestep, dtype=t_value.dtype)
        use_primary = t_value >= boundary

        def run_primary(_):
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                encoder_hidden_states_image=encoder_hidden_states_image,
                guidance_scale=guidance_scale,
                req=req,
                **kwargs,
            )

        def run_secondary(_):
            return self.transformer_2(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                encoder_hidden_states_image=encoder_hidden_states_image,
                guidance_scale=guidance_scale,
                req=req,
                **kwargs,
            )

        return jax.lax.cond(use_primary, run_primary, run_secondary, operand=None)

    def load_weights(self, model_path: str | WanModelConfig | None = None) -> None:
        """
        Load weights for both transformers from HuggingFace safetensors files.

        Expects the model directory to contain:
        - transformer/ directory with high-noise expert weights
        - transformer_2/ directory with low-noise expert weights

        Args:
            model_path: Path to the model directory containing transformer subdirectories,
                       or a config object with a model_path field. If None, uses
                       self.model_config.model_path.
        """
        import os

        resolved_model_path = None
        if model_path is not None and not isinstance(model_path, str):
            resolved_model_path = getattr(model_path, "model_path", None)
        else:
            resolved_model_path = model_path

        if resolved_model_path is None:
            resolved_model_path = self.model_config.model_path

        if resolved_model_path is None:
            raise ValueError("model_path must be provided for WanDualTransformer3DModel")

        base_path = resolved_model_path.rstrip(os.sep)
        base_name = os.path.basename(base_path)
        if base_name in ("transformer", "transformer_2"):
            base_path = os.path.dirname(base_path)

        # Load weights for primary transformer (high-noise expert)
        transformer_path = os.path.join(base_path, "transformer")
        if os.path.exists(transformer_path):
            # Temporarily set model_path for WeightLoader
            original_path = self.transformer.model_config.model_path
            self.transformer.model_config.model_path = transformer_path
            self.transformer.load_weights(transformer_path)
            self.transformer.model_config.model_path = original_path
            logger.info(
                "Loaded weights for transformer (high-noise expert) from %s", transformer_path
            )
        else:
            raise FileNotFoundError(f"transformer directory not found at {transformer_path}")

        # Load weights for secondary transformer (low-noise expert)
        transformer_2_path = os.path.join(base_path, "transformer_2")
        if os.path.exists(transformer_2_path):
            # Temporarily set model_path for WeightLoader
            original_path_2 = self.transformer_2.model_config.model_path
            self.transformer_2.model_config.model_path = transformer_2_path
            self.transformer_2.load_weights(transformer_2_path)
            self.transformer_2.model_config.model_path = original_path_2
            logger.info(
                "Loaded weights for transformer_2 (low-noise expert) from %s", transformer_2_path
            )
        else:
            raise FileNotFoundError(f"transformer_2 directory not found at {transformer_2_path}")

    @staticmethod
    def get_config_class() -> WanModelConfig:
        return WanModelConfig


# Factory function to create the appropriate model based on config
def create_wan_transformer(
    config: WanModelConfig,
    *,
    dtype=jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
) -> WanTransformer3DModel | WanDualTransformer3DModel:
    """
    Factory function to create the appropriate Wan transformer model.

    For Wan2.1 (single transformer): Creates WanTransformer3DModel
    For Wan2.2 (dual transformer MoE): Creates WanDualTransformer3DModel

    Args:
        config: Model configuration
        dtype: Data type for computations
        mesh: JAX device mesh for sharding

    Returns:
        WanTransformer3DModel or WanDualTransformer3DModel based on config
    """
    if config.boundary_ratio is not None:
        logger.info("Creating WanDualTransformer3DModel (Wan2.2 MoE mode)")
        return WanDualTransformer3DModel(config, dtype=dtype, mesh=mesh)
    else:
        logger.info("Creating WanTransformer3DModel (single transformer mode)")
        return WanTransformer3DModel(config, dtype=dtype, mesh=mesh)


EntryClass = WanTransformer3DModel
