import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.multimodal.layers.attention.layer import USPAttention
from sgl_jax.srt.multimodal.layers.layernorm import (
    FP32LayerNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from sgl_jax.srt.multimodal.layers.linear import ReplicatedLinear
from sgl_jax.srt.multimodal.layers.mlp import MLP
from sgl_jax.srt.multimodal.layers.rotary_embedding import NDRotaryEmbedding
from sgl_jax.srt.multimodal.layers.visual_embedding import PatchEmbed


class WanTransformerBlock(nnx.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(num_features=dim, epsilon=eps)

        self.to_q = ReplicatedLinear(input_size=dim, output_size=dim, use_bias=True)
        self.to_k = ReplicatedLinear(input_size=dim, output_size=dim, use_bias=True)
        self.to_v = ReplicatedLinear(input_size=dim, output_size=dim, use_bias=True)
        self.to_out = ReplicatedLinear(input_size=dim, output_size=dim, use_bias=True)

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
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
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
                eps=eps,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nnx.Param(jax.random.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        freqs_cis: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_len, _ = hidden_states.shape
        origin_dtype = hidden_states.dtypeq
        if temb.ndim == 4:
            # temb: [batch, seq_len, 6, inner_dim]
            e = self.scale_shift_table[None, None, :, :] + temb.astype(jnp.float32)

            # [batch, seq_len, 1, inner_dim]
            chunks = jnp.split(e, 6, axis=2)

            (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa) = [
                jnp.squeeze(x, 2) for x in chunks
            ]

        else:
            # temb: [batch, 6, inner_dim]
            e = self.scale_shift_table[None, :, :] + temb.astype(jnp.float32)

            (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa) = jnp.split(
                e, 6, axis=1
            )

        assert shift_msa.dtype == jnp.float32
        # 1. Self-attention
        norm1 = self.norm1(hidden_states.astype(jnp.float32))
        norm_hidden_states = (norm1 * (1 + scale_msa) + shift_msa).astype(origin_dtype)
        q, _ = self.to_q(norm_hidden_states)
        k, _ = self.to_k(norm_hidden_states)
        v, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        q = q.squeeze(1).reshape(q.shape[0], q.shape[2], self.num_attention_heads, -1)
        k = k.squeeze(1).reshape(k.shape[0], k.shape[2], self.num_attention_heads, -1)
        v = v.squeeze(1).reshape(v.shape[0], v.shape[2], self.num_attention_heads, -1)
        # Apply rotary embeddings
        cos, sin = freqs_cis
        q, k = _apply_rotary_emb(q, cos, sin, is_neox_style=False), _apply_rotary_emb(
            k, cos, sin, is_neox_style=False
        )
        attn_output = self.attn1(q, k, v)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = jnp.zeros_like((1,), dtype=origin_dtype)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.astype(
            origin_dtype
        ), hidden_states.astype(origin_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, context_len=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.astype(
            origin_dtype
        ), hidden_states.astype(origin_dtype)

        # 3. Feed-forward
        ffn_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ffn_output, c_gate_msa)
        hidden_states = hidden_states.astype(origin_dtype)
        return hidden_states


def _apply_rotary_emb(
    x: jax.Array, freqs_cos: jax.Array, freqs_sin: jax.Array, is_neox_style: bool = False
) -> jax.Array:
    pass


class WanT2VCrossAttention(nnx.Module):
    pass


class WanI2VCrossAttention(nnx.Module):
    pass


class WanTimeTextImageEmbedding(nnx.Module):
    pass


class WanTransformer3DModel:
    def __init__(self, config):
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_dim
        self.num_attention_heads = config.num_heads
        self.sp_size = 1
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        inner_dim = config.num_attention_heads * config.attention_head_dim

        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        # attn_backend = get_global_server_args().attention_backend
        transformer_block = WanTransformerBlock
        self.blocks = nnx.list(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                )
                for i in range(config.num_layers)
            ]
        )

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=jnp.float32,
        )
        pass

    def forward(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | list[jax.Array],
        timesteps: jax.Array,
        encoder_hidden_states_image: jax.Array | list[jax.Array] | None,
        guidance_scale=1.0 | None,
        **kwargs,
    ):
        # origin_dtype = hidden_states.dtype
        if isinstance(encoder_hidden_states, list):
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

        # freqs_cis = (
        #     (freqs_cos.astype(jnp.float32), freqs_sin.astype(jnp.float32))
        #     if freqs_cos is not None
        #     else None
        # )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timesteps.dim() == 2:
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
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))


EntryClass = WanTransformer3DModel
