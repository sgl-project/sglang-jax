import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import apply_rotary_emb
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
from sgl_jax.srt.multimodal.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)


class WanImageEmbedding(nnx.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(num_features=in_features, rngs=nnx.Rngs(0))
        self.ff = MLP(
            input_dim=in_features,
            mlp_hidden_dim=in_features,
            output_dim=out_features,
            act_type="gelu",
        )
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
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(num_features=dim, epsilon=epsilon, rngs=nnx.Rngs(0))

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
            self.norm_q = RMSNorm(dim_head, epsilon=epsilon)
            self.norm_k = RMSNorm(dim_head, epsilon=epsilon)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, epsilon=epsilon)
            self.norm_k = RMSNorm(dim, epsilon=epsilon)
        else:
            print("QK Norm type not supported")
            raise Exception
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
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                epsilon=epsilon,
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
        self.ffn = MLP(input_dim=dim, mlp_hidden_dim=ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nnx.Param(
            jax.random.normal(jax.random.key(0), (1, 6, dim)) / (dim**0.5)
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        freqs_cis: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_len, _ = hidden_states.shape
        origin_dtype = hidden_states.dtype
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
            e = self.scale_shift_table + temb.astype(jnp.float32)

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
        q, k = apply_rotary_emb(q, cos, sin, is_neox_style=False), apply_rotary_emb(
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


class WanSelfAttention(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        epsilon=1e-6,
        parallel_attention=False,
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
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
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
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
                v = self.to_v(context)[0].view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
            v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
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
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            epsilon,
        )

        self.add_k_proj = ReplicatedLinear(dim, dim)
        self.add_v_proj = ReplicatedLinear(dim, dim)
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
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
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
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
        self.text_embedder = MLP(
            input_dim=text_embed_dim,
            mlp_hidden_dim=dim,
            output_dim=dim,
            bias=True,
            act_type="gelu_pytorch_tanh",
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = nnx.data(
                WanImageEmbedding(in_features=image_embed_dim, out_features=dim)
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
    def __init__(self, config, *, rngs: nnx.Rngs = None):
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_dim
        self.num_attention_heads = config.num_heads
        self.in_channels = config.in_channels
        self.sp_size = 1
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        inner_dim = config.num_attention_heads * config.attention_head_dim

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
            out_features=out_channels * int(jnp.prod(jnp.array(config.patch_size))),
            use_bias=True,
            rngs=rngs,
        )
        self.scale_shift_table = nnx.Param(
            jax.random.normal(jax.random.key(0), (1, 2, inner_dim)) / (inner_dim**0.5)
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | list[jax.Array],
        timesteps: jax.Array,
        encoder_hidden_states_image: jax.Array | list[jax.Array] | None,
        guidance_scale=1.0,
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

        # Concatenate image and text embeddings if image embeddings exist
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = jnp.concatenate(
                [encoder_hidden_states_image, encoder_hidden_states], axis=1
            )

        # 4. Transformer blocks
        freqs_cis = (freqs_cos, freqs_sin)
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            combined = self.scale_shift_table[None, :, :, :] + temb[:, :, None, :]
            # Split into shift and scale
            shift, scale = jnp.split(combined, 2, axis=2)
            shift = jnp.squeeze(shift, axis=2)
            scale = jnp.squeeze(scale, axis=2)
        else:
            # batch_size, inner_dim
            combined = self.scale_shift_table + temb[:, None, :]
            shift, scale = jnp.split(combined, 2, axis=1)

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


EntryClass = WanTransformer3DModel
