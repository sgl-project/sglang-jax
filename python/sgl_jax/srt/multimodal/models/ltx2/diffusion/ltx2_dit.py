import logging
import math

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.dits.ltx2_model_config import LTX2ModelConfig
from sgl_jax.srt.multimodal.layers.attention.layer import USPAttention
from sgl_jax.srt.multimodal.layers.layernorm import FP32LayerNorm
from sgl_jax.srt.multimodal.layers.mlp import MLP
from sgl_jax.srt.multimodal.layers.rotary_embedding import NDRotaryEmbedding
from sgl_jax.srt.multimodal.layers.visual_embedding import TimestepEmbedder

logger = logging.getLogger(__name__)


def _apply_split_rotary_flat(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array:
    """Apply split rotary embeddings on flat [B, T, D] representation.

    Matches PyTorch's apply_split_rotary_emb exactly.
    """
    batch_size, seq_len, d = x.shape
    # Infer num_heads from cos shape [1, num_heads, seq_len, head_dim/2]
    num_heads = cos.shape[1]
    head_dim = d // num_heads

    # PyTorch LTX-2 uses [B, H, T, D_head] for rotation
    # x: [B, T, num_heads, head_dim]
    x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim)
    # swap to [B, num_heads, T, head_dim]
    x_reshaped = jnp.swapaxes(x_reshaped, 1, 2)

    # Split each head's dimension into two halves: [..., 2, head_dim/2]
    split_input = x_reshaped.reshape(*x_reshaped.shape[:-1], 2, head_dim // 2)

    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    # We assume cos/sin are passed as [1, num_heads, seq_len, head_dim/2]
    cos_exp = jnp.expand_dims(cos, -2)
    sin_exp = jnp.expand_dims(sin, -2)

    output = split_input * cos_exp

    first_half_output = output[..., :1, :] - sin_exp * second_half_input
    second_half_output = output[..., 1:, :] + sin_exp * first_half_input

    output = jnp.concatenate([first_half_output, second_half_output], axis=-2)
    output = output.reshape(*output.shape[:-2], head_dim)

    # Swap back to [B, T, num_heads, head_dim]
    output = jnp.swapaxes(output, 1, 2)
    # Flatten back to [B, T, D]
    return output.reshape(batch_size, seq_len, d)


def compute_video_pe(T: int, H: int, W: int, fps: float, theta: float, inner_dim: int, num_attention_heads: int):
    """Compute video positional embeddings in split RoPE format.

    Matches the PyTorch ltx_core reference implementation:
    - Uses geometric frequency spacing (theta^linspace * pi/2)
    - Causal fix on temporal axis
    - Midpoint coordinates for all axes
    - Fractional positions normalized by max_pos, scaled to [-1, 1]
    - Split format: cos/sin reshaped to [1, num_heads, seq_len, head_dim/2]

    Returns (freqs_cos, freqs_sin) each of shape [1, num_heads, seq_len, head_dim/2].
    """
    start = 1.0
    end = theta
    n_elem = 2 * 3  # 2 * max_pos_count (T, H, W)
    indices_len = inner_dim // n_elem

    lin = jnp.linspace(
        math.log(start) / math.log(theta),
        math.log(end) / math.log(theta),
        indices_len,
        dtype=jnp.float32,
    )

    indices = theta ** lin
    indices = indices * (math.pi / 2.0)

    max_pos = [20, 2048, 2048]
    scale_factors = [8, 32, 32]

    # Temporal coordinates with causal fix
    t_start = jnp.arange(T, dtype=jnp.float32) * scale_factors[0]
    t_end = t_start + scale_factors[0]
    shift = scale_factors[0] - 1
    t_start = jnp.maximum(t_start - shift, 0.0)
    t_end = jnp.maximum(t_end - shift, 0.0)
    t_start = t_start / fps
    t_end = t_end / fps
    t_mid = (t_start + t_end) / 2.0

    # Spatial coordinates with midpoints
    h_start = jnp.arange(H, dtype=jnp.float32) * scale_factors[1]
    h_end = h_start + scale_factors[1]
    h_mid = (h_start + h_end) / 2.0

    w_start = jnp.arange(W, dtype=jnp.float32) * scale_factors[2]
    w_end = w_start + scale_factors[2]
    w_mid = (w_start + w_end) / 2.0

    grid_t, grid_h, grid_w = jnp.meshgrid(t_mid, h_mid, w_mid, indexing='ij')

    # Fractional positions normalized by max_pos, scaled to [-1, 1]
    frac_t = grid_t / max_pos[0]
    frac_h = grid_h / max_pos[1]
    frac_w = grid_w / max_pos[2]

    fractional_positions = jnp.stack([frac_t, frac_h, frac_w], axis=-1)
    num_tokens = T * H * W
    fractional_positions = fractional_positions.reshape(num_tokens, 3)

    frac_scaled = (fractional_positions * 2.0 - 1.0)[..., None]
    indices_unsqueezed = indices[None, ...]

    freqs = frac_scaled * indices_unsqueezed  # [num_tokens, 3, 682]
    # Transpose to interleave axes: [num_tokens, 682, 3] so that after
    # flattening each head gets a MIX of T/H/W frequencies at different
    # frequency bands, matching PyTorch's .transpose(-1,-2).flatten(2).
    freqs = jnp.swapaxes(freqs, -1, -2)
    freqs = freqs.reshape(num_tokens, -1)  # [num_tokens, 2046]

    # Split format: cos/sin without repeat_interleave, reshaped per head
    cos_freq = jnp.cos(freqs)
    sin_freq = jnp.sin(freqs)

    # Pad to inner_dim // 2 (split format pads to half the total dim)
    expected_freqs = inner_dim // 2  # 4096 // 2 = 2048
    pad_size = expected_freqs - cos_freq.shape[-1]  # 2048 - 2046 = 2
    if pad_size > 0:
        cos_padding = jnp.ones((num_tokens, pad_size), dtype=jnp.float32)
        sin_padding = jnp.zeros((num_tokens, pad_size), dtype=jnp.float32)
        cos_freq = jnp.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = jnp.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape to [1, num_heads, num_tokens, head_dim/2]
    cos_freq = cos_freq.reshape(1, num_tokens, num_attention_heads, -1)
    cos_freq = jnp.swapaxes(cos_freq, 1, 2)
    sin_freq = sin_freq.reshape(1, num_tokens, num_attention_heads, -1)
    sin_freq = jnp.swapaxes(sin_freq, 1, 2)

    return cos_freq, sin_freq


def compute_1d_pe(midpoints, theta, inner_dim, num_heads, max_pos=20):
    """Compute 1D temporal RoPE in split format.

    Used for audio self-attention PE and audio-video cross-attention PE.

    Args:
        midpoints: [seq_len] temporal midpoints in seconds
        theta: RoPE theta (e.g. 10000.0)
        inner_dim: total dimension (num_heads * head_dim)
        num_heads: number of attention heads for split reshape
        max_pos: maximum position in seconds for normalization

    Returns:
        (cos, sin) each of shape [1, num_heads, seq_len, head_dim/2]
    """
    n_elem = 2  # 1D: 2 * num_spatial_dims
    indices_len = inner_dim // n_elem

    lin = jnp.linspace(
        math.log(1.0) / math.log(theta),
        1.0,
        indices_len,
        dtype=jnp.float32,
    )
    indices = theta ** lin * (math.pi / 2.0)

    frac = midpoints / max_pos
    frac_scaled = (frac * 2.0 - 1.0)[:, None]  # [seq_len, 1]

    freqs = frac_scaled * indices[None, :]  # [seq_len, indices_len]

    # Split format: cos/sin without repeat_interleave, reshaped per head
    cos_freq = jnp.cos(freqs)
    sin_freq = jnp.sin(freqs)

    # Pad to inner_dim // 2
    expected_freqs = inner_dim // 2
    pad_size = expected_freqs - cos_freq.shape[-1]
    if pad_size > 0:
        cos_padding = jnp.ones((midpoints.shape[0], pad_size), dtype=jnp.float32)
        sin_padding = jnp.zeros((midpoints.shape[0], pad_size), dtype=jnp.float32)
        cos_freq = jnp.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = jnp.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape to [1, num_heads, seq_len, head_dim/2]
    seq_len = midpoints.shape[0]
    cos_freq = cos_freq.reshape(1, seq_len, num_heads, -1)
    cos_freq = jnp.swapaxes(cos_freq, 1, 2)
    sin_freq = sin_freq.reshape(1, seq_len, num_heads, -1)
    sin_freq = jnp.swapaxes(sin_freq, 1, 2)

    return cos_freq, sin_freq


def _apply_interleaved_rotary_flat(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply interleaved rotary embeddings on flat [B, T, D] representation.

    Matches PyTorch's apply_interleaved_rotary_emb exactly.
    Pairs adjacent elements: (x0, x1), (x2, x3), ...
    cos, sin: [1, T, D] (already repeat_interleaved from freq computation).
    """
    x_paired = x.reshape(*x.shape[:-1], -1, 2)  # [..., D/2, 2]
    x0, x1 = x_paired[..., 0], x_paired[..., 1]
    x_rot = jnp.stack([-x1, x0], axis=-1)  # [..., D/2, 2]
    x_rot = x_rot.reshape(x.shape)
    return x * cos + x_rot * sin


def _audio_temporal_midpoints(T_audio):
    """Audio temporal midpoints in seconds from causal audio patchifier.

    Matches PyTorch AudioPatchifier.get_patch_grid_bounds with causal=True.
    """
    downsample_factor = 4
    hop_length = 160
    sample_rate = 16000
    causal_offset = 1

    mel_frames = jnp.arange(T_audio, dtype=jnp.float32) * downsample_factor
    start_mel = jnp.maximum(mel_frames + causal_offset - downsample_factor, 0.0)
    end_mel = mel_frames + causal_offset

    t_start = start_mel * hop_length / sample_rate
    t_end = end_mel * hop_length / sample_rate
    return (t_start + t_end) / 2.0


def _video_temporal_midpoints(T, fps=25.0):
    """Video temporal midpoints in seconds (same computation as compute_video_pe)."""
    scale_factor = 8
    t_start = jnp.arange(T, dtype=jnp.float32) * scale_factor
    t_end = t_start + scale_factor
    shift = scale_factor - 1
    t_start = jnp.maximum(t_start - shift, 0.0)
    t_end = jnp.maximum(t_end - shift, 0.0)
    t_start = t_start / fps
    t_end = t_end / fps
    return (t_start + t_end) / 2.0


class LTX2Attention(nnx.Module):
    """LTX-2 Attention module with RoPE support."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        context_dim: int | None = None,
        epsilon: float = 1e-6,
        apply_gated_attention: bool = False,
        rope_type: str = "split",
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        context_dim = context_dim or query_dim
        self.apply_gated_attention = apply_gated_attention
        self.rope_type = rope_type
        self.mesh = mesh

        self.to_q = LinearBase(
            input_size=query_dim,
            output_size=inner_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_k = LinearBase(
            input_size=context_dim,
            output_size=inner_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_v = LinearBase(
            input_size=context_dim,
            output_size=inner_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.to_out = LinearBase(
            input_size=inner_dim,
            output_size=query_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

        self.norm_q = RMSNorm(inner_dim, epsilon=epsilon)
        self.norm_k = RMSNorm(inner_dim, epsilon=epsilon)

        if apply_gated_attention:
            # Per-head gating: Linear(query_dim -> heads), matching reference
            # Uses 2*sigmoid so zero-init gives identity (2*0.5=1.0)
            self.to_gate_logits = LinearBase(
                input_size=query_dim,
                output_size=heads,
                use_bias=True,
                mesh=mesh,
                kernel_axes=(None, None),
            )

        self.attn = USPAttention(
            num_heads=heads,
            head_size=dim_head,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | None = None,
        mask: jax.Array | None = None,
        pe: jax.Array | None = None,
        k_pe: jax.Array | None = None,
        req=None,
    ) -> jax.Array:
        """
        Args:
            x: Query tensor [B, L1, C]
            context: Context tensor [B, L2, C] (if None, self-attention)
            mask: Attention mask
            pe: Positional embeddings (cos, sin) each [T, inner_dim] interleaved
            k_pe: Positional embeddings for keys
        """
        context = context if context is not None else x
        b, n, d = x.shape[0], self.heads, self.dim_head

        # Compute Q/K/V in flat representation [B, T, inner_dim]
        q_flat = self.norm_q(self.to_q(x)[0])        # [B, T, inner_dim=4096]
        k_flat = self.norm_k(self.to_k(context)[0])  # [B, T, inner_dim=4096]
        v = self.to_v(context)[0].reshape(b, -1, n, d)  # [B, T, H, D]

        # Apply rotary embeddings on FLAT representation (before per-head reshape)
        # This matches PyTorch which applies RoPE on [B, T, D] so each head
        # gets different rotation frequencies.
        if pe is not None:
            cos, sin = pe
            if self.rope_type == "interleaved":
                # Interleaved: cos/sin are [1, T, D], pairs adjacent elements
                q_flat = _apply_interleaved_rotary_flat(q_flat, cos, sin)
                k_pe_to_use = k_pe if k_pe is not None else pe
                cos_k, sin_k = k_pe_to_use
                k_flat = _apply_interleaved_rotary_flat(k_flat, cos_k, sin_k)
            else:
                # Split: cos/sin are [1, H, T, D_head/2]
                q_flat = _apply_split_rotary_flat(q_flat, cos, sin, self.mesh)
                k_pe_to_use = k_pe if k_pe is not None else pe
                cos_k, sin_k = k_pe_to_use
                k_flat = _apply_split_rotary_flat(k_flat, cos_k, sin_k, self.mesh)

        # Reshape to per-head AFTER RoPE
        q = q_flat.reshape(b, -1, n, d)
        k = k_flat.reshape(b, -1, n, d)

        # Compute attention
        attn_output = self.attn(q, k, v, req=req, mask=mask)

        # Flatten: (B, seq, heads, head_dim) -> (B, seq, heads*head_dim)
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)

        # Per-head gating BEFORE to_out (matching reference attention.py)
        if self.apply_gated_attention:
            gate_logits, _ = self.to_gate_logits(x)  # (B, T, H) from input x
            gates = 2.0 * jax.nn.sigmoid(gate_logits)  # 2*sigmoid for identity at zero-init
            b_g, t_g, _ = attn_output.shape
            attn_output = attn_output.reshape(b_g, t_g, self.heads, self.dim_head)
            attn_output = attn_output * gates[..., None]  # (B, T, H, D) * (B, T, H, 1)
            attn_output = attn_output.reshape(b_g, t_g, self.heads * self.dim_head)

        attn_output, _ = self.to_out(attn_output)

        return attn_output


class LTX2TransformerBlock(nnx.Module):
    """
    LTX-2 Transformer Block supporting both video-only and audio-video modes.

    Structure:
    1. Self-attention with adaptive layer norm
    2. Cross-attention with text embeddings
    3. Audio-Video cross-attention (optional, only in audio-video mode)
    4. Feed-forward network
    """

    def __init__(
        self,
        idx: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        context_dim: int,
        epsilon: float = 1e-6,
        audio_dim: int | None = None,
        audio_num_heads: int | None = None,
        audio_context_dim: int | None = None,
        apply_gated_attention: bool = False,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.num_heads = num_heads

        # Video components
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            context_dim=None,
            epsilon=epsilon,
            apply_gated_attention=apply_gated_attention,
            mesh=mesh,
        )

        self.attn2 = LTX2Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            context_dim=context_dim,
            epsilon=epsilon,
            apply_gated_attention=apply_gated_attention,
            mesh=mesh,
        )

        self.ff = MLP(
            input_dim=dim,
            mlp_hidden_dim=ffn_dim,
            output_dim=dim,
            act_type="gelu_pytorch_tanh",
            mesh=mesh,
        )

        self.scale_shift_table = nnx.Param(
            jax.random.normal(jax.random.key(0), (6, dim)) / (dim**0.5)
        )

        # Audio components (optional)
        self.has_audio = audio_dim is not None
        if self.has_audio:
            self.audio_attn1 = LTX2Attention(
                query_dim=audio_dim,
                heads=audio_num_heads,
                dim_head=audio_dim // audio_num_heads,
                context_dim=None,
                epsilon=epsilon,
                apply_gated_attention=apply_gated_attention,
                mesh=mesh,
            )

            self.audio_attn2 = LTX2Attention(
                query_dim=audio_dim,
                heads=audio_num_heads,
                dim_head=audio_dim // audio_num_heads,
                context_dim=audio_context_dim,
                epsilon=epsilon,
                apply_gated_attention=apply_gated_attention,
                mesh=mesh,
            )

            self.audio_ff = MLP(
                input_dim=audio_dim,
                mlp_hidden_dim=audio_dim * 4,  # Audio uses its own 4x expansion (2048*4=8192)
                output_dim=audio_dim,
                act_type="gelu_pytorch_tanh",
                mesh=mesh,
            )

            self.audio_scale_shift_table = nnx.Param(
                jax.random.normal(jax.random.key(0), (6, audio_dim)) / (audio_dim**0.5)
            )

            # Audio-Video cross attention
            self.audio_to_video_attn = LTX2Attention(
                query_dim=dim,
                heads=audio_num_heads,
                dim_head=audio_dim // audio_num_heads,
                context_dim=audio_dim,
                epsilon=epsilon,
                apply_gated_attention=apply_gated_attention,
                mesh=mesh,
            )

            self.video_to_audio_attn = LTX2Attention(
                query_dim=audio_dim,
                heads=audio_num_heads,
                dim_head=audio_dim // audio_num_heads,
                context_dim=dim,
                epsilon=epsilon,
                apply_gated_attention=apply_gated_attention,
                mesh=mesh,
            )

            self.scale_shift_table_a2v_ca_audio = nnx.Param(
                jax.random.normal(jax.random.key(0), (5, audio_dim)) / (audio_dim**0.5)
            )
            self.scale_shift_table_a2v_ca_video = nnx.Param(
                jax.random.normal(jax.random.key(0), (5, dim)) / (dim**0.5)
            )

        self.epsilon = epsilon
        self.mesh = mesh

    def __call__(
        self,
        video_args: dict | None,
        audio_args: dict | None,
        req=None,
    ) -> tuple[dict | None, dict | None]:
        """
        Forward pass for LTX-2 transformer block.

        Args:
            video_args: Dict with keys: x, timesteps, context, context_mask, pe
            audio_args: Dict with keys: x, timesteps, context, context_mask, pe (optional)
        """
        if video_args is None and audio_args is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video_args["x"] if video_args is not None else None
        ax = audio_args["x"] if audio_args is not None else None

        # Video self-attention
        if vx is not None:
            # Get adaptive norm parameters from timesteps
            vshift_msa, vscale_msa, vgate_msa = self._get_ada_values(
                self.scale_shift_table, video_args["timesteps"], slice(0, 3)
            )

            # Self-attention with adaptive norm
            norm_vx = self._rms_norm(vx) * (1 + vscale_msa) + vshift_msa

            # STG (Spatio-Temporal Guidance): zero out self-attention for perturbed batch elements
            v_mask = 1.0
            if "stg_mask" in video_args and self.idx in video_args.get("stg_blocks", []):
                v_mask = video_args["stg_mask"]

            vx = vx + self.attn1(norm_vx, pe=video_args.get("pe"), req=req) * vgate_msa * v_mask

            # Cross-attention with text
            cross_attn_out = self.attn2(
                self._rms_norm(vx),
                context=video_args["context"],
                mask=video_args.get("context_mask"),
                req=req,
            )
            vx = vx + cross_attn_out

        # Audio self-attention
        if ax is not None and self.has_audio:
            ashift_msa, ascale_msa, agate_msa = self._get_ada_values(
                self.audio_scale_shift_table, audio_args["timesteps"], slice(0, 3)
            )

            norm_ax = self._rms_norm(ax) * (1 + ascale_msa) + ashift_msa

            # STG: zero out audio self-attention for perturbed batch elements
            # (same mask and blocks as video self-attention)
            a_mask = 1.0
            if "stg_mask" in video_args and self.idx in video_args.get("stg_blocks", []):
                a_mask = video_args["stg_mask"]

            ax = ax + self.audio_attn1(norm_ax, pe=audio_args.get("pe"), req=req) * agate_msa * a_mask

            ax = ax + self.audio_attn2(
                self._rms_norm(ax),
                context=audio_args["context"],
                mask=audio_args.get("context_mask"),
                req=req,
            )

        # Audio-Video cross attention
        if vx is not None and ax is not None and self.has_audio:
            vx_norm = self._rms_norm(vx)
            ax_norm = self._rms_norm(ax)

            # Modality mask: zeros out A2V/V2A cross-attention for the
            # isolated-modality batch element (4th forward pass).
            # Shape [B, 1, 1] like stg_mask, defaults to 1.0 (no masking).
            mod_mask = video_args.get("modality_mask", 1.0)

            # Get adaptive norm parameters for A2V cross attention
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_v2a,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                audio_args["cross_scale_shift_timestep"],
                audio_args["cross_gate_timestep"],
            )

            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_a2v,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                video_args["cross_scale_shift_timestep"],
                video_args["cross_gate_timestep"],
            )

            # Audio to Video (masked by modality_mask)
            vx_scaled = vx_norm * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
            ax_scaled = ax_norm * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
            vx = vx + (
                self.audio_to_video_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=video_args.get("cross_pe"),
                    k_pe=audio_args.get("cross_pe"),
                    req=req,
                )
                * gate_a2v
                * mod_mask
            )

            # Video to Audio (masked by modality_mask)
            ax_scaled = ax_norm * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
            vx_scaled = vx_norm * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
            ax = ax + (
                self.video_to_audio_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=audio_args.get("cross_pe"),
                    k_pe=video_args.get("cross_pe"),
                    req=req,
                )
                * gate_v2a
                * mod_mask
            )

        # Feed-forward
        if vx is not None:
            vshift_mlp, vscale_mlp, vgate_mlp = self._get_ada_values(
                self.scale_shift_table, video_args["timesteps"], slice(3, None)
            )
            vx_scaled = self._rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        if ax is not None and self.has_audio:
            ashift_mlp, ascale_mlp, agate_mlp = self._get_ada_values(
                self.audio_scale_shift_table, audio_args["timesteps"], slice(3, None)
            )
            ax_scaled = self._rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        # Update dictionaries
        if video_args is not None:
            video_args = {**video_args, "x": vx}
        if audio_args is not None:
            audio_args = {**audio_args, "x": ax}

        return video_args, audio_args

    def _rms_norm(self, x: jax.Array) -> jax.Array:
        """Apply RMS normalization."""
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.epsilon)

    def _get_ada_values(
        self, scale_shift_table: jax.Array, timestep: jax.Array, indices: slice
    ) -> tuple[jax.Array, ...]:
        """Get adaptive normalization values."""
        batch_size = timestep.shape[0]
        num_ada_params = scale_shift_table.shape[0]
        inner_dim = timestep.shape[-1] // num_ada_params
        new_shape = (batch_size, timestep.shape[1], num_ada_params, inner_dim)

        if self.mesh is not None:
            # With Explicit mesh, reshape of a sharded dim requires out_sharding to be specified.
            # timestep: [B, 1, 6*inner_dim@tensor] → [B, 1, 6, inner_dim@tensor]
            from jax.sharding import NamedSharding, PartitionSpec as P
            timestep_reshaped = jax.lax.reshape(
                timestep,
                new_shape,
                out_sharding=NamedSharding(self.mesh, P(None, None, None, "tensor")),
            )
        else:
            timestep_reshaped = timestep.reshape(new_shape)

        ada_values = scale_shift_table[indices][None, None, :, :] + timestep_reshaped[:, :, indices, :]

        return tuple(ada_values[:, :, i, :] for i in range(ada_values.shape[2]))

    def _get_av_ca_ada_values(
        self,
        scale_shift_table: jax.Array,
        scale_shift_timestep: jax.Array,
        gate_timestep: jax.Array,
        num_scale_shift_values: int = 4,
    ) -> tuple[jax.Array, ...]:
        """Get audio-video cross attention adaptive normalization values."""
        scale_shift_ada = self._get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada = self._get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            gate_timestep,
            slice(None, None),
        )
        return (*scale_shift_ada, *gate_ada)


class LTX2Transformer3DModel(nnx.Module):
    """
    LTX-2 3D Transformer Model for audio-video generation.

    This is a JAX/Flax implementation following the sglang-jax wan pattern.
    """

    def __init__(
        self,
        config: LTX2ModelConfig,
        *,
        dtype=jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh
        rngs = nnx.Rngs(0)

        # Video components
        if config.is_video_enabled:
            self.inner_dim = config.num_attention_heads * config.attention_head_dim
            self.patch_size = (1, 1, 1)  # Linear patchify (no spatial aggregation)

            # Linear patchify projection: [B, T*H*W, C] → [B, T*H*W, inner_dim]
            # Matches PyTorch's nn.Linear(in_channels, inner_dim)
            self.patch_embedding = nnx.Linear(
                in_features=config.in_channels,
                out_features=self.inner_dim,
                use_bias=True,
                rngs=rngs,
            )

            self.adaln_single = TimestepEmbedder(
                self.inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=self.mesh,
                embedding_coefficient=6,
            )

            # Text projection: 2-layer MLP with GELU tanh (PixArtAlphaTextProjection)
            self.caption_projection = MLP(
                input_dim=config.caption_channels,
                mlp_hidden_dim=self.inner_dim,
                output_dim=self.inner_dim,
                act_type="gelu_pytorch_tanh",
                mesh=mesh,
            )

            # Output layers
            self.scale_shift_table = nnx.Param(
                jax.random.normal(jax.random.key(0), (2, self.inner_dim)) / (self.inner_dim**0.5)
            )
            self.norm_out = FP32LayerNorm(
                num_features=self.inner_dim,
                epsilon=config.epsilon,
                use_scale=False,
                use_bias=False,
                rngs=rngs,
            )
            self.proj_out = nnx.Linear(
                in_features=self.inner_dim,
                out_features=config.out_channels,  # No patch expansion (linear patchify)
                use_bias=True,
                rngs=rngs,
            )

            # Rotary embeddings for video
            d = self.inner_dim // config.num_attention_heads
            self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
            self.rotary_emb = NDRotaryEmbedding(
                rope_dim_list=self.rope_dim_list,
                rope_theta=config.positional_embedding_theta,
                dtype=jnp.float32,
            )

        # Audio components
        if config.is_audio_enabled:
            self.audio_inner_dim = config.audio_num_attention_heads * config.audio_attention_head_dim

            self.audio_patch_embedding = nnx.Linear(
                in_features=config.audio_in_channels,
                out_features=self.audio_inner_dim,
                use_bias=True,
                rngs=rngs,
            )

            self.audio_adaln_single = TimestepEmbedder(
                self.audio_inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=mesh,
                embedding_coefficient=6,  # 3 for self-attn + 3 for FFN
            )

            self.audio_caption_projection = MLP(
                input_dim=config.caption_channels,
                mlp_hidden_dim=self.audio_inner_dim,
                output_dim=self.audio_inner_dim,
                act_type="gelu_pytorch_tanh",
                mesh=mesh,
            )

            # Audio output layers
            self.audio_scale_shift_table = nnx.Param(
                jax.random.normal(jax.random.key(0), (2, self.audio_inner_dim))
                / (self.audio_inner_dim**0.5)
            )
            self.audio_norm_out = FP32LayerNorm(
                num_features=self.audio_inner_dim,
                epsilon=config.epsilon,
                use_scale=False,
                use_bias=False,
                rngs=rngs,
            )
            # Audio is 1D, no spatial patching needed
            self.audio_proj_out = nnx.Linear(
                in_features=self.audio_inner_dim,
                out_features=config.audio_out_channels,
                use_bias=True,
                rngs=rngs,
            )

        # Audio-video cross attention components
        if config.is_video_enabled and config.is_audio_enabled:
            self.av_ca_video_scale_shift_adaln = TimestepEmbedder(
                self.inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=mesh,
                embedding_coefficient=4,
            )
            self.av_ca_a2v_gate_adaln = TimestepEmbedder(
                self.inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=mesh,
                embedding_coefficient=1,
            )
            self.av_ca_audio_scale_shift_adaln = TimestepEmbedder(
                self.audio_inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=mesh,
                embedding_coefficient=4,
            )
            self.av_ca_v2a_gate_adaln = TimestepEmbedder(
                self.audio_inner_dim,
                frequency_embedding_size=256,
                act_layer="silu",
                mesh=mesh,
                embedding_coefficient=1,
            )

        # Transformer blocks
        self.blocks = nnx.List(
            [
                LTX2TransformerBlock(
                    idx=i,
                    dim=self.inner_dim if config.is_video_enabled else 0,
                    ffn_dim=config.hidden_size * 4,  # Standard 4x expansion
                    num_heads=config.num_attention_heads,
                    context_dim=config.cross_attention_dim,
                    epsilon=config.epsilon,
                    audio_dim=self.audio_inner_dim if config.is_audio_enabled else None,
                    audio_num_heads=config.audio_num_attention_heads if config.is_audio_enabled else None,
                    audio_context_dim=config.audio_cross_attention_dim if config.is_audio_enabled else None,
                    apply_gated_attention=config.apply_gated_attention,
                    mesh=mesh,
                )
                for i in range(config.num_layers)
            ]
        )

    def __call__(
        self,
        hidden_states: jax.Array | None = None,
        encoder_hidden_states: jax.Array | None = None,
        timesteps: jax.Array | None = None,
        video_latent: jax.Array | None = None,
        audio_latent: jax.Array | None = None,
        video_context: jax.Array | None = None,
        audio_context: jax.Array | None = None,
        video_positions: jax.Array | None = None,
        audio_positions: jax.Array | None = None,
        video_context_mask: jax.Array | None = None,
        audio_context_mask: jax.Array | None = None,
        video_pe: tuple | None = None,
        stg_mask: jax.Array | None = None,
        stg_blocks: list[int] | None = None,
        modality_mask: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        req=None,
        **kwargs,
    ):
        """
        Forward pass for LTX-2 transformer.

        Accepts both the native LTX-2 interface (video_latent, video_context) and the
        DiffusionModelRunner interface (hidden_states, encoder_hidden_states) for
        seamless integration with the existing pipeline.

        Args:
            modality_mask: [B, 1, 1] mask that zeros out A2V/V2A cross-attention
                for the isolated-modality batch element. Used for modality guidance
                (4th forward pass). Default None means all cross-attention is active.
        """
        # Bridge: map generic runner params to LTX-2 native params
        if video_latent is None and hidden_states is not None:
            video_latent = hidden_states
        if video_context is None and encoder_hidden_states is not None:
            video_context = encoder_hidden_states
        if video_context_mask is None and attention_mask is not None:
            video_context_mask = attention_mask

        # Auto-compute video PE if not provided
        if video_pe is None and video_latent is not None:
            b, c, t, h, w = video_latent.shape
            fps = getattr(self.config, "fps", 25.0)
            inner_dim = self.config.attention_head_dim * self.config.num_attention_heads
            video_pe = compute_video_pe(
                t, h, w, fps, self.config.positional_embedding_theta,
                inner_dim, self.config.num_attention_heads,
            )

        # Default STG config when batch has CFG+STG (batch_size=3 or 4)
        if video_latent is not None and video_latent.shape[0] >= 3:
            if stg_mask is None:
                stg_mask = jnp.array([[[1.]], [[1.]], [[0.]]])
                if video_latent.shape[0] == 4:
                    # 4th element (modality-isolated) keeps self-attention active
                    stg_mask = jnp.array([[[1.]], [[1.]], [[0.]], [[1.]]])
            if stg_blocks is None:
                stg_blocks = (29,)

        video_args = None
        audio_args = None

        # Prepare video
        if video_latent is not None and self.config.is_video_enabled:
            batch_size, _, T, H, W = video_latent.shape

            # Linear patchify: flatten to [B, T*H*W, C] then project to inner_dim
            video_latent_flat = jnp.transpose(video_latent, (0, 2, 3, 4, 1))  # [B, T, H, W, C]
            video_latent_flat = video_latent_flat.reshape(batch_size, T * H * W, -1)  # [B, T*H*W, C]
            vx = self.patch_embedding(video_latent_flat)  # [B, T*H*W, inner_dim]

            # Timestep embedding: returns (adaln_cond [B, 6*dim], embedded_ts [B, dim])
            v_adaln_cond, v_embedded_ts = self.adaln_single(
                timesteps.flatten() * self.config.timestep_scale_multiplier
            )
            v_adaln_cond = v_adaln_cond.reshape(batch_size, 1, v_adaln_cond.shape[-1])

            # Context projection (2-layer MLP with GELU)
            v_context = self.caption_projection(video_context)

            # Positional embeddings: use pre-computed PE if provided, else compute from grid
            if video_pe is not None:
                freqs_cos, freqs_sin = video_pe
            else:
                freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                    video_positions, shard_dim=0, start_frame=0
                )

            video_args = {
                "x": vx,
                "timesteps": v_adaln_cond,  # [B, 1, 6*inner_dim] for AdaLN modulation
                "embedded_timestep": v_embedded_ts,  # [B, inner_dim] for output scale-shift
                "context": v_context,
                "context_mask": video_context_mask,
                "pe": (freqs_cos, freqs_sin),
                "grid_size": (T, H, W),  # Full latent resolution (no spatial reduction)
            }
            if stg_mask is not None and stg_blocks is not None:
                video_args["stg_mask"] = stg_mask
                video_args["stg_blocks"] = stg_blocks
            if modality_mask is not None:
                video_args["modality_mask"] = modality_mask

            # Add cross attention timesteps and cross PE if audio is enabled
            if self.config.is_audio_enabled:
                v_cross_ss_adaln, _ = self.av_ca_video_scale_shift_adaln(
                    timesteps.flatten() * self.config.timestep_scale_multiplier
                )
                v_cross_gate_adaln, _ = self.av_ca_a2v_gate_adaln(
                    timesteps.flatten() * self.config.av_ca_timestep_scale_multiplier
                )
                video_args["cross_scale_shift_timestep"] = v_cross_ss_adaln.reshape(
                    batch_size, 1, v_cross_ss_adaln.shape[-1]
                )
                video_args["cross_gate_timestep"] = v_cross_gate_adaln.reshape(
                    batch_size, 1, v_cross_gate_adaln.shape[-1]
                )

                # Video cross PE: 1D temporal midpoints expanded to T*H*W tokens
                # Uses video's num_attention_heads with audio_cross_attention_dim
                fps = getattr(self.config, "fps", 25.0)
                video_t_mid = _video_temporal_midpoints(T, fps)
                video_cross_midpoints = jnp.repeat(video_t_mid, H * W)
                video_cross_pe = compute_1d_pe(
                    video_cross_midpoints, self.config.positional_embedding_theta,
                    self.config.audio_cross_attention_dim,
                    self.config.num_attention_heads,
                )
                video_args["cross_pe"] = video_cross_pe

        # Prepare audio
        if audio_latent is not None and self.config.is_audio_enabled:
            ax = self.audio_patch_embedding(audio_latent)
            batch_size = ax.shape[0]

            # Timestep embedding: returns (adaln_cond, embedded_ts) tuple
            a_adaln_cond, a_embedded_ts = self.audio_adaln_single(
                timesteps.flatten() * self.config.timestep_scale_multiplier
            )
            a_adaln_cond = a_adaln_cond.reshape(batch_size, 1, a_adaln_cond.shape[-1])

            # Context projection (2-layer MLP with GELU)
            a_context = self.audio_caption_projection(audio_context)

            # Audio 1D temporal RoPE
            T_audio = audio_latent.shape[1]
            audio_midpoints = _audio_temporal_midpoints(T_audio)
            audio_pe = compute_1d_pe(
                audio_midpoints, self.config.positional_embedding_theta,
                self.audio_inner_dim, self.config.audio_num_attention_heads,
            )
            # Audio cross PE is identical to self-attention PE (same inner_dim, num_heads)
            audio_cross_pe = audio_pe

            audio_args = {
                "x": ax,
                "timesteps": a_adaln_cond,
                "embedded_timestep": a_embedded_ts,
                "context": a_context,
                "context_mask": audio_context_mask,
                "pe": audio_pe,
                "cross_pe": audio_cross_pe,
            }

            # Add cross attention timesteps if video is enabled
            if self.config.is_video_enabled:
                a_cross_ss_adaln, _ = self.av_ca_audio_scale_shift_adaln(
                    timesteps.flatten() * self.config.timestep_scale_multiplier
                )
                a_cross_gate_adaln, _ = self.av_ca_v2a_gate_adaln(
                    timesteps.flatten() * self.config.av_ca_timestep_scale_multiplier
                )
                audio_args["cross_scale_shift_timestep"] = a_cross_ss_adaln.reshape(
                    batch_size, 1, a_cross_ss_adaln.shape[-1]
                )
                audio_args["cross_gate_timestep"] = a_cross_gate_adaln.reshape(
                    batch_size, 1, a_cross_gate_adaln.shape[-1]
                )

        # Process through transformer blocks
        for block in self.blocks:
            video_args, audio_args = block(video_args, audio_args, req)

        # Process outputs
        vx_out = None
        ax_out = None

        if video_args is not None:
            vx = video_args["x"]
            v_embedded_ts = video_args["embedded_timestep"]  # [B, inner_dim]
            T, H, W = video_args["grid_size"]

            # Output scale-shift: scale_shift_table[2, dim] + embedded_timestep[B, 1, dim]
            # Matches PyTorch: scale_shift_table[None, None] + embedded_timestep[:, :, None]
            # embedded_timestep shape: [B, 1, dim], scale_shift_table: [2, dim]
            embedded_ts_expanded = v_embedded_ts.reshape(vx.shape[0], 1, 1, self.inner_dim)
            scale_shift = (
                self.scale_shift_table.value[None, None, :, :]  # [1, 1, 2, dim]
                + embedded_ts_expanded                           # [B, 1, 1, dim] broadcast → [B, 1, 2, dim]
            )
            shift = scale_shift[:, :, 0, :]  # [B, 1, dim]
            scale = scale_shift[:, :, 1, :]  # [B, 1, dim]

            vx = self.norm_out(vx.astype(jnp.float32))
            vx = vx * (1 + scale) + shift
            vx = self.proj_out(vx)  # [B, T*H*W, out_channels=128]

            # Reshape to channel-first: [B, 128, T, H, W]
            batch_size = vx.shape[0]
            vx = vx.reshape(batch_size, T, H, W, self.config.out_channels)
            vx_out = jnp.transpose(vx, (0, 4, 1, 2, 3))  # [B, C, T, H, W]

        if audio_args is not None:
            ax = audio_args["x"]
            a_embedded_ts = audio_args["embedded_timestep"]

            embedded_ts_expanded = a_embedded_ts.reshape(ax.shape[0], 1, 1, self.audio_inner_dim)
            scale_shift = (
                self.audio_scale_shift_table.value[None, None, :, :]
                + embedded_ts_expanded
            )
            shift = scale_shift[:, :, 0, :]
            scale = scale_shift[:, :, 1, :]

            ax = self.audio_norm_out(ax.astype(jnp.float32))
            ax = ax * (1 + scale) + shift
            ax = self.audio_proj_out(ax)
            ax_out = ax

        # Return (video, audio) tuple when audio is generated, else just video
        if ax_out is not None:
            return vx_out, ax_out
        return vx_out

    def load_weights(self, model_config) -> None:
        from jax import ShapeDtypeStruct

        # Helper to materialize abstract params into concrete zero arrays
        def _replace_abstract(x):
            if isinstance(x, ShapeDtypeStruct):
                from jax.sharding import NamedSharding, PartitionSpec
                pspec = PartitionSpec()
                if hasattr(x, "sharding") and x.sharding is not None and hasattr(x.sharding, "spec"):
                    pspec = x.sharding.spec
                mesh = getattr(self, "mesh", None)
                if mesh:
                    concrete_sharding = NamedSharding(mesh, pspec)
                    return jax.device_put(jnp.zeros(x.shape, x.dtype), concrete_sharding)
                return jnp.zeros(x.shape, x.dtype)
            return x

        if getattr(model_config, "_dummy_mode", False):
            state = nnx.state(self)
            concrete_state = jax.tree_util.tree_map(_replace_abstract, state)
            nnx.update(self, concrete_state)
            return

        from sgl_jax.srt.utils.weight_utils import WeightLoader
        from .ltx2_dit_weights_mapping import to_ltx2_video_only_mappings, to_ltx2_audio_video_mappings

        # Use only the main LTX-2 checkpoint file (not FP4/FP8/distilled variants)
        from sgl_jax.srt.multimodal.models.ltx2.utils import get_ltx2_checkpoint_dir, cleanup_ltx2_checkpoint_dir
        _ckpt_dir = get_ltx2_checkpoint_dir()
        if _ckpt_dir:
            model_config.model_path = _ckpt_dir

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        # Use audio-video mappings when audio is enabled, video-only otherwise
        if self.config.is_audio_enabled:
            raw_mappings = to_ltx2_audio_video_mappings(self.config.num_layers)
        else:
            raw_mappings = to_ltx2_video_only_mappings(self.config.num_layers)
        # Checkpoint keys have 'model.diffusion_model.' prefix; add it to mapping keys
        prefixed_mappings = {
            f"model.diffusion_model.{k}": v for k, v in raw_mappings.items()
        }
        loader.load_weights_from_safetensors(prefixed_mappings)
        cleanup_ltx2_checkpoint_dir(_ckpt_dir)

        # Materialize any remaining abstract params (from nnx.eval_shape) to concrete zeros
        state = nnx.state(self)
        concrete_state = jax.tree_util.tree_map(_replace_abstract, state)
        nnx.update(self, concrete_state)

    @staticmethod
    def get_config_class() -> type[LTX2ModelConfig]:
        return LTX2ModelConfig


EntryClass = LTX2Transformer3DModel
