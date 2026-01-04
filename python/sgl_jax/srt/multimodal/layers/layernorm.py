import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.layernorm import RMSNorm


# adapted from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
# NOTE(will): Needed to match behavior of diffusers and wan2.1 even while using
# FSDP's MixedPrecisionPolicy
class FP32LayerNorm(nnx.LayerNorm):
    def __call__(self, inputs: jax.Array) -> jax.Array:
        origin_dtype = inputs.dtype
        x = inputs.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        if self.scale is not None:
            x = x * self.scale.value.astype(jnp.float32)

        if self.bias is not None:
            x = x + self.bias.value.astype(jnp.float32)

        return x.astype(origin_dtype)


class ScaleResidual(nnx.Module):
    """
    Applies gated residual connection.
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward(self, residual: jax.Array, x: jax.Array, gate: jax.Array) -> jax.Array:
        """Apply gated residual connection."""
        # x.shape: [batch_size, seq_len, inner_dim]
        if gate.dim() == 4:
            # gate.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            return residual + (x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(
                1, 2
            )
        else:
            # gate.shape: [batch_size, 1, inner_dim]
            return residual + x * gate


class ScaleResidualLayerNormScaleShift(nnx.Module):
    """
    Fused operation that combines:
    1. Gated residual connection
    2. LayerNorm
    3. Scale and shift operations

    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        epsilon: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype | None = None,
        prefix: str = "",
    ):
        if norm_type == "rms":
            self.norm = RMSNorm(
                hidden_size,
                use_scale=elementwise_affine,
                epsilon=epsilon,
                dtype=dtype,
            )
        elif norm_type == "layer":
            if compute_dtype == jnp.float32:
                self.norm = FP32LayerNorm(
                    hidden_size,
                    epsilon=epsilon,
                    use_scale=elementwise_affine,
                    use_bias=elementwise_affine,
                    dtype=dtype,
                )
            else:
                self.norm = nnx.LayerNorm(
                    hidden_size,
                    epsilon=epsilon,
                    use_scale=elementwise_affine,
                    use_bias=elementwise_affine,
                    dtype=dtype,
                )
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def __call__(
        self,
        residual: jax.Array,
        x: jax.Array,
        gate: jax.Array | int,
        shift: jax.Array,
        scale: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Apply gated residual connection, followed by layernorm and
        scale/shift in a single fused operation.

        Returns:
            Tuple containing:
            - normalized and modulated output of shape: [batch_size, seq_len, inner_dim]
            - residual value (value after residual connection
              but before normalization)
        """
        # x.shape: [batch_size, seq_len, inner_dim]
        # Apply residual connection with gating
        if isinstance(gate, int):
            # used by cross-attention, should be 1
            assert gate == 1
            residual_output = residual + x
        elif isinstance(gate, jax.Array):
            if gate.ndim == 4:
                # gate.shape: [batch_size, num_frames, 1, inner_dim]
                b, s, d = x.shape
                num_frames = gate.shape[1]
                assert s % num_frames == 0, "seq_len must be divisible by num_frames for 4D gate"
                frame_seqlen = s // num_frames

                # Reshape x to match gate's structure for broadcasting
                # x -> [batch, num_frames, frame_seqlen, inner_dim]
                x_reshaped = x.reshape((b, num_frames, frame_seqlen, d))

                # Apply gate and flatten back
                gated_x = (x_reshaped * gate).reshape((b, s, d))
                residual_output = residual + gated_x
            else:
                # used by bidirectional self attention
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        # residual_output.shape: [batch_size, seq_len, inner_dim]

        # Apply normalization
        normalized = self.norm(residual_output)

        # Apply scale and shift
        # Handle 4D scale/shift for multimodal cases: [B, F, 1, C]
        if scale.ndim == 4:
            b, seq_len, c = normalized.shape
            num_frames = scale.shape[1]
            assert (
                seq_len % num_frames == 0
            ), "seq_len must be divisible by num_frames for 4D scale/shift"
            frame_seqlen = seq_len // num_frames

            # Reshape normalized to [B, F, frame_seqlen, C] to match scale/shift [B, F, 1, C]
            normalized_reshaped = normalized.reshape((b, num_frames, frame_seqlen, c))

            modulated_reshaped = normalized_reshaped * (1 + scale) + shift
            modulated = modulated_reshaped.reshape((b, seq_len, c))
        else:
            # Standard broadcasting (e.g. [B, 1, C] or scalar)
            modulated = normalized * (1 + scale) + shift

        return modulated, residual_output
