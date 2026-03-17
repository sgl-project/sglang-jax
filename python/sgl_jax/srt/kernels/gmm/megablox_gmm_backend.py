import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import gmm_v2 as gmm_v2_kernel


def gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] | None = None,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    acc_dtype: jnp.dtype | None = None,
    activation_quantized_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Dispatch GMM to v2.

    Args:
        lhs: [m, k] input matrix.
        rhs: [num_groups, k, n] weight matrix.
        group_sizes: [num_groups] group sizes.
        preferred_element_type: Output dtype.
        rhs_scale: Optional [num_groups, num_blocks, 1, n] scale.
        rhs_bias: Optional [num_groups, 1, n] bias.
        tiling: Unused. Kept for API compatibility.
        group_offset: Optional group offset for sharding.
        existing_out: Unused. Kept for API compatibility.
        interpret: Unused. Kept for API compatibility.
        maybe_quantize_lhs: If True, gmm_v2 will quantize lhs when rhs is
            quantized. Set to False to keep activations unquantized.
        zero_initialize: Whether to initialize unvisited output elements to zero.
        acc_dtype: Optional accumulation dtype for gmm_v2 (e.g. jnp.float32).
        activation_quantized_dtype: Unused. Kept for API compatibility.
    """
    del tiling, existing_out, interpret, activation_quantized_dtype

    return gmm_v2_kernel(
        lhs,
        rhs,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        preferred_element_type=preferred_element_type,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
        acc_dtype=acc_dtype,
    )
