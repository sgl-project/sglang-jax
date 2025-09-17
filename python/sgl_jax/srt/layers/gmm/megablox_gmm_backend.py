import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.gmm.megablox_gmm_kernel.gmm import gmm as gmm_kernel

gmm = jax.custom_vjp(gmm_kernel, nondiff_argnums=(3, 4, 7, 8, 10))


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    vmem_limit_bytes: int = 64 * (1 << 20),
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
]:
    out = gmm_kernel(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])


def _gmm_bwd(
    group_sizes,
    preferred_element_type,
    tiling,
    group_offset,
    existing_out,
    transpose_rhs,
    interpret,
    gmm_grad,
):
    # implement the backward pass
    return (
        gmm_grad,
        None,
        None,
        None,
        None,
        None,
    )


gmm.defvjp(_gmm_fwd, _gmm_bwd)
