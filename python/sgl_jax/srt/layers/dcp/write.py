"""Device-side DCP write remapping (jax). Host math stays in ``layout.py``."""

from __future__ import annotations

import jax.numpy as jnp


def physical_write_loc_jax(
    virtual_loc: jnp.ndarray,
    dcp_size: int,
    dcp_rank: int | jnp.ndarray,
    interleave: int = 1,
) -> jnp.ndarray:
    """JAX twin of ``layout.physical_write_loc``. Identity when ``dcp_size==1``."""
    loc = virtual_loc.astype(jnp.int32)
    if dcp_size <= 1:
        return loc
    if interleave < 1:
        raise ValueError(f"interleave must be >= 1, got {interleave}")
    padded = loc < 0
    safe = jnp.where(padded, 0, loc)
    if interleave == 1:
        own, phys = safe % dcp_size, safe // dcp_size
    else:
        blk = safe // interleave
        own = blk % dcp_size
        phys = (blk // dcp_size) * interleave + (safe % interleave)
    drop = padded | (own != dcp_rank)
    return jnp.where(drop, jnp.int32(-1), phys.astype(jnp.int32))


def owned_len_jax(
    upto: jnp.ndarray,
    dcp_size: int,
    dcp_rank: int | jnp.ndarray,
    interleave: int = 1,
) -> jnp.ndarray:
    """Count of virtual tokens in ``[0, upto)`` owned by ``dcp_rank``.

    JAX twin of ``layout.get_dcp_lens(upto, ...)`` with ``start=0``.
    """
    x = upto.astype(jnp.int32)
    if dcp_size <= 1:
        return x
    if interleave == 1:
        return x // dcp_size + (dcp_rank < (x % dcp_size)).astype(jnp.int32)
    full, rem = x // interleave, x % interleave
    # numerator >= 0 because dcp_rank <= dcp_size - 1
    whole = (full - dcp_rank + dcp_size - 1) // dcp_size
    tail = jnp.where((full % dcp_size) == dcp_rank, rem, 0)
    return (whole * interleave + tail).astype(jnp.int32)


def physical_positions_jax(
    positions: jnp.ndarray,
    dcp_size: int,
    dcp_rank: int | jnp.ndarray,
    interleave: int = 1,
) -> jnp.ndarray:
    """Map virtual query positions to the causal bound in **physical** slot space.

    ``virtual_index`` is monotone in the physical slot, so this rank's owned tokens
    appear in virtual order at physical slots ``0, 1, 2, ...``. Hence
    "virtual key position <= ``positions``" is exactly "physical slot <
    ``owned_len(positions + 1)``", and the attend kernels' existing
    ``kp <= positions`` mask becomes correct once we hand them
    ``owned_len(positions + 1) - 1``.

    A rank owning nothing up to ``positions`` gets ``-1``: no key passes, the row's
    LSE comes back ``-inf``, and the merge drops it.
    """
    p = positions.astype(jnp.int32)
    if dcp_size <= 1:
        return p
    return owned_len_jax(p + 1, dcp_size, dcp_rank, interleave) - 1
