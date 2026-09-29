"""Shared helpers for HCA kernels."""

import jax.numpy as jnp


def searchsorted_right(table, values):
    """Right insertion indices in a sorted request table, using compare-and-count."""
    table = jnp.asarray(table)
    values = jnp.asarray(values)
    return jnp.sum(
        (table.reshape((1,) * values.ndim + (-1,)) <= values[..., None]).astype(jnp.int32),
        axis=-1,
    )
