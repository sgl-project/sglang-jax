import math

import jax
from jax.sharding import PartitionSpec as P

from sgl_jax.global_config import global_config


def should_scatter(dim_size: int, num_devices: int) -> bool:
    """Return True if a row-parallel output should be reduce-scattered on `dim`.

    Requires the per-device slice to be at least ``tpu_scatter_min_local_size``
    and the full dimension to divide evenly across devices (a hard requirement
    of ``psum_scatter(..., tiled=True)``).
    """
    if num_devices <= 1:
        return False
    return (
        dim_size >= num_devices * global_config.tpu_scatter_min_local_size
        and dim_size % num_devices == 0
    )


def prepare_scattered_spec_if_needed(
    out_specs: P,
    scatter_dim: int,
    scatter_axis: str,
    full_dim_size: int,
    mesh: jax.sharding.Mesh,
) -> tuple[P, bool]:
    """Stack ``scatter_axis`` onto ``out_specs[scatter_dim]`` if scatter fires.

    The decision uses the *local* shard size — ``full_dim_size`` divided by
    however many mesh axes already partition ``scatter_dim``.

    In the cases like DP attention, there's existing DP sharding on the sequences dimension. This
    will influence sequence parallel behavior

    Returns ``(new_out_specs, did_combine)``.
    """
    existing = out_specs[scatter_dim]
    if existing is None:
        existing_factor = 1
    elif isinstance(existing, tuple):
        existing_factor = math.prod(mesh.shape[a] for a in existing)
    else:
        existing_factor = mesh.shape[existing]

    if not should_scatter(full_dim_size // existing_factor, mesh.shape[scatter_axis]):
        return out_specs, False

    if existing is None:
        combined: tuple[str, ...] | str = scatter_axis
    elif isinstance(existing, tuple):
        combined = existing + (scatter_axis,)
    else:
        combined = (existing, scatter_axis)

    new_out_specs = P(*(combined if i == scatter_dim else axis for i, axis in enumerate(out_specs)))
    return new_out_specs, True
