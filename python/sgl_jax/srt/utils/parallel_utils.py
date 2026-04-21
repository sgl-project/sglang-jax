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
