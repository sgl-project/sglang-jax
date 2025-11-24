from collections.abc import Sequence

import jax
import numpy as np
from jax._src import mesh_utils


def create_device_mesh(
    ici_parallelism: Sequence[int],
    dcn_parallelism: Sequence[int],
    devices=None,
    device_indexes: list[int] = None,
    num_slices: int = 1,
    allow_split_physical_axes: bool = True,
    use_explicit_sharding: bool = True,
) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    if device_indexes is not None:
        max_index = max(device_indexes)
        if max_index >= len(devices):
            raise RuntimeError("Device index out of range")
        devices_dict = {device.id: device for device in devices}
        devices = [devices_dict.get(i) for i in list(set(device_indexes))]

    ici_parallelism = fill_unspecified_parallelism(ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(dcn_parallelism, num_slices)
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        devices_array = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices=devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )

    if use_explicit_sharding:
        axis_types = (jax.sharding.AxisType.Explicit,) * len(mesh_axes)
        mesh = jax.sharding.Mesh(devices_array, mesh_axes, axis_types=axis_types)
    else:
        mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(parallelism: Sequence[int], num_devices: int) -> Sequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert (
        determined_val >= 1 and determined_val.is_integer
    ), "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism


mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
]
