from dataclasses import dataclass

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def apply_data_sharding(x: jax.Array, mesh: Mesh, spec: PartitionSpec) -> jax.Array:
    sharding = NamedSharding(mesh, spec)
    if "data" in mesh.explicit_axes:
        return jax.sharding.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def resolve_encoder_tp(mesh: Mesh | None, mode: str) -> bool:
    if mode != "tp" or mesh is None:
        return False
    return "tensor" in mesh.shape and int(mesh.shape["tensor"]) > 1


@dataclass(frozen=True)
class VisionShardSpecs:
    """Vision-encoder sharding built from two axes.

    ``batch_axis`` shards the leading dim; ``tensor_axis`` is ``"tensor"`` under
    tensor parallelism and ``None`` otherwise. Call sites spell out the layout
    directly via :meth:`sharding`, e.g. a column-parallel output is
    ``sharding(batch_axis, None, tensor_axis)``. JAX pads a spec shorter than the
    array rank with replicated trailing dims, so batch-only sharding is simply
    ``sharding(batch_axis)`` regardless of ndim.
    """

    mesh: Mesh | None
    tp: bool

    @property
    def batch_axis(self) -> str | tuple[str, str]:
        if self.tp:
            return "data"
        if self.mesh is not None and "tensor" in self.mesh.axis_names:
            return ("data", "tensor")
        return "data"

    @property
    def tensor_axis(self) -> str | None:
        return "tensor" if self.tp else None

    @property
    def col_kernel_axes(self) -> tuple[None, str | None]:
        return (None, self.tensor_axis)

    @property
    def row_kernel_axes(self) -> tuple[str | None, None]:
        return (self.tensor_axis, None)

    def sharding(self, *spec: str | tuple[str, ...] | None) -> NamedSharding | None:
        if self.mesh is None:
            return None
        return NamedSharding(self.mesh, PartitionSpec(*spec))
