"""Shared ViT sharding utilities for in-model VLM encoders.

Provides a single-source-of-truth :class:`VisionShardSpecs` that derives every
``PartitionSpec`` / ``NamedSharding`` from a mesh and a TP toggle.  Both
replicated and tensor-parallel modes are covered; no axis literal appears at a
call site.
"""

from dataclasses import dataclass

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def apply_data_sharding(x: jax.Array, mesh: Mesh, spec: PartitionSpec) -> jax.Array:
    """Reshard or constrain *x* to *spec* under *mesh*."""
    sharding = NamedSharding(mesh, spec)
    if "data" in mesh.abstract_mesh.explicit_axes:
        return jax.sharding.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def resolve_encoder_tp(mesh: Mesh | None, mode: str) -> bool:
    if mode != "tp" or mesh is None:
        return False
    return "tensor" in mesh.shape and int(mesh.shape["tensor"]) > 1


def encode_lane_count(mesh: Mesh | None, encoder_tp: bool) -> int:
    if encoder_tp or mesh is None or "tensor" not in mesh.shape:
        return 1
    return int(mesh.shape["tensor"])


@dataclass(frozen=True)
class VisionShardSpecs:
    """Single source of truth for ViT sharding, parameterized by TP mode.

    Replicated (``tp=False``): weights are replicated, the batch is sharded
    over every device (``("data", "tensor")``).

    Tensor-parallel (``tp=True``): weights are column/row-parallel over
    ``"tensor"``, the batch is sharded over ``"data"`` only, and attention
    heads are split across the tensor axis.

    Biases, norms, and the patch-embed conv stay replicated in both modes.
    """

    mesh: Mesh | None
    tp: bool

    # -- batch spec ----------------------------------------------------------

    @property
    def batch_axis(self):
        if self.tp:
            return "data"
        if self.mesh is not None and "tensor" in self.mesh.axis_names:
            return ("data", "tensor")
        return "data"

    @property
    def head_axis(self):
        return "tensor" if self.tp else None

    # -- kernel axes ---------------------------------------------------------

    @property
    def col_kernel_axes(self):
        """Column-parallel: split output axis."""
        return (None, "tensor") if self.tp else (None, None)

    @property
    def row_kernel_axes(self):
        """Row-parallel: split input axis (all-reduce on output)."""
        return ("tensor", None) if self.tp else (None, None)

    # -- PartitionSpec builders ----------------------------------------------

    def batch_spec(self, *tail) -> PartitionSpec:
        return PartitionSpec(self.batch_axis, *tail)

    # -- NamedSharding builders (None when mesh is absent) -------------------

    def batch_sharding(self, *tail) -> NamedSharding | None:
        if self.mesh is None:
            return None
        return NamedSharding(self.mesh, self.batch_spec(*tail))

    def col_out(self, ndim: int) -> NamedSharding | None:
        """Output sharding for a column-parallel linear."""
        if self.mesh is None:
            return None
        if self.tp:
            spec = PartitionSpec("data", *([None] * (ndim - 2)), "tensor")
        else:
            spec = PartitionSpec(self.batch_axis, *([None] * (ndim - 1)))
        return NamedSharding(self.mesh, spec)

    def row_out(self, ndim: int) -> NamedSharding | None:
        """Output sharding for a row-parallel linear."""
        if self.mesh is None:
            return None
        if self.tp:
            spec = PartitionSpec("data", *([None] * (ndim - 1)))
        else:
            spec = PartitionSpec(self.batch_axis, *([None] * (ndim - 1)))
        return NamedSharding(self.mesh, spec)

    def qkv_reshape_sharding(self) -> NamedSharding | None:
        """Sharding for ``[B, T, heads, head_dim]`` reshape.

        In TP the fused QKV column-split lands on the heads axis.
        """
        if self.mesh is None:
            return None
        if self.tp:
            spec = PartitionSpec("data", None, "tensor", None)
        else:
            spec = PartitionSpec(self.batch_axis, None, None, None)
        return NamedSharding(self.mesh, spec)
