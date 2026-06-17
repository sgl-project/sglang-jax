"""CPU-only regression test for #<ISSUE_NUM>: EPLB dispatch under explicit-axis Mesh.

Run: XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu pytest <this>
"""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=16")

import jax  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from sgl_jax.srt.eplb.expert_location import (  # noqa: E402
    ExpertLocationMetadata,
    _topk_ids_logical_to_physical_static,
)

NUM_LAYERS, NUM_EXPERTS, EP = 8, 64, 16


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < EP:
        pytest.skip(f"need {EP} devices, have {jax.device_count()}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:EP]).reshape(1, EP),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def _make_metadata(mesh):
    rng = np.random.default_rng(0)
    p2l = np.stack([rng.permutation(NUM_EXPERTS) for _ in range(NUM_LAYERS)])
    l2r = np.argsort(p2l, axis=1)  # inverse permutation
    l2p = l2r[..., None]
    l2p_nv = np.ones((NUM_LAYERS, NUM_EXPERTS), np.int32)
    with jax.set_mesh(mesh):
        md = ExpertLocationMetadata(
            ep_dispatch_algorithm="static",
            logical_to_rank_dispatch_physical_map=l2r,
            logical_to_all_physical_map=l2p,
            logical_to_all_physical_map_num_valid=l2p_nv,
            physical_to_logical_map=p2l,
            num_physical_experts=NUM_EXPERTS,
        )
    rep = NamedSharding(mesh, P())
    for a in (
        "logical_to_rank_dispatch_physical_map",
        "logical_to_all_physical_map",
        "logical_to_all_physical_map_num_valid",
        "physical_to_logical_map",
    ):
        setattr(md, a, jax.device_put(getattr(md, a), rep))
    return md, l2r


def test_static_dispatch_traces_under_explicit_mesh(mesh):
    md, l2r = _make_metadata(mesh)
    idx_np = np.random.default_rng(1).integers(0, NUM_EXPERTS, (256, 4), np.int32)
    idx = jax.device_put(idx_np, NamedSharding(mesh, P("data", None)))

    @jax.jit
    def f(m, i):
        return _topk_ids_logical_to_physical_static(i, m, layer_id=3)

    with jax.set_mesh(mesh):
        out = f(md, idx)
    np.testing.assert_array_equal(np.asarray(out), l2r[3, idx_np])
    assert out.sharding.spec == P("data", None)
