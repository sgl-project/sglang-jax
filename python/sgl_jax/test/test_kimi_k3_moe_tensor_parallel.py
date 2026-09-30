"""K3's fp4 MoE must compute the same result whatever the MoE tensor axis size.

wo shards its contraction dim K over "tensor". Its per-32 block scales have to be split the same
way: if they are replicated, each rank pairs its K slice with the whole scale array, the kernel
derives block_size = local_K // num_blocks, and every block scale lands on the wrong rows.
Runs on two host devices; the single-device result is the reference.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from sgl_jax.srt.models.kimi_k3 import KimiK3EPMoE  # noqa: E402

E, H, INTER = 2, 64, 128
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def _fp4(rng, shape):
    mag = E2M1[rng.integers(0, 8, shape)] * rng.choice([-1.0, 1.0], shape)
    return jnp.asarray(mag).astype(jnp.float4_e2m1fn)


def _run(ndev):
    mesh = Mesh(
        np.array(jax.devices()[:ndev]).reshape(1, ndev),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    rng = np.random.default_rng(0)
    with jax.set_mesh(mesh):
        moe = KimiK3EPMoE(
            hidden_size=H,
            num_experts=E,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=INTER,
            dtype=jnp.float32,
            fp4=True,
        )
        with jax.sharding.use_abstract_mesh(moe.updated_mesh):
            for name, shape in (
                ("wi_0", (E, H, INTER)),
                ("wi_1", (E, H, INTER)),
                ("wo", (E, INTER, H)),
            ):
                param = getattr(moe, name)
                param.value = jax.device_put(_fp4(rng, shape), param.value.sharding)
            for name in ("wi_0_scale", "wi_1_scale", "wo_scale"):
                param = getattr(moe, name)
                scale = (2.0 ** rng.integers(-3, 2, param.value.shape)).astype(np.float32)
                param.value = jax.device_put(jnp.asarray(scale), param.value.sharding)
        x = jnp.asarray(rng.standard_normal((8, H)), jnp.float32)
        ids = jnp.asarray(rng.integers(0, E, (8, 1)), jnp.int32)
        return np.asarray(moe(x, jnp.ones((8, 1), jnp.float32), ids)), moe.tp_size


def test_fp4_moe_output_does_not_depend_on_the_tensor_axis():
    if jax.device_count() < 2:
        pytest.skip("needs two host devices (XLA_FLAGS set after jax was initialized)")
    ref, tp1 = _run(1)
    got, tp2 = _run(2)
    assert (tp1, tp2) == (1, 2)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-5, f"tensor-parallel fp4 MoE diverges from single-device: rel L2 {rel:.3e}"
