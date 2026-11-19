"""Grouped expert matmul vs dense path on random fp8+rowscale experts (8 virtual CPU devices, 2x4 mesh); also checks the idle-row mask
(tokens with w==0 must contribute nothing) and several chunk sizes. Prints max|diff| (expect ~1e-7, NOT exactly 0 unless identical order)."""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_MOE"] = "grouped"
sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
import ml_dtypes
import model_ref as J
import numpy as np
import sharding as SH
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

E, D, F, K, N = 16, 64, 48, 6, 37
from types import SimpleNamespace

a = SimpleNamespace(
    dim=D,
    n_routed_experts=E,
    n_activated_experts=K,
    route_scale=1.5,
    swiglu_limit=10.0,
    score_func="sqrtsoftplus",
    norm_topk_prob=True,
    gate_temp=1.0,
)
rng = np.random.default_rng(0)


def f8(shape):
    return jnp.asarray((rng.standard_normal(shape) * 0.5).astype(ml_dtypes.float8_e4m3fn))


Pm = {
    "gate.weight": jnp.asarray(rng.standard_normal((E, D)).astype(np.float32)),
    "gate.bias": jnp.zeros((E,), jnp.float32),
    "experts.w1": f8((E, F, D)),
    "experts.w3": f8((E, F, D)),
    "experts.w2": f8((E, D, F)),
    "experts.w1_rowscale": jnp.asarray(rng.uniform(0.5, 2, (E, F)).astype(np.float32)),
    "experts.w3_rowscale": jnp.asarray(rng.uniform(0.5, 2, (E, F)).astype(np.float32)),
    "experts.w2_rowscale": jnp.asarray(rng.uniform(0.5, 2, (E, D)).astype(np.float32)),
    "shared_experts.w1.weight": jnp.asarray(rng.standard_normal((F, D)).astype(np.float32) * 0.1),
    "shared_experts.w2.weight": jnp.asarray(rng.standard_normal((D, F)).astype(np.float32) * 0.1),
    "shared_experts.w3.weight": jnp.asarray(rng.standard_normal((F, D)).astype(np.float32) * 0.1),
}
x = jnp.asarray(rng.standard_normal((1, N, D)).astype(np.float32))
mask = jnp.asarray(rng.random(N) > 0.3)  # some idle tokens
mesh = SH.make_mesh(jax.devices(), 2, 4)
Pm_s = {
    k: jax.device_put(v, NamedSharding(mesh, SH.param_spec("layers.0.ffn." + k)))
    for k, v in Pm.items()
}
J.MOE_MESH = None
dense = np.asarray(jax.jit(lambda p, x, m: J.moe(p, x, a, m))(Pm_s, x, mask))
for C in (128, 4, 1):
    J.MOE_MESH = mesh
    J.MOE_CHUNK = C
    got = np.asarray(jax.jit(lambda p, x, m: J.moe(p, x, a, m))(Pm_s, x, mask))
    d = np.abs(got - dense)
    print(
        f"chunk {C:>3}: max|d|={d.max():.3e} rel={d.max()/np.abs(dense).max():.3e}   idle tokens zero in both: {bool(np.allclose(got[0][~np.asarray(mask)], dense[0][~np.asarray(mask)]))}"
    )
# sanity that the test can fail: corrupt one expert's scale in the grouped path only
Pb = dict(Pm_s)
Pb["experts.w2_rowscale"] = Pm_s["experts.w2_rowscale"].at[3].multiply(1.5)
J.MOE_MESH = mesh
J.MOE_CHUNK = 128
bad = np.asarray(jax.jit(lambda p, x, m: J.moe(p, x, a, m))(Pb, x, mask))
print(f"control (expert 3 corrupted): max|d|={np.abs(bad - dense).max():.3e}  (must be >> 1e-6)")
