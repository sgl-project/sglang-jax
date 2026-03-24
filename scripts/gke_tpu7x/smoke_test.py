#!/usr/bin/env python3
"""Smoke test for TPU v7x-8: verifies JAX distributed, sharding, and FusedEPMoE.

ALL processes run the SAME code — required by JAX for sharded computations.
Use conditional logging (not conditional computation) to control output.

Usage:
    python3 -u /tmp/launcher.py scripts/gke_tpu7x/smoke_test.py
"""
import jax
import jax.numpy as jnp

proc = jax.process_index()
is_main = proc == 0


def log(msg):
    if is_main:
        print(msg, flush=True)


log(f"JAX {jax.__version__}, {jax.device_count()} devices, local {jax.local_device_count()}")

# --- Test 1: simple jit matmul (single device) ---
log("\n--- Test 1: simple jit matmul ---")


@jax.jit
def matmul(a, b):
    return a @ b


a = jnp.ones((128, 256), dtype=jnp.bfloat16)
b = jnp.ones((256, 128), dtype=jnp.bfloat16)
c = matmul(a, b)
c.block_until_ready()
log(f"  Result shape: {c.shape}, sum: {float(c.sum())}")

# --- Test 2: sharded computation across all devices ---
log("\n--- Test 2: sharded across {0} devices ---".format(jax.device_count()))
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

devices = jax.devices()
mesh = Mesh(devices, axis_names=("x",))
sharding = NamedSharding(mesh, P("x"))

n = jax.device_count()
x = jax.device_put(jnp.arange(n * 1024, dtype=jnp.float32).reshape(n, 1024), sharding)
y = jax.jit(lambda x: x * 2)(x)
y.block_until_ready()
log(f"  Sharded result shape: {y.shape}, first: {float(y[0, 0])}, last: {float(y[-1, -1])}")

# --- Test 3: FusedEPMoE forward pass ---
log("\n--- Test 3: FusedEPMoE forward (ep_size=4) ---")
from flax import nnx

from sgl_jax.srt.layers.moe import FusedEPMoE, TopK
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

moe_mesh = create_device_mesh(
    ici_parallelism=[1, 4],
    dcn_parallelism=[1, 1],
    devices=devices[:4],
    mesh_axes=("data", "tensor"),
)
with jax.set_mesh(moe_mesh):
    moe = FusedEPMoE(
        hidden_size=2048,
        num_experts=4,
        num_experts_per_tok=2,
        ep_size=4,
        mesh=moe_mesh,
        intermediate_dim=512,
        weight_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        activation="silu",
        layer_id=0,
    )
    topk_mod = TopK(topk=2, renormalize=True, layer_id=0)

    tokens = jnp.ones((16, 2048), dtype=jnp.bfloat16)
    tokens = jax.device_put(tokens, NamedSharding(moe_mesh, P("tensor", None)))
    logits = jnp.ones((16, 4), dtype=jnp.bfloat16)
    logits = jax.device_put(logits, NamedSharding(moe_mesh, P("tensor", None)))

    topk_weights, topk_ids = topk_mod(logits)
    out = moe(tokens, topk_weights, topk_ids)
    out.block_until_ready()
    log(f"  MoE output shape: {out.shape}")

log("\n=== ALL TESTS PASSED ===")
