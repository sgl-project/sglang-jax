"""Verify FP8 fused_moe kernel precision with optimized large-tile configs.

Compares fused_ep_moe (Pallas kernel) output vs ref_moe (pure JAX reference)
when both use the same FP8-quantized weights. This validates that the
scale-group Python for-loop optimization produces numerically correct results
with the larger bd1c/bfc tile sizes.

Usage:
    python -m benchmark.moe.verify_fp8_precision
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    ref_moe,
)
from sgl_jax.srt.layers.moe import TopK
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.test.test_utils import create_device_mesh


def gen_moe_inputs(dtype, top_k, num_experts, hidden_size, intermediate_size, num_tokens, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 9)
    k0, k1, k2, k3, _, _, _, k7, k8 = keys

    a = jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype) / 10
    w1 = (
        jax.random.normal(k1, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w2 = (
        jax.random.normal(k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w3 = (
        jax.random.normal(k3, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)

    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None], axis=1
    )
    gating_output = (gating_output + one_hot).astype(dtype)

    return a, w1, w2, w3, gating_output


def run_test(mesh, num_tokens, block_config, atol=2e-1, rtol=2e-1):
    dtype = jnp.bfloat16
    w_dtype = jnp.float8_e4m3fn
    top_k = 8
    num_experts = 256
    hidden_size = 4096
    intermediate_size = 2048
    subc_quant_wsz = 256

    a, w1, w2, w3, gating_output = gen_moe_inputs(
        dtype, top_k, num_experts, hidden_size, intermediate_size, num_tokens
    )

    # Quantize weights
    w1, w1_scale_3d = quantize_tensor(w_dtype, w1, axis=1, block_size=subc_quant_wsz)
    w3, w3_scale_3d = quantize_tensor(w_dtype, w3, axis=1, block_size=subc_quant_wsz)
    w2, w2_scale_3d = quantize_tensor(w_dtype, w2, axis=1, block_size=subc_quant_wsz)

    w1_scale = w1_scale_3d.reshape(
        w1_scale_3d.shape[0], w1_scale_3d.shape[1], 1, w1_scale_3d.shape[2]
    )
    w3_scale = w3_scale_3d.reshape(
        w3_scale_3d.shape[0], w3_scale_3d.shape[1], 1, w3_scale_3d.shape[2]
    )
    w2_scale = w2_scale_3d.reshape(
        w2_scale_3d.shape[0], w2_scale_3d.shape[1], 1, w2_scale_3d.shape[2]
    )

    topk_module = TopK(topk=top_k, renormalize=False)
    topk_weights, topk_ids = topk_module(gating_output)

    # Fused kernel
    actual = fused_ep_moe(
        mesh=mesh,
        tokens=a,
        w1=w1,
        w2=w2,
        w3=w3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=top_k,
        act_fn="silu",
        subc_quant_wsz=subc_quant_wsz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w3_scale=w3_scale,
        block_config=block_config,
        tp_axis_name="tensor",
    )

    # Reference implementation
    expected = ref_moe(
        a,
        w1,
        w2,
        w3,
        gating_output,
        top_k,
        act_fn="silu",
        subc_quant_wsz=subc_quant_wsz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w3_scale=w3_scale,
    )

    # Gather for multi-host
    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P(
                ("data", "tensor"),
            ),
        ),
        out_specs=P(),
        check_vma=False,
    )
    def _replicate(x):
        x = lax.all_gather(x, axis_name="tensor", axis=0, tiled=True)
        x = lax.all_gather(x, axis_name="data", axis=0, tiled=True)
        return x

    actual_np = np.asarray(jax.device_get(_replicate(actual)))
    expected_np = np.asarray(jax.device_get(_replicate(expected)))

    abs_diff = np.abs(actual_np - expected_np)
    max_abs_err = float(np.max(abs_diff))
    mean_abs_err = float(np.mean(abs_diff))

    # Relative error (avoid div by zero)
    denom = np.maximum(np.abs(expected_np), 1e-8)
    max_rel_err = float(np.max(abs_diff / denom))
    mean_rel_err = float(np.mean(abs_diff / denom))

    close = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)

    return close, max_abs_err, mean_abs_err, max_rel_err, mean_rel_err


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print(f"Local devices: {jax.local_device_count()}")
    print()

    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    # Test configs: (num_tokens, block_config_name, FusedMoEBlockConfig)
    # These are the tuned FP8 configs with large bd1c/bfc
    test_cases = [
        (
            64,
            "tuned-large-tile",
            FusedMoEBlockConfig(
                bt=4, bf=2048, bd1=4096, bd2=4096, btc=64, bfc=2048, bd1c=4096, bd2c=4096, bse=2048
            ),
        ),
        (
            256,
            "tuned-large-tile",
            FusedMoEBlockConfig(
                bt=16,
                bf=2048,
                bd1=4096,
                bd2=4096,
                btc=256,
                bfc=2048,
                bd1c=4096,
                bd2c=4096,
                bse=2048,
            ),
        ),
        (
            512,
            "tuned-large-tile",
            FusedMoEBlockConfig(
                bt=32,
                bf=2048,
                bd1=4096,
                bd2=4096,
                btc=512,
                bfc=2048,
                bd1c=4096,
                bd2c=4096,
                bse=2048,
            ),
        ),
        (
            1024,
            "tuned-large-tile",
            FusedMoEBlockConfig(
                bt=64,
                bf=2048,
                bd1=4096,
                bd2=4096,
                btc=512,
                bfc=2048,
                bd1c=4096,
                bd2c=4096,
                bse=2048,
            ),
        ),
        # Also test with old small-tile config for comparison
        (
            256,
            "old-small-tile",
            FusedMoEBlockConfig(
                bt=16, bf=2048, bd1=4096, bd2=4096, btc=256, bfc=256, bd1c=512, bd2c=512, bse=2048
            ),
        ),
    ]

    all_pass = True
    print(
        f"{'nt':>6}  {'config':<20}  {'pass':>4}  {'max_abs':>10}  {'mean_abs':>10}  {'max_rel':>10}  {'mean_rel':>10}"
    )
    print("-" * 90)

    for num_tokens, cfg_name, block_config in test_cases:
        try:
            ok, max_abs, mean_abs, max_rel, mean_rel = run_test(mesh, num_tokens, block_config)
            status = "OK" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(
                f"{num_tokens:>6}  {cfg_name:<20}  {status:>4}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {max_rel:>10.6f}  {mean_rel:>10.6f}"
            )
        except Exception as e:
            all_pass = False
            print(f"{num_tokens:>6}  {cfg_name:<20}  ERR   {e}")

    print()
    if all_pass:
        print("ALL PASSED: FP8 precision verified with large-tile configs.")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
