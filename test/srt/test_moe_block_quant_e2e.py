# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


def get_cosine_similarity(a, b, mesh: Mesh):
    """Cosine similarity in fp32, used as the bf16/quantised numerical-correctness metric.

    Why not allclose: at production-scale dimensions with bf16 / FP8 matmuls,
    element-wise tolerances are either too strict (false positives) or too
    loose to discriminate. Cosine similarity measures directional agreement
    and is invariant to per-element magnitude noise, which is the failure
    mode bf16 matmul chains actually exhibit.

    Threshold convention in this repo: assert >= 0.99. FlashInfer uses the
    same approach in their absorbed-MLA decode kernel tests
    (flashinfer-ai/flashinfer#551).
    """
    a_flat = jax.sharding.reshard(a.flatten().astype(jnp.float32), NamedSharding(mesh, P()))
    b_flat = jax.sharding.reshard(b.flatten().astype(jnp.float32), NamedSharding(mesh, P()))
    return jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat))


def _create_test_mesh():
    num_devices = len(jax.devices())
    ep_size = num_devices
    tp_size = 1
    devices = np.array(jax.devices()).reshape(ep_size, tp_size)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _make_quant_config(weight_dtype, weight_block_size):
    class BlockQuantConfig:
        def get_moe_weight_dtype(self):
            return weight_dtype

        def get_moe_activation_dtype(self):
            return None

        @property
        def weight_block_size(self):
            return None if weight_block_size is None else list(weight_block_size)

    return BlockQuantConfig()


def _quantize_moe_weight(weight, weight_dtype, weight_block_size, scale_format):
    # Weight layout is [E, k, n], quantize along k-dim (axis=1)
    if scale_format == "per_channel":
        return quantize_tensor(weight_dtype, weight, axis=1)

    block_size_out, block_size_k = weight_block_size

    if scale_format == "block_channel":
        return quantize_tensor(weight_dtype, weight, axis=1, block_size=block_size_k)

    if scale_format == "block_quant":
        w_q, scale = quantize_tensor(
            weight_dtype,
            weight,
            axis=(1, 2),
            block_size=(block_size_k, block_size_out),
        )
        # quantize_tensor produces (E, k_blocks, out_blocks) following axis order,
        # but offline checkpoints store scales as (E, out_blocks, k_blocks).
        scale = jnp.transpose(scale, (0, 2, 1))
        return w_q, scale

    raise ValueError(f"Unsupported scale_format={scale_format}")


def _get_scale_shardings(scale_format):
    if scale_format == "per_channel":
        return (
            P("expert", "tensor"),
            P("expert", "tensor"),
            P("expert", None),
        )

    if scale_format == "block_channel":
        return (
            P("expert", None, "tensor"),
            P("expert", None, "tensor"),
            P("expert", None, None),
        )

    if scale_format == "block_quant":
        return (
            P("expert", None, "tensor"),
            P("expert", None, "tensor"),
            P("expert", None, None),
        )

    raise ValueError(f"Unsupported scale_format={scale_format}")


@pytest.mark.parametrize(
    ("scale_format", "weight_block_size"),
    [
        ("per_channel", None),
        ("block_channel", (1, 128)),
        ("block_quant", (128, 128)),
    ],
    ids=["per-channel", "block-channel", "block-quant"],
)
def test_epmoe_block_quant_accuracy(scale_format, weight_block_size):
    print(f"\n>>> Running E2E EP MoE block quantization case: {scale_format} <<<")

    mesh = _create_test_mesh()

    hidden_size = 512
    intermediate_dim = 1024
    num_experts = 4 * len(jax.devices())
    num_experts_per_tok = 1
    compute_dtype = jnp.bfloat16
    weight_dtype = jnp.int8

    with jax.set_mesh(mesh):
        moe_ref = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=len(jax.devices()),
            mesh=mesh,
            intermediate_dim=intermediate_dim,
            quantization_config=None,
        )
        moe_quant = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=len(jax.devices()),
            mesh=mesh,
            intermediate_dim=intermediate_dim,
            quantization_config=_make_quant_config(weight_dtype, weight_block_size),
        )

    key = jax.random.PRNGKey(42)
    k_wi0, k_wi1, k_wo, k_x, k_topk = jax.random.split(key, 5)

    # Weight layout: [E, k, n]
    # wi_0/wi_1: [E, hidden_size, intermediate_dim], wo: [E, intermediate_dim, hidden_size]
    w_wi0_fp = jax.random.normal(
        k_wi0,
        (num_experts, hidden_size, intermediate_dim),
        dtype=compute_dtype,
    )
    w_wi1_fp = jax.random.normal(
        k_wi1,
        (num_experts, hidden_size, intermediate_dim),
        dtype=compute_dtype,
    )
    w_wo_fp = jax.random.normal(
        k_wo,
        (num_experts, intermediate_dim, hidden_size),
        dtype=compute_dtype,
    )

    wi0_q, wi0_scale = _quantize_moe_weight(
        w_wi0_fp,
        weight_dtype,
        weight_block_size,
        scale_format,
    )
    wi1_q, wi1_scale = _quantize_moe_weight(
        w_wi1_fp,
        weight_dtype,
        weight_block_size,
        scale_format,
    )
    wo_q, wo_scale = _quantize_moe_weight(
        w_wo_fp,
        weight_dtype,
        weight_block_size,
        scale_format,
    )
    wi0_scale_sharding, wi1_scale_sharding, wo_scale_sharding = _get_scale_shardings(scale_format)

    with jax.set_mesh(moe_ref.moe_mesh):
        moe_ref.wi_0 = nnx.Param(w_wi0_fp, out_sharding=P("expert", None, "tensor"))
        moe_ref.wi_1 = nnx.Param(w_wi1_fp, out_sharding=P("expert", None, "tensor"))
        moe_ref.wo = nnx.Param(w_wo_fp, out_sharding=P("expert", "tensor", None))

        moe_quant.wi_0 = nnx.Param(wi0_q, out_sharding=P("expert", None, "tensor"))
        moe_quant.wi_1 = nnx.Param(wi1_q, out_sharding=P("expert", None, "tensor"))
        moe_quant.wo = nnx.Param(wo_q, out_sharding=P("expert", "tensor", None))

        del moe_quant.wi_0_scale
        del moe_quant.wi_1_scale
        del moe_quant.wo_scale
        moe_quant.wi_0_scale = nnx.Param(wi0_scale, out_sharding=wi0_scale_sharding)
        moe_quant.wi_1_scale = nnx.Param(wi1_scale, out_sharding=wi1_scale_sharding)
        moe_quant.wo_scale = nnx.Param(wo_scale, out_sharding=wo_scale_sharding)

    expected_scale_ndim = 2 if scale_format == "per_channel" else 3
    assert moe_quant.wi_0_scale.value.ndim == expected_scale_ndim
    assert moe_quant.wi_1_scale.value.ndim == expected_scale_ndim
    assert moe_quant.wo_scale.value.ndim == expected_scale_ndim

    batch_size = 16
    x = jax.random.normal(k_x, (batch_size, hidden_size), dtype=compute_dtype)
    topk_weights = jnp.ones((batch_size, num_experts_per_tok), dtype=compute_dtype)
    topk_ids = jax.random.randint(k_topk, (batch_size, num_experts_per_tok), 0, num_experts)

    with jax.set_mesh(moe_ref.moe_mesh):
        out_ref = moe_ref(x, topk_weights, topk_ids)
        out_quant = moe_quant(x, topk_weights, topk_ids)

    cos_sim = get_cosine_similarity(out_ref, out_quant, mesh)
    mae = jnp.mean(jnp.abs(out_ref - out_quant))
    rel_error = mae / jnp.mean(jnp.abs(out_ref))

    print("\n--- Accuracy Results ---")
    print(f"  scale_format:      {scale_format}")
    print(f"  Cosine Similarity: {cos_sim.item():.6f}")
    print(f"  Relative Error:    {rel_error.item():.6%}")
    print(f"  MAE:               {mae.item():.6f}")

    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.6f}"
    assert rel_error < 0.05, f"Relative error too high: {rel_error:.6%}"


if __name__ == "__main__":
    test_epmoe_block_quant_accuracy("per_channel", None)
    test_epmoe_block_quant_accuracy("block_channel", (1, 128))
    test_epmoe_block_quant_accuracy("block_quant", (128, 128))
