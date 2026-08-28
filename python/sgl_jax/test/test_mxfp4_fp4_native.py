"""The fp4-native MoE load path: same numbers as the bf16 path, a quarter of the HBM.

Dequantizing K3's experts to bf16 works at 4 layers and cannot work at 93 -- 2.723 T routed-expert
params cost **5,072 GiB** as bf16 against **1,347 GiB** kept as fp4, i.e. 26.8 chips of weights
instead of 7.1 at v7x's measured 189.5 GiB/chip. So the load path keeps the weights native fp4
in HBM, and the MoE widens per block scale at matmul time.

These tests pin the three things that make that substitution safe:

1. the unpacked weight is still **4 bits per value** (a widening anywhere defeats the point);
2. it carries the **same values** the bf16 path produces, in the layout gmm_v2 declares;
3. the scale lands in gmm_v2's ``(size_group, num_k_blocks, 1, size_n)`` shape and is decoded by
   exponent arithmetic, not a cast.

Real checkpoint tensors are used where available -- the group-count and K-major assertions are only
meaningful against real shapes.
"""

from __future__ import annotations

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.quantization.mxfp4 import (
    MXFP4_GROUP_SIZE,
    dequantize_tensor_from_mxfp4_packed,
)
from sgl_jax.srt.layers.quantization.mxfp4_moe import (
    build_fp4_expert_weights,
    e8m0_scale_to_kernel_layout,
    unpack_fp4_to_e2m1,
)

MODEL_DIR = os.environ.get("KIMI_K3_MODEL_DIR", "/dev/shm/k3_4l")


def _real_expert(proj: str = "w1"):
    """One real expert's (packed, scale) pair, or skip."""
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    if not files:
        pytest.skip(f"no checkpoint under {MODEL_DIR}; set KIMI_K3_MODEL_DIR")
    from safetensors import safe_open

    for path in files:
        with safe_open(path, "numpy") as h:
            for key in h.keys():
                if key.endswith(f".{proj}.weight_packed"):
                    scale_key = key.replace("weight_packed", "weight_scale")
                    return h.get_tensor(key), h.get_tensor(scale_key)
    pytest.skip(f"no packed {proj} tensor in {MODEL_DIR}")


def _device_bytes(make) -> tuple[int, int]:
    """(bytes actually resident in HBM, value count) for one freshly-allocated array.

    ``Array.nbytes`` is a HOST-side figure that rounds ``itemsize`` up to a whole byte, so it
    reports 1 B/value for a 4-bit dtype and would make a correct fp4 path look like a 2x
    regression. The device allocator is the only honest source, and this whole change is
    justified by a device-footprint claim, so it is the number under test.
    """
    import gc

    device = jax.devices()[0]

    def in_use() -> int:
        jax.block_until_ready(jnp.zeros(1))
        return device.memory_stats()["bytes_in_use"]

    gc.collect()
    base = in_use()
    arr = jax.block_until_ready(make())
    used = in_use() - base
    count = int(np.prod(arr.shape))
    del arr
    gc.collect()
    return used, count


def test_unpacked_weight_is_sub_byte_in_hbm():
    """The claim the full-model plan rests on: fp4 stays ~0.5 B/value in HBM, not 1 or 2.

    K3's 2.723 T routed-expert params are 5,072 GiB at 2 B/value and 1,347 GiB at 0.5 -- 26.8
    chips of weights versus 7.1 at v7x's 189.5 GiB/chip. If XLA silently widened fp4 to a byte,
    the port would still be numerically correct and the model would still not fit, so this is
    measured rather than assumed.
    """
    packed, _ = _real_expert()
    w = unpack_fp4_to_e2m1(jnp.asarray(packed))
    assert w.dtype == jnp.float4_e2m1fn, w.dtype
    assert jnp.finfo(w.dtype).bits == 4

    fp4_bytes, count = _device_bytes(lambda: unpack_fp4_to_e2m1(jnp.asarray(packed)))
    bf16_bytes, _ = _device_bytes(
        lambda: dequantize_tensor_from_mxfp4_packed(
            jnp.asarray(packed), jnp.asarray(_real_expert()[1]), axis=-1, out_dtype=jnp.bfloat16
        )
    )

    per_value = fp4_bytes / count
    # 0.5 exactly, plus whatever the allocator rounds up on a single array
    assert per_value < 0.75, f"fp4 is {per_value:.3f} B/value -- XLA did not pack it"
    assert fp4_bytes * 3 < bf16_bytes, (
        f"fp4 {fp4_bytes} B vs bf16 {bf16_bytes} B -- expected roughly a 4x saving"
    )


def test_layout_is_k_major_as_gmm_v2_declares():
    """``[N, K/2]`` -> ``[K, N]``. gmm_v2 asserts ``rhs.shape == (group, size_k, size_n)``."""
    packed, _ = _real_expert()
    n, k_half = packed.shape
    w = unpack_fp4_to_e2m1(jnp.asarray(packed))
    assert w.shape == (k_half * 2, n), (w.shape, packed.shape)


def test_fp4_path_carries_the_same_values_as_the_bf16_path():
    """Substitution safety: fp4-then-scale must equal the dequantize-to-bf16 result.

    Compared in fp32 after applying the scale, transposed to the same orientation. Any mismatch
    means the K-major swap and the group axis disagree -- which would corrupt every expert while
    still loading and running.
    """
    packed, scale = _real_expert()
    packed_j, scale_j = jnp.asarray(packed), jnp.asarray(scale)

    # bf16 path (validated elsewhere against an independent oracle), as [K, N] fp32
    ref = dequantize_tensor_from_mxfp4_packed(
        packed_j, scale_j, axis=-1, out_dtype=jnp.float32
    ).T

    # fp4-native path: values stay fp4; the scale is applied by the kernel, so apply it here
    w4 = unpack_fp4_to_e2m1(packed_j)
    s4 = e8m0_scale_to_kernel_layout(scale_j)  # [blocks, 1, N]
    k, n = w4.shape
    blocks = s4.shape[0]
    got = (w4.astype(jnp.float32).reshape(blocks, k // blocks, n) * s4).reshape(k, n)

    assert got.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_scale_lands_in_the_shape_gmm_v2_validates():
    """gmm_v2: ``rhs_scale.shape == (size_group, num_k_blocks, 1, size_n)`` and
    ``size_k % num_k_blocks == 0``. Both are asserted in its ``validate_inputs``."""
    packed, scale = _real_expert()
    n, k_half = packed.shape
    size_k = k_half * 2

    s4 = e8m0_scale_to_kernel_layout(jnp.asarray(scale))
    num_k_blocks = s4.shape[0]

    assert s4.shape == (num_k_blocks, 1, n), s4.shape
    assert s4.dtype == jnp.float32
    assert num_k_blocks == size_k // MXFP4_GROUP_SIZE
    assert size_k % num_k_blocks == 0, "gmm_v2 rejects a K that its block count does not divide"


def test_stacked_weights_match_the_kernels_rhs_and_scale_contract():
    """Per-layer assembly: ``[E, K, N]`` fp4 + ``[E, blocks, 1, N]`` fp32, experts complete.

    Uses a REAL expert's tensors replicated across a SMALL expert count. Materializing a real
    layer's full 896 experts is ~15 GB of packed weights plus device copies; an earlier version of
    this test did exactly that and OOM-killed the pod. The contract under test is per-expert
    shape/dtype/stacking -- the expert COUNT adds nothing to it, and the real shapes are what
    matter, which the replication preserves.
    """
    packed, scale = _real_expert()
    num_experts, prefix, layer = 4, "m.layers", 0

    tensors = {}
    for proj in ("w1", "w3", "w2"):
        for e in range(num_experts):
            base = f"{prefix}.{layer}.block_sparse_moe.experts.{e}.{proj}"
            tensors[f"{base}.weight_packed"] = jnp.asarray(packed)
            tensors[f"{base}.weight_scale"] = jnp.asarray(scale)

    built = build_fp4_expert_weights(tensors, layer, num_experts, prefix=prefix)
    assert set(built) == {"wi_0", "wi_1", "wo"}, sorted(built)

    for name, (w, s) in built.items():
        assert w.dtype == jnp.float4_e2m1fn, (name, w.dtype)
        assert s.dtype == jnp.float32, (name, s.dtype)
        e, size_k, size_n = w.shape
        assert e == num_experts, (name, w.shape)
        assert s.shape == (num_experts, s.shape[1], 1, size_n), (name, s.shape)
        assert size_k % s.shape[1] == 0, (name, size_k, s.shape[1])
        assert size_k == packed.shape[-1] * 2 and size_n == packed.shape[0]


def test_missing_expert_raises_rather_than_stacking_short():
    """Same guarantee the bf16 builder gives: a gap must not silently reorder the expert axis."""
    packed, scale = _real_expert()
    prefix, layer = "m.layers", 0
    tensors = {}
    for e in (0, 2):  # deliberately skip expert 1
        base = f"{prefix}.{layer}.block_sparse_moe.experts.{e}.w1"
        tensors[f"{base}.weight_packed"] = jnp.asarray(packed)
        tensors[f"{base}.weight_scale"] = jnp.asarray(scale)

    with pytest.raises(KeyError, match="missing"):
        build_fp4_expert_weights(tensors, layer, num_experts=3, prefix=prefix)
