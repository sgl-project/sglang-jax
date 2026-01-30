# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tests/kernels/fused_moe_v1_test.py
# Copyright 2025 The tpu-inference Authors. All rights reserved.
import glob
import os
import re
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    ref_moe,
)
from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.test.test_utils import create_device_mesh

jax.config.parse_flags_with_absl()


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
    has_shared_expert=False,
    se_intermediate_size=None,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 12)
    k0, k1, k2, k3, k4, k5, k6, k7, k8 = keys[:9]

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

    if has_bias:
        b1 = (
            jax.random.normal(k4, (num_experts, 1, intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        b2 = (jax.random.normal(k5, (num_experts, 1, hidden_size), dtype=jnp.float32) / 10).astype(
            dtype
        )
        b3 = (
            jax.random.normal(k6, (num_experts, 1, intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
    else:
        b1 = b2 = b3 = None

    # Shared Expert Weights
    w1_shared = w2_shared = w3_shared = None
    if has_shared_expert:
        if se_intermediate_size is None:
            se_intermediate_size = intermediate_size

        k9, k10, k11 = keys[9:]
        w1_shared = (
            jax.random.normal(k9, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w2_shared = (
            jax.random.normal(k10, (se_intermediate_size, hidden_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w3_shared = (
            jax.random.normal(k11, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)

    # Construct gating logits with deterministic, strictly-ordered top-k per token.
    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)

    # Generate unique top-k indices per token (sample without replacement).
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)

    # Add a strictly decreasing boost so top-1 > top-2 > ... > top-k
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    gating_output = (gating_output + one_hot).astype(dtype)

    return a, w1, w2, w3, b1, b2, b3, gating_output, w1_shared, w2_shared, w3_shared


def sub_channel_quantize(x, quant_dtype, wsz=256):
    """Quantizes x with sub-channel quantization on the 2nd minor."""
    if jnp.issubdtype(quant_dtype, jnp.floating):
        dtype_info = jnp.finfo(quant_dtype)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)
    w_lst, scale_lst = [], []
    assert len(x.shape) >= 2
    assert x.shape[-2] % wsz == 0
    for i in range(0, x.shape[-2], wsz):
        y = x[..., i : i + wsz, :]
        abs_max = jnp.abs(y).max(axis=-2, keepdims=True)
        scale = (abs_max / dtype_max).astype(jnp.float32)
        w = (y / scale).astype(quant_dtype)
        w_lst.append(w)
        scale_lst.append(scale)
    return jnp.concat(w_lst, axis=-2), jnp.expand_dims(jnp.concat(scale_lst, axis=-2), axis=-2)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        # Use the shared helper so multi-host runs get a consistent device ordering.
        # Mesh axes are ("data", "tensor"), matching fused_ep_moe defaults.
        self.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    def _test_moe(
        self,
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed,
        renormalize_topk_logits,
        bt: int | None = None,
        bf: int | None = None,
        bd1: int | None = None,
        bd2: int | None = None,
        btc: int | None = None,
        bfc: int | None = None,
        bd1c: int | None = None,
        bd2c: int | None = None,
        bse: int | None = None,
        act_fn="silu",
        w_dtype=None,
        subc_quant_wsz=None,
        has_bias=False,
        has_shared_expert=False,
        use_grouped_topk=False,
        num_groups=1,
        top_k_groups=1,
        atol=2e-1,
        rtol=2e-1,
    ):
        a, w1, w2, w3, b1, b2, b3, gating_output, w1_shared, w2_shared, w3_shared = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=seed,
            has_bias=has_bias,
            has_shared_expert=has_shared_expert,
        )
        w1_scale = w2_scale = w3_scale = None
        w1_shared_scale = w2_shared_scale = w3_shared_scale = None

        if w_dtype is not None:
            if subc_quant_wsz is None:
                subc_quant_wsz = 256
            # Match FusedEPMoE's quantization path: block-quantize along axis=1.
            w1, w1_scale_3d = quantize_tensor(w_dtype, w1, axis=1, block_size=subc_quant_wsz)
            w3, w3_scale_3d = quantize_tensor(w_dtype, w3, axis=1, block_size=subc_quant_wsz)
            w2, w2_scale_3d = quantize_tensor(w_dtype, w2, axis=1, block_size=subc_quant_wsz)

            # Reshape scales to the 4D layout expected by the kernel.
            w1_scale = w1_scale_3d.reshape(
                w1_scale_3d.shape[0], w1_scale_3d.shape[1], 1, w1_scale_3d.shape[2]
            )
            w3_scale = w3_scale_3d.reshape(
                w3_scale_3d.shape[0], w3_scale_3d.shape[1], 1, w3_scale_3d.shape[2]
            )
            w2_scale = w2_scale_3d.reshape(
                w2_scale_3d.shape[0], w2_scale_3d.shape[1], 1, w2_scale_3d.shape[2]
            )

            if has_shared_expert:
                # Shared-expert weights use per-column scaling (axis=0) to keep
                # scale tensors compact and avoid sub-channel tiling issues.
                w1_shared, w1_se_scale_1d = quantize_tensor(w_dtype, w1_shared, axis=0)
                w3_shared, w3_se_scale_1d = quantize_tensor(w_dtype, w3_shared, axis=0)
                w2_shared, w2_se_scale_1d = quantize_tensor(w_dtype, w2_shared, axis=0)

                w1_shared_scale = w1_se_scale_1d.reshape(1, 1, w1_se_scale_1d.shape[0])
                w3_shared_scale = w3_se_scale_1d.reshape(1, 1, w3_se_scale_1d.shape[0])
                w2_shared_scale = w2_se_scale_1d.reshape(1, 1, w2_se_scale_1d.shape[0])

        block_config = None
        block_params = (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c, bse)
        if any(p is not None for p in block_params):
            assert all(p is not None for p in block_params), (
                "Either provide all explicit block params (bt/bf/bd1/bd2/btc/bfc/bd1c/bd2c/bse) "
                "or omit them all to use tuned configs."
            )
            block_config = FusedMoEBlockConfig(
                bt=cast(int, bt),
                bf=cast(int, bf),
                bd1=cast(int, bd1),
                bd2=cast(int, bd2),
                btc=cast(int, btc),
                bfc=cast(int, bfc),
                bd1c=cast(int, bd1c),
                bd2c=cast(int, bd2c),
                bse=cast(int, bse),
            )

        actual = fused_ep_moe(
            mesh=self.mesh,
            tokens=a,
            w1=w1,
            w2=w2,
            w3=w3,
            gating_output=gating_output,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            b1=b1,
            b2=b2,
            b3=b3,
            w1_shared=w1_shared,
            w2_shared=w2_shared,
            w3_shared=w3_shared,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            block_config=block_config,
            tp_axis_name="tensor",
        )
        expected = ref_moe(
            a,
            w1,
            w2,
            w3,
            gating_output,
            top_k,
            use_grouped_topk=use_grouped_topk,
            num_groups=num_groups,
            top_k_groups=top_k_groups,
            b1=b1,
            b2=b2,
            b3=b3,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared,
            w2_shared=w2_shared,
            w3_shared=w3_shared,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
        )

        # In multi-host runs, `actual` is sharded across processes and is not fully addressable
        # on any single process. Gather to a fully-replicated array for comparison.
        @jax.jit
        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(
                P(
                    ("data", "tensor"),
                ),
            ),
            out_specs=P(),
            check_vma=False,
        )
        def _replicate_tokens(x):
            x = lax.all_gather(x, axis_name="tensor", axis=0, tiled=True)
            x = lax.all_gather(x, axis_name="data", axis=0, tiled=True)
            return x

        actual_host = np.asarray(jax.device_get(_replicate_tokens(actual)))
        expected_host = np.asarray(jax.device_get(_replicate_tokens(expected)))
        self.assertAllClose(actual_host, expected_host, atol=atol, rtol=rtol)

    @parameterized.product(
        renormalize_topk_logits=[True, False],
    )
    def test_basic(self, renormalize_topk_logits):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    def test_shared_expert(self):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            has_shared_expert=True,
            bt=32,
            bf=512,  # smaller bf to test loop
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    def test_grouped_topk(self):
        dtype = jnp.float32
        top_k = 4
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            use_grouped_topk=True,
            num_groups=4,
            top_k_groups=2,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    @parameterized.product(
        act_fn=["silu", "gelu", "swigluoai"],
    )
    def test_activation(self, act_fn):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=True,
            act_fn=act_fn,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    def test_benchmark_qwen_235b(self):
        num_experts = 128
        top_k = 8
        hidden_size = 4096
        intermediate_size = 1536
        dtype = jnp.bfloat16
        num_tokens = 8 * 64
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    def test_benchmark_qwen_30b_a3b(self):
        num_experts = 128
        top_k = 8
        hidden_size = 2048
        intermediate_size = 768
        dtype = jnp.bfloat16
        num_tokens = 512
        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            bt=16,
            bf=384,
            bd1=512,
            bd2=512,
            btc=16,
            bfc=384,
            bd1c=256,
            bd2c=256,
            bse=512,
            act_fn="silu",
            atol=5e-2,
            rtol=5e-2,
        )

    @parameterized.product(
        w_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float4_e2m1fn],
    )
    def test_sub_channel_quantization(self, w_dtype):
        if w_dtype in (
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.float4_e2m1fn,
        ) and not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            w_dtype=w_dtype,
            subc_quant_wsz=256,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    @parameterized.product(
        w_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float4_e2m1fn],
    )
    def test_shared_expert_quantized(self, w_dtype):
        if w_dtype in (
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.float4_e2m1fn,
        ) and not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            w_dtype=w_dtype,
            subc_quant_wsz=256,
            has_shared_expert=True,
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )

    def test_fused_moe_weight_mapping_shards_experts_across_ep_mesh(self):
        # This guards an easy-to-miss end-to-end correctness issue: the fused
        # kernel shards experts across EP=(data*tensor), so weight loading must
        # shard expert dim (axis=0) across ("data","tensor"), not just "tensor".
        mappings = create_moe_weights_mapping(
            prefix="model.layers.0",
            target_prefix="model.layers.0",
            num_experts=8,
            moe_backend="fused",
            moe_path="mlp",
            source_expert_pattern="experts.{i}",
        )
        for m in mappings.values():
            # All fused expert weights are 3D and should shard axis0 across EP mesh.
            self.assertEqual(m.sharding, (("data", "tensor"), None, None))

    def test_real_model_ling_mini_fused_moe_smoke(self):
        # Optional slow integration test: load a small subset of real MoE weights
        # from a local HF-style safetensors directory and compare fused_ep_moe vs
        # ref_moe on a tiny problem.
        #
        # Usage (example):
        #   SGLANG_RUN_SLOW_TESTS=1 \
        #   SGLANG_TEST_MODEL_DIR=/path/to/inclusionAI/Ling-mini-2.0 \
        #   python3 -m unittest fused_moe_v1_test.MoEKernelTest.test_real_model_ling_mini_fused_moe_smoke
        if os.getenv("SGLANG_RUN_SLOW_TESTS", "0") != "1":
            self.skipTest("Set SGLANG_RUN_SLOW_TESTS=1 to enable.")

        model_dir = os.getenv("SGLANG_TEST_MODEL_DIR")
        if not model_dir:
            self.skipTest("Set SGLANG_TEST_MODEL_DIR to a local model directory.")

        try:
            from safetensors import safe_open  # type: ignore[import-not-found]
        except Exception as e:
            self.skipTest(f"safetensors is required for this test: {e}")

        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+ for fused MoE kernel.")

        layer_idx = int(os.getenv("SGLANG_TEST_LAYER_IDX", "0"))
        num_experts = int(os.getenv("SGLANG_TEST_NUM_EXPERTS", "8"))
        top_k = int(os.getenv("SGLANG_TEST_TOPK", "2"))
        num_tokens = int(os.getenv("SGLANG_TEST_NUM_TOKENS", "32"))

        # Find expert weights in safetensors.
        st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not st_files:
            raise RuntimeError(f"No *.safetensors files found under {model_dir}")

        # Patterns for HF checkpoints.
        # gate/up: (intermediate, hidden) in PT -> transpose to (hidden, intermediate)
        # down:    (hidden, intermediate) in PT -> transpose to (intermediate, hidden)
        base = rf"model\\.layers\\.{layer_idx}\\.mlp\\.experts\\.(\\d+)\\."
        gate_re = re.compile(base + r"gate_proj\\.weight$")
        up_re = re.compile(base + r"up_proj\\.weight$")
        down_re = re.compile(base + r"down_proj\\.weight$")

        se_gate_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
        se_up_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
        se_down_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"

        key_to_file: dict[str, str] = {}
        available_experts: set[int] = set()
        has_shared = False

        for st in st_files:
            with safe_open(st, framework="np", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    if k in (se_gate_key, se_up_key, se_down_key):
                        key_to_file[k] = st
                        has_shared = True
                        continue

                    m = gate_re.match(k) or up_re.match(k) or down_re.match(k)
                    if m:
                        e = int(m.group(1))
                        if e < num_experts:
                            key_to_file[k] = st
                            available_experts.add(e)

        if len(available_experts) < num_experts:
            raise RuntimeError(
                f"Found only {len(available_experts)} experts (<{num_experts}) for layer {layer_idx} under {model_dir}"
            )

        def _load(k: str) -> np.ndarray:
            st = key_to_file[k]
            with safe_open(st, framework="np", device="cpu") as f:
                return f.get_tensor(k)

        # Load and stack expert weights.
        w1_list, w2_list, w3_list = [], [], []
        for e in range(num_experts):
            gate_k = f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            up_k = f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
            down_k = f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"

            w_gate = _load(gate_k).astype(np.float32).T  # (hidden, inter)
            w_up = _load(up_k).astype(np.float32).T  # (hidden, inter)
            w_down = _load(down_k).astype(np.float32).T  # (inter, hidden)

            w1_list.append(w_gate)
            w3_list.append(w_up)
            w2_list.append(w_down)

        w1 = jnp.asarray(np.stack(w1_list, axis=0), dtype=jnp.bfloat16)
        w3 = jnp.asarray(np.stack(w3_list, axis=0), dtype=jnp.bfloat16)
        w2 = jnp.asarray(np.stack(w2_list, axis=0), dtype=jnp.bfloat16)

        hidden_size = int(w1.shape[1])
        _ = int(w2.shape[1])

        # Optional shared expert.
        w1_shared = w2_shared = w3_shared = None
        if has_shared:
            w1_shared = jnp.asarray(_load(se_gate_key).astype(np.float32).T, dtype=jnp.bfloat16)
            w3_shared = jnp.asarray(_load(se_up_key).astype(np.float32).T, dtype=jnp.bfloat16)
            w2_shared = jnp.asarray(_load(se_down_key).astype(np.float32).T, dtype=jnp.bfloat16)

        # Random inputs + deterministic top-k.
        key = jax.random.key(0)
        tokens = (jax.random.normal(key, (num_tokens, hidden_size), dtype=jnp.float32) / 10).astype(
            jnp.bfloat16
        )
        router_logits = jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)
        topk_ids = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
            jax.random.split(jax.random.key(1), num_tokens)
        ).astype(jnp.int32)
        boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
        one_hot = jnp.sum(
            jax.nn.one_hot(topk_ids, num_experts, dtype=jnp.float32) * boosts[..., None],
            axis=1,
        )
        router_logits = (router_logits + one_hot).astype(jnp.bfloat16)

        # Use a 1D EP mesh for this smoke test (data = all devices, tensor = 1)
        mesh = create_device_mesh(ici_parallelism=[-1, 1], dcn_parallelism=[1, 1])
        ep_size = mesh.shape["data"] * mesh.shape["tensor"]
        if num_tokens % ep_size != 0:
            self.skipTest(f"num_tokens ({num_tokens}) must be divisible by ep_size ({ep_size}).")

        actual = fused_ep_moe(
            mesh=mesh,
            tokens=tokens,
            w1=w1,
            w2=w2,
            w3=w3,
            gating_output=router_logits,
            top_k=top_k,
            renormalize_topk_logits=False,
            act_fn="silu",
            w1_shared=w1_shared,
            w2_shared=w2_shared,
            w3_shared=w3_shared,
            tp_axis_name="tensor",
        )
        expected = ref_moe(
            tokens,
            w1,
            w2,
            w3,
            router_logits,
            top_k,
            renormalize_topk_logits=False,
            act_fn="silu",
            w1_shared=w1_shared,
            w2_shared=w2_shared,
            w3_shared=w3_shared,
        )

        # Bring both to host for comparison (single-host smoke).
        self.assertAllClose(
            np.asarray(jax.device_get(actual)),
            np.asarray(jax.device_get(expected)),
            atol=3e-1,
            rtol=3e-1,
        )

    def test_bias(self):
        dtype = jnp.bfloat16
        top_k = 8
        num_experts = 128
        hidden_size = 1024
        intermediate_size = 1024
        num_tokens = 8 * 32
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=1234,
            renormalize_topk_logits=False,
            has_bias=True,
            bt=32,
            bf=512,
            bd1=512,
            bd2=512,
            btc=32,
            bfc=256,
            bd1c=256,
            bd2c=256,
            bse=512,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
