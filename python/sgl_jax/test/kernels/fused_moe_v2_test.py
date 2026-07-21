# Copyright 2025 The sgl-jax Authors. All rights reserved.
"""Correctness tests for the fused EP-MoE v2 kernel (vs the reference impl).

Mirrors the style of ``fused_moe_v1_test.py``: build small random inputs, run
``fused_ep_moe_v2`` and ``ref_moe`` on the same routing, gather to host and
compare. Covers bf16 / fp8 (block-wise + per-channel), the shared expert, grouped
top-k, the activation variants, and the per-expert SwiGLU clamp.
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
    ref_moe,
)
from sgl_jax.srt.layers.moe import TopK
from sgl_jax.test.test_utils import create_device_mesh

jax.config.parse_flags_with_absl()

FP8 = jnp.float8_e4m3fn


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_shared_expert=False,
    se_intermediate_size=None,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 12)
    k0, k1, k2, k3, k7, k8 = keys[0], keys[1], keys[2], keys[3], keys[7], keys[8]

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

    w1_shared = w2_shared = w3_shared = None
    if has_shared_expert:
        if se_intermediate_size is None:
            se_intermediate_size = intermediate_size
        k9, k10, k11 = keys[9], keys[10], keys[11]
        w1_shared = (
            jax.random.normal(k9, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w2_shared = (
            jax.random.normal(k10, (se_intermediate_size, hidden_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w3_shared = (
            jax.random.normal(k11, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)

    # Strictly-ordered, deterministic top-k per token (top-1 > top-2 > ...).
    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    gating_output = (gating_output + one_hot).astype(dtype)
    return a, w1, w2, w3, gating_output, w1_shared, w2_shared, w3_shared


def _quant_expert_w(w, w_dtype, quant_block_k):
    """Quantize routed expert weights (E, K, N) along the K axis.

    Returns (fp8 w, scale). Matches bench_v2: per-channel (quant_block_k=None) ->
    scale (E, 1, 1, N); block-wise -> scale (E, K // quant_block_k, 1, N).
    """
    w_f32 = w.astype(jnp.float32)
    E, K, N = w.shape
    if quant_block_k is None:
        amax = jnp.max(jnp.abs(w_f32), axis=1, keepdims=True)  # (E, 1, N)
        scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
        w_q = (w_f32 / scale).astype(w_dtype)
        scale = scale[:, :, None, :]  # (E, 1, 1, N)
    else:
        wr = w_f32.reshape(E, K // quant_block_k, quant_block_k, N)
        amax = jnp.max(jnp.abs(wr), axis=2, keepdims=True)  # (E, K//qbk, 1, N)
        scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
        w_q = (wr / scale).astype(w_dtype).reshape(E, K, N)
    return w_q, scale.astype(jnp.float32)


def _quant_shared_w(w, w_dtype):
    """Quantize a shared-expert weight (K, N) per-channel over K -> scale (1, N)."""
    w_f32 = w.astype(jnp.float32)
    amax = jnp.max(jnp.abs(w_f32), axis=0, keepdims=True)  # (1, N)
    scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
    return (w_f32 / scale).astype(w_dtype), scale.astype(jnp.float32)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEV2KernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        # Mesh axes ("data", "tensor"); ep = data * tensor. Single line of devices
        # on the tensor axis matches fused_ep_moe_v2 defaults.
        self.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    def _test_moe(
        self,
        *,
        dtype=jnp.bfloat16,
        top_k=4,
        num_experts=16,
        hidden_size=512,
        intermediate_size=256,
        num_tokens=128,
        seed=1234,
        renormalize=True,
        act_fn="silu",
        swiglu_limit=None,
        shared_swiglu_limit=None,
        w_dtype=None,
        quant_block_k=None,
        direct_scaled_dot=False,
        has_shared_expert=False,
        se_intermediate_size=None,
        use_grouped_topk=False,
        num_groups=1,
        top_k_groups=1,
        bt=32,
        bf=128,
        btc=32,
        bse=128,
        atol=2e-1,
        rtol=2e-1,
    ):
        a, w1, w2, w3, gating, w1_sh, w2_sh, w3_sh = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=seed,
            has_shared_expert=has_shared_expert,
            se_intermediate_size=se_intermediate_size,
        )
        w1_scale = w2_scale = w3_scale = None
        w1_sh_scale = w2_sh_scale = w3_sh_scale = None
        if w_dtype is not None:
            w1, w1_scale = _quant_expert_w(w1, w_dtype, quant_block_k)
            w2, w2_scale = _quant_expert_w(w2, w_dtype, quant_block_k)
            w3, w3_scale = _quant_expert_w(w3, w_dtype, quant_block_k)
            if has_shared_expert:
                w1_sh, w1_sh_scale = _quant_shared_w(w1_sh, w_dtype)
                w2_sh, w2_sh_scale = _quant_shared_w(w2_sh, w_dtype)
                w3_sh, w3_sh_scale = _quant_shared_w(w3_sh, w_dtype)

        topk_module = TopK(
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_groups,
            topk_group=top_k_groups,
        )
        topk_weights, topk_ids = topk_module(gating)

        block_config = FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse)

        # fused_ep_moe_v2 feeds these straight into its shard_map, which shards dim 0 across the
        # EP mesh. ref_moe's scatter cannot take a sharded operand, so keep its originals.
        ep = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor")))

        def shard(x):
            return None if x is None else jax.device_put(x, ep)

        actual = fused_ep_moe_v2(
            self.mesh,
            shard(a),
            shard(w1),
            shard(w2),
            shard(w3),
            shard(topk_weights),
            shard(topk_ids),
            top_k,
            act_fn=act_fn,
            swiglu_limit=swiglu_limit,
            shared_swiglu_limit=shared_swiglu_limit,
            block_config=block_config,
            quant_block_k=quant_block_k,
            w1_scale=shard(w1_scale),
            w2_scale=shard(w2_scale),
            w3_scale=shard(w3_scale),
            w1_shared=w1_sh,
            w2_shared=w2_sh,
            w3_shared=w3_sh,
            w1_shared_scale=w1_sh_scale,
            w2_shared_scale=w2_sh_scale,
            w3_shared_scale=w3_sh_scale,
            direct_scaled_dot=direct_scaled_dot,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )
        expected = ref_moe(
            a,
            w1,
            w2,
            w3,
            topk_weights,
            topk_ids,
            top_k,
            act_fn=act_fn,
            swiglu_limit=swiglu_limit,
            shared_swiglu_limit=shared_swiglu_limit,
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_sh,
            w2_shared=w2_sh,
            w3_shared=w3_sh,
            w1_shared_scale=w1_sh_scale,
            w2_shared_scale=w2_sh_scale,
            w3_shared_scale=w3_sh_scale,
        )

        # Gather sharded outputs to a fully-replicated host array for comparison.
        @jax.jit
        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(P(("data", "tensor")),),
            out_specs=P(),
            check_vma=False,
        )
        def _replicate(x):
            x = lax.all_gather(x, axis_name="tensor", axis=0, tiled=True)
            x = lax.all_gather(x, axis_name="data", axis=0, tiled=True)
            return x

        # _replicate gathers from the EP layout, so the replicated ref output must match.
        expected = jax.reshard(expected, ep)
        actual_host = np.asarray(jax.device_get(_replicate(actual)))
        expected_host = np.asarray(jax.device_get(_replicate(expected)))
        self.assertAllClose(actual_host, expected_host, atol=atol, rtol=rtol)

    # ------------------------------------------------------------------ bf16

    @parameterized.product(renormalize=[True, False])
    def test_basic(self, renormalize):
        self._test_moe(renormalize=renormalize)

    def test_shared_expert(self):
        self._test_moe(has_shared_expert=True)

    def test_grouped_topk(self):
        self._test_moe(
            top_k=4,
            use_grouped_topk=True,
            num_groups=4,
            top_k_groups=2,
        )

    @parameterized.product(act_fn=["silu", "gelu", "swigluoai"])
    def test_activation(self, act_fn):
        self._test_moe(act_fn=act_fn)

    # ----------------------------------------------------------- swiglu clamp

    @parameterized.product(swiglu_limit=[4.0, 0.1])
    def test_swiglu_clamp_routed(self, swiglu_limit):
        self._test_moe(swiglu_limit=swiglu_limit)

    def test_swiglu_clamp_with_shared(self):
        # Routed and shared experts use different per-layer limits (mirrors
        # Ling3-Flash: routed=4, shared=5 on the late layers).
        self._test_moe(
            has_shared_expert=True,
            swiglu_limit=4.0,
            shared_swiglu_limit=5.0,
        )

    # -------------------------------------------------------------------- fp8

    def test_fp8_block_wise(self):
        self._test_moe(
            w_dtype=FP8,
            quant_block_k=128,
            direct_scaled_dot=False,
            atol=5e-2,
            rtol=5e-2,
        )

    def test_fp8_per_channel_with_shared(self):
        # Per-channel fp8 requires direct_scaled_dot=True; the in-kernel shared
        # expert reuses the routed fp8 buffers, so shared weights are fp8 too.
        self._test_moe(
            w_dtype=FP8,
            quant_block_k=None,
            direct_scaled_dot=True,
            has_shared_expert=True,
            atol=5e-2,
            rtol=5e-2,
        )

    def test_fp8_block_wise_swiglu_clamp(self):
        self._test_moe(
            w_dtype=FP8,
            quant_block_k=128,
            direct_scaled_dot=False,
            swiglu_limit=4.0,
            atol=5e-2,
            rtol=5e-2,
        )

    def test_fp8_per_channel_shared_swiglu_clamp(self):
        self._test_moe(
            w_dtype=FP8,
            quant_block_k=None,
            direct_scaled_dot=True,
            has_shared_expert=True,
            swiglu_limit=4.0,
            shared_swiglu_limit=5.0,
            atol=5e-2,
            rtol=5e-2,
        )

    # --------------------------------- num_bf==1 (global-rolling weight buffer)

    def test_num_bf1_rolling_wb(self):
        # bf == intermediate_size => num_bf == 1 => the global-rolling weight
        # double-buffer path (expert-parity slot), now the default. Previously
        # only reachable via the removed SGLJAX_MOE_V2_GLOBAL_ROLLING_WB flag.
        self._test_moe(bf=256)

    def test_num_bf1_rolling_wb_fp8_shared(self):
        # num_bf==1 (single bf block per expert) + per-channel fp8 weights +
        # in-kernel shared expert.
        self._test_moe(
            bf=256,
            bse=256,
            w_dtype=FP8,
            quant_block_k=None,
            direct_scaled_dot=True,
            has_shared_expert=True,
            atol=5e-2,
            rtol=5e-2,
        )

    # ------------------------------------------ compact loop empty-expert skip

    def test_compact_skip_empty_experts(self):
        # Many experts + few tokens + top_k=1 => several local experts receive
        # no tokens, exercising the compact active-expert list (n_active <
        # local_num_experts).
        self._test_moe(num_experts=32, num_tokens=16, top_k=1)

    # --------------------------------------------------- block config (no TPU)

    def test_effective_for_bt_gcd_reduction(self):
        cfg = FusedMoEBlockConfig(bt=64, bf=512, btc=64, bse=512)
        eff = cfg.effective_for(num_tokens=256, ep_size=8)  # local = 32
        self.assertEqual(eff.bt, 32)  # min(64, 32) then gcd(32, 32)
        self.assertEqual(32 % eff.bt, 0)

    def test_effective_for_bts_defaults_to_bt(self):
        cfg = FusedMoEBlockConfig(bt=16, bf=512, btc=16, bse=512, bts=None)
        eff = cfg.effective_for(num_tokens=256, ep_size=8)
        self.assertEqual(eff.bts, eff.bt)
        self.assertEqual(eff.bts % eff.btc, 0)

    def test_effective_for_raises_on_indivisible_tokens(self):
        cfg = FusedMoEBlockConfig(bt=16, bf=512, btc=16, bse=512)
        with self.assertRaises(ValueError):
            cfg.effective_for(num_tokens=100, ep_size=8)  # 100 % 8 != 0


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
