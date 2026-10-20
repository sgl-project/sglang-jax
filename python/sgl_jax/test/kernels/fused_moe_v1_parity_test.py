"""Numerical parity, metadata edge-case, and TPU AOT compilation tests for FusedEPMoE v1."""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.experimental.topologies import get_topology_desc
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    compute_local_expert_sizes,
    fused_ep_moe,
    jax_allreduce_metadata_by_bt,
)
from sgl_jax.srt.utils.jax_utils import compilation_target
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

jax.config.parse_flags_with_absl()


def _dense_iota_expert_sizes_ref(topk_ids: jax.Array, num_experts: int) -> jax.Array:
    """Reference dense iota-comparison expert count (ignores invalid ids < 0 or >= num_experts)."""
    expert_iota = lax.broadcasted_iota(jnp.int32, (1, 1, num_experts), 2)
    mask = (topk_ids[:, :, None] == expert_iota).astype(jnp.int32)
    return jnp.sum(mask, axis=(0, 1), keepdims=True).reshape(1, num_experts)


def _sublane_unpack(x_3d: jax.Array) -> jax.Array:
    return jnp.transpose(x_3d, (1, 0, 2))


def _sublane_pack(x_p0: jax.Array, x_p1: jax.Array) -> jax.Array:
    return jnp.transpose(jnp.stack([x_p0, x_p1], axis=0), (1, 0, 2))


def _simulate_expert_ffn_bf_slices(
    tokens_3d: jax.Array,
    w1_4d: jax.Array,
    w3_4d: jax.Array,
    w2_4d: jax.Array,
    *,
    num_bf: int,
    use_relayout: bool,
    use_sublane_pack: bool,
) -> jax.Array:
    """Simulate dynamic_ffn1 + dynamic_ffn2 across bf_id chunks with and without 2D VMEM relayout."""
    num_tokens, t_packing, bd_per_pack = tokens_3d.shape
    bf = w1_4d.shape[-1] // num_bf
    t_dtype = tokens_3d.dtype
    use_2d_cache = use_relayout and (num_bf > 1)

    x_2d_cache = (
        jnp.zeros((t_packing, num_tokens, bd_per_pack), dtype=t_dtype) if use_2d_cache else None
    )
    res_2d_cache = (
        jnp.zeros((t_packing, num_tokens, bd_per_pack), dtype=t_dtype) if use_2d_cache else None
    )
    res_vmem = jnp.zeros((num_tokens, t_packing, bd_per_pack), dtype=t_dtype)

    for bf_id in range(num_bf):
        bf_slice = slice(bf_id * bf, (bf_id + 1) * bf)
        acc1 = jnp.zeros((num_tokens, bf), dtype=jnp.float32)
        acc3 = jnp.zeros((num_tokens, bf), dtype=jnp.float32)

        t_unpacked = _sublane_unpack(tokens_3d) if use_sublane_pack and bf_id == 0 else None
        if use_2d_cache and bf_id == 0 and t_unpacked is not None:
            x_2d_cache = t_unpacked

        for p_id in range(t_packing):
            if use_2d_cache and bf_id > 0:
                t_slice = x_2d_cache[p_id]
            elif t_unpacked is not None:
                t_slice = t_unpacked[p_id]
            else:
                t_slice = tokens_3d[:, p_id, :]
                if use_2d_cache and bf_id == 0:
                    x_2d_cache = x_2d_cache.at[p_id].set(t_slice)

            acc1 += jnp.dot(t_slice, w1_4d[p_id, :, bf_slice], preferred_element_type=jnp.float32)
            acc3 += jnp.dot(t_slice, w3_4d[p_id, :, bf_slice], preferred_element_type=jnp.float32)

        act = jax.nn.silu(acc1) * acc3

        packed_slices = []
        prev_unpacked = (
            _sublane_unpack(res_vmem)
            if (use_sublane_pack and not use_2d_cache and bf_id > 0)
            else None
        )
        for p_id in range(t_packing):
            res = jnp.dot(act, w2_4d[p_id, bf_slice, :], preferred_element_type=jnp.float32)
            if use_2d_cache:
                if bf_id == 0:
                    res_2d_cache = res_2d_cache.at[p_id].set(res.astype(t_dtype))
                elif bf_id < num_bf - 1:
                    prev = res_2d_cache[p_id].astype(jnp.float32)
                    res_2d_cache = res_2d_cache.at[p_id].set((prev + res).astype(t_dtype))
                elif use_sublane_pack:
                    prev = res_2d_cache[p_id].astype(jnp.float32)
                    packed_slices.append((prev + res).astype(t_dtype))
                else:
                    prev = res_2d_cache[p_id].astype(jnp.float32)
                    res_vmem = res_vmem.at[:, p_id, :].set((prev + res).astype(t_dtype))
            elif use_sublane_pack:
                if bf_id == 0:
                    packed_slices.append(res.astype(t_dtype))
                else:
                    prev = prev_unpacked[p_id].astype(jnp.float32)
                    packed_slices.append((prev + res).astype(t_dtype))
            else:
                if bf_id == 0:
                    res_vmem = res_vmem.at[:, p_id, :].set(res.astype(t_dtype))
                else:
                    prev = res_vmem[:, p_id, :].astype(jnp.float32)
                    res_vmem = res_vmem.at[:, p_id, :].set((prev + res).astype(t_dtype))

        if packed_slices:
            res_vmem = _sublane_pack(packed_slices[0], packed_slices[1])

    return res_vmem


class FusedMoEV1ParityTest(jtu.JaxTestCase):

    @parameterized.parameters(
        dict(num_tokens=16, top_k=4, num_experts=8, mode="uniform"),
        dict(num_tokens=16, top_k=4, num_experts=6, mode="non_pow2_experts"),
        dict(num_tokens=32, top_k=8, num_experts=16, mode="all_to_one_expert"),
        dict(num_tokens=32, top_k=8, num_experts=16, mode="zero_token_experts"),
        dict(num_tokens=24, top_k=6, num_experts=12, mode="with_invalid_padding"),
    )
    def test_compute_local_expert_sizes_matches_dense_iota(
        self, num_tokens, top_k, num_experts, mode
    ):
        key = jax.random.key(42)
        if mode == "all_to_one_expert":
            topk_ids = jnp.full((num_tokens, top_k), 3, dtype=jnp.int32)
        elif mode == "zero_token_experts":
            # Route only to even-numbered experts; odd experts get zero tokens.
            raw = jax.random.randint(key, (num_tokens, top_k), 0, num_experts // 2, dtype=jnp.int32)
            topk_ids = raw * 2
        elif mode == "with_invalid_padding":
            topk_ids = jax.random.randint(key, (num_tokens, top_k), 0, num_experts, dtype=jnp.int32)
            # Mark tail tokens and out-of-range ids as invalid sentinels (-1 and num_experts + 5).
            topk_ids = topk_ids.at[-5:, :].set(-1)
            topk_ids = topk_ids.at[0, -1].set(num_experts + 5)
        else:
            topk_ids = jax.random.randint(key, (num_tokens, top_k), 0, num_experts, dtype=jnp.int32)

        actual = compute_local_expert_sizes(topk_ids, num_experts)
        expected = _dense_iota_expert_sizes_ref(topk_ids, num_experts)
        self.assertArraysEqual(actual, expected)

    def test_jax_allreduce_metadata_by_bt_single_device_parity(self):
        devices = jax.devices("cpu")[:1]
        mesh = jax.sharding.Mesh(np.array(devices).reshape(1, 1), ("data", "tensor"))
        bt = 8
        num_bt = 3
        num_tokens = bt * num_bt
        top_k = 4
        num_experts = 6  # non-power-of-two expert count

        key = jax.random.key(7)
        topk_ids = jax.random.randint(key, (num_tokens, top_k), 0, num_experts, dtype=jnp.int32)
        topk_ids = topk_ids.at[-3:, :].set(-1)  # partial-tile invalid padding tokens

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(("data", "tensor"), None),),
            out_specs=(
                P(("data", "tensor"), None, None),
                P(("data", "tensor"), None, None),
                P(("data", "tensor"), None, None, None),
            ),
            check_vma=False,
        )
        def _run(ids):
            return jax_allreduce_metadata_by_bt(
                ids,
                num_experts=num_experts,
                bt=bt,
                num_devices=1,
                dp_axis_name="data",
                tp_axis_name="tensor",
            )

        starts, sizes, d2e_counts = _run(topk_ids)
        self.assertEqual(starts.shape, (num_bt, 1, num_experts))
        self.assertEqual(sizes.shape, (num_bt, 1, num_experts))
        self.assertEqual(d2e_counts.shape, (num_bt, 1, 1, num_experts))
        # On rank 0, prefix starts must be all zeros.
        self.assertArraysEqual(starts, jnp.zeros_like(starts))
        for bt_id in range(num_bt):
            tile_ids = topk_ids[bt_id * bt : (bt_id + 1) * bt]
            expected_sizes = _dense_iota_expert_sizes_ref(tile_ids, num_experts)
            self.assertArraysEqual(sizes[bt_id], expected_sizes)
            self.assertArraysEqual(d2e_counts[bt_id, 0], expected_sizes)

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        num_bf=[1, 2, 3],
    )
    def test_2d_tile_relayout_and_sublane_pack_bit_exact_parity(self, dtype, num_bf):
        key = jax.random.key(123)
        k0, k1, k2, k3 = jax.random.split(key, 4)
        num_tokens = 16
        t_packing = 2
        bd_per_pack = 64
        bf = 64
        inter_size = bf * num_bf

        tokens_3d = jax.random.normal(k0, (num_tokens, t_packing, bd_per_pack), dtype=dtype) / 4
        w1_4d = jax.random.normal(k1, (t_packing, bd_per_pack, inter_size), dtype=dtype) / 4
        w3_4d = jax.random.normal(k2, (t_packing, bd_per_pack, inter_size), dtype=dtype) / 4
        w2_4d = jax.random.normal(k3, (t_packing, inter_size, bd_per_pack), dtype=dtype) / 4

        baseline = _simulate_expert_ffn_bf_slices(
            tokens_3d,
            w1_4d,
            w3_4d,
            w2_4d,
            num_bf=num_bf,
            use_relayout=False,
            use_sublane_pack=False,
        )
        sublane_only = _simulate_expert_ffn_bf_slices(
            tokens_3d,
            w1_4d,
            w3_4d,
            w2_4d,
            num_bf=num_bf,
            use_relayout=False,
            use_sublane_pack=True,
        )
        relayout_and_sublane = _simulate_expert_ffn_bf_slices(
            tokens_3d,
            w1_4d,
            w3_4d,
            w2_4d,
            num_bf=num_bf,
            use_relayout=True,
            use_sublane_pack=True,
        )
        # Sublane pack/unpack and 2D VMEM tile caching only reorder memory layout
        # and reuse identical slices across bf_id, so outputs are bit-exact.
        self.assertArraysEqual(sublane_only, baseline)
        self.assertArraysEqual(relayout_and_sublane, baseline)

    @parameterized.parameters(
        # 1. Standard decode with 2D VMEM relayout active (num_bd1=1, num_bd2=1, num_bf=2)
        dict(
            num_tokens=64,
            num_experts=64,
            top_k=8,
            hidden_size=1024,
            inter_size=1024,
            bf=512,
            bd1=1024,
            bd2=1024,
            has_shared=True,
            relayout=True,
            sublane=True,
        ),
        # 2. Baseline flags disabled (regression check)
        dict(
            num_tokens=64,
            num_experts=64,
            top_k=8,
            hidden_size=1024,
            inter_size=1024,
            bf=512,
            bd1=1024,
            bd2=1024,
            has_shared=False,
            relayout=False,
            sublane=False,
        ),
        # 3. Non-power-of-two local_num_experts (48 experts / 8 devices = 6 local experts)
        dict(
            num_tokens=64,
            num_experts=48,
            top_k=6,
            hidden_size=1024,
            inter_size=1024,
            bf=512,
            bd1=1024,
            bd2=1024,
            has_shared=False,
            relayout=True,
            sublane=True,
        ),
        # 4. Multi-slice hidden dimension (num_bd1=2, num_bd2=2 -> 2D cache auto-disabled)
        dict(
            num_tokens=64,
            num_experts=64,
            top_k=8,
            hidden_size=2048,
            inter_size=1024,
            bf=512,
            bd1=1024,
            bd2=1024,
            has_shared=False,
            relayout=True,
            sublane=True,
        ),
    )
    def test_tpu_aot_compile_variants(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        inter_size,
        bf,
        bd1,
        bd2,
        has_shared,
        relayout,
        sublane,
    ):
        devices = get_topology_desc(
            platform="tpu",
            topology_name="v6e:2x4",
            chip_config_name="default",
            chips_per_host_bounds=(2, 2, 1),
            num_slices=1,
            wrap=(False, False, False),
        ).devices
        mesh = create_device_mesh([8, 1], [1, 1], devices=devices, is_full_topology=True)
        ep_sharding = NamedSharding(mesh, P(("data", "tensor"), None))
        w_sharding = NamedSharding(mesh, P(("data", "tensor"), None, None))
        se_sharding = NamedSharding(mesh, P(None, None))

        bt = num_tokens // 8
        cfg = FusedMoEBlockConfig(
            bt=bt,
            bf=bf,
            bd1=bd1,
            bd2=bd2,
            bts=bt,
            btc=bt,
            bfc=min(bf, 256),
            bd1c=min(bd1, 512),
            bd2c=min(bd2, 512),
            bse=bf,
        )
        tokens_sd = jax.ShapeDtypeStruct(
            (num_tokens, hidden_size), jnp.bfloat16, sharding=ep_sharding
        )
        w1_sd = jax.ShapeDtypeStruct(
            (num_experts, hidden_size, inter_size), jnp.bfloat16, sharding=w_sharding
        )
        w2_sd = jax.ShapeDtypeStruct(
            (num_experts, inter_size, hidden_size), jnp.bfloat16, sharding=w_sharding
        )
        w3_sd = jax.ShapeDtypeStruct(
            (num_experts, hidden_size, inter_size), jnp.bfloat16, sharding=w_sharding
        )
        topk_w_sd = jax.ShapeDtypeStruct((num_tokens, top_k), jnp.float32, sharding=ep_sharding)
        topk_id_sd = jax.ShapeDtypeStruct((num_tokens, top_k), jnp.int32, sharding=ep_sharding)

        w1_se_sd = (
            jax.ShapeDtypeStruct((hidden_size, inter_size), jnp.bfloat16, sharding=se_sharding)
            if has_shared
            else None
        )
        w2_se_sd = (
            jax.ShapeDtypeStruct((inter_size, hidden_size), jnp.bfloat16, sharding=se_sharding)
            if has_shared
            else None
        )
        w3_se_sd = (
            jax.ShapeDtypeStruct((hidden_size, inter_size), jnp.bfloat16, sharding=se_sharding)
            if has_shared
            else None
        )

        def _fn(tokens, w1, w2, w3, topk_w, topk_id, w1_se, w2_se, w3_se):
            return fused_ep_moe(
                mesh=mesh,
                tokens=tokens,
                w1=w1,
                w2=w2,
                w3=w3,
                topk_weights=topk_w,
                topk_ids=topk_id,
                top_k=top_k,
                act_fn="silu",
                block_config=cfg,
                w1_shared=w1_se,
                w2_shared=w2_se,
                w3_shared=w3_se,
                vmem_2d_tile_relayout=relayout,
                vmem_packed_sublane_transpose=sublane,
                dp_axis_name="data",
                tp_axis_name="tensor",
            )

        with compilation_target(mesh):
            lowered = jax.jit(_fn, out_shardings=ep_sharding).lower(
                tokens_sd,
                w1_sd,
                w2_sd,
                w3_sd,
                topk_w_sd,
                topk_id_sd,
                w1_se_sd,
                w2_se_sd,
                w3_se_sd,
            )
            hlo_text = lowered.as_text("hlo")
            compiled = lowered.compile()

        self.assertIn("fused_ep_moe", hlo_text)
        # Verify bincount lowering is preserved when jax_allreduce_metadata is active.
        self.assertIsNotNone(compiled)

    def test_tpu_aot_compile_pipelined_ring_buffer_path(self):
        """Force expert_buffer_count=2 < local_num_experts=8 to verify the pipelined A2A path."""
        devices = get_topology_desc(
            platform="tpu",
            topology_name="v6e:2x4",
            chip_config_name="default",
            chips_per_host_bounds=(2, 2, 1),
            num_slices=1,
            wrap=(False, False, False),
        ).devices
        mesh = create_device_mesh([8, 1], [1, 1], devices=devices, is_full_topology=True)
        ep_sharding = NamedSharding(mesh, P(("data", "tensor"), None))
        w_sharding = NamedSharding(mesh, P(("data", "tensor"), None, None))

        cfg = FusedMoEBlockConfig(
            bt=8,
            bf=512,
            bd1=1024,
            bd2=1024,
            bts=8,
            btc=8,
            bfc=256,
            bd1c=512,
            bd2c=512,
            bse=512,
        )
        tokens_sd = jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16, sharding=ep_sharding)
        w1_sd = jax.ShapeDtypeStruct((64, 1024, 1024), jnp.bfloat16, sharding=w_sharding)
        w2_sd = jax.ShapeDtypeStruct((64, 1024, 1024), jnp.bfloat16, sharding=w_sharding)
        w3_sd = jax.ShapeDtypeStruct((64, 1024, 1024), jnp.bfloat16, sharding=w_sharding)
        topk_w_sd = jax.ShapeDtypeStruct((64, 8), jnp.float32, sharding=ep_sharding)
        topk_id_sd = jax.ShapeDtypeStruct((64, 8), jnp.int32, sharding=ep_sharding)

        def _fn(tokens, w1, w2, w3, topk_w, topk_id):
            return fused_ep_moe(
                mesh=mesh,
                tokens=tokens,
                w1=w1,
                w2=w2,
                w3=w3,
                topk_weights=topk_w,
                topk_ids=topk_id,
                top_k=8,
                act_fn="silu",
                block_config=cfg,
                dp_axis_name="data",
                tp_axis_name="tensor",
            )

        with (
            mock.patch(
                "sgl_jax.srt.kernels.fused_moe.v1.kernel._device_hbm_bytes",
                return_value=1024,
            ),
            compilation_target(mesh),
        ):
            lowered = jax.jit(_fn, out_shardings=ep_sharding).lower(
                tokens_sd, w1_sd, w2_sd, w3_sd, topk_w_sd, topk_id_sd
            )
            compiled = lowered.compile()

        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
