"""SwiGLU epilogue layout contract: interleave_gate_up + glu_from_interleaved == silu(x@w0) * (x@w1)."""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    glu_from_interleaved,
    validate_glu_chunk,
)
from sgl_jax.srt.layers.moe import glu_epilogue_chunk, interleave_gate_up


class GluEpilogueLayoutTest(unittest.TestCase):
    def test_interleaved_matmul_matches_separate_gate_up(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        m, k, n, chunk = 64, 256, 1024, 256
        x = jax.random.normal(k1, (m, k), jnp.float32)
        w0 = jax.random.normal(k2, (k, n), jnp.float32) * 0.05
        w1 = jax.random.normal(k3, (k, n), jnp.float32) * 0.05
        fused = interleave_gate_up(w0, w1, chunk)
        self.assertEqual(fused.shape, (k, 2 * n))
        # layout: chunk c of w0 then chunk c of w1
        np.testing.assert_array_equal(np.asarray(fused[:, 0:chunk]), np.asarray(w0[:, 0:chunk]))
        np.testing.assert_array_equal(
            np.asarray(fused[:, chunk : 2 * chunk]), np.asarray(w1[:, 0:chunk])
        )
        np.testing.assert_array_equal(
            np.asarray(fused[:, 2 * chunk : 3 * chunk]), np.asarray(w0[:, chunk : 2 * chunk])
        )
        # exact: apply the epilogue to the interleaved per-column products
        g = x @ w0
        u = x @ w1
        pre = interleave_gate_up(g, u, chunk)
        got = glu_from_interleaved(pre, chunk)
        ref = jax.nn.silu(g) * u
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        # end to end through one matmul on the interleaved weight (dot blocking may differ)
        got2 = glu_from_interleaved(x @ fused, chunk)
        np.testing.assert_allclose(np.asarray(got2), np.asarray(ref), rtol=1e-5, atol=1e-5)

    def test_interleave_handles_scale_layouts(self):
        e, n, chunk = 3, 512, 128
        s0 = jnp.arange(e * n, dtype=jnp.float32).reshape(e, 1, 1, n)
        s1 = -s0
        fs = interleave_gate_up(s0, s1, chunk)
        self.assertEqual(fs.shape, (e, 1, 1, 2 * n))
        np.testing.assert_array_equal(
            np.asarray(fs[1, 0, 0, chunk : 2 * chunk]), np.asarray(s1[1, 0, 0, :chunk])
        )
        bs0 = jnp.ones((e, 4, 1, n))
        self.assertEqual(interleave_gate_up(bs0, bs0, chunk).shape, (e, 4, 1, 2 * n))

    def test_validate_glu_chunk(self):
        validate_glu_chunk(tile_n=4096, size_n=4096, chunk=256)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=4096, size_n=4096, chunk=100)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=2048 + 256, size_n=4096, chunk=256)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=4096, size_n=4096 + 256, chunk=256)

    def test_auto_tiler_tile_n_snaps_to_glu_pairs(self):
        # Review repro (#1810): bf16 SiLU experts, MoE TP=1, hidden 6144, intermediate 2048 pass
        # the width check but the auto-tiler picks tile_n=1408 (3 n tiles) on v6e/v7x, which
        # validate_glu_chunk rejected before compilation. make_gmm_configs now snaps an
        # auto-selected tile_n down to whole chunk pairs; explicit TileSizes stay strict.
        from unittest import mock

        from jax._src.tpu_info import ChipVersion
        from jax.experimental.pallas import tpu as pltpu

        import sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 as gmm_v2_mod
        from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
            Dimensions,
            TileSizes,
            align_glu_tile_n,
            calculate_tiling,
            make_gmm_configs,
        )

        self.assertEqual(align_glu_tile_n(tile_n=1408, size_n=4096, chunk=256), 1024)
        self.assertEqual(align_glu_tile_n(tile_n=4096, size_n=4096, chunk=256), 4096)
        self.assertEqual(align_glu_tile_n(tile_n=512, size_n=4096, chunk=256), 512)
        self.assertEqual(align_glu_tile_n(tile_n=384, size_n=4096, chunk=256), 512)
        self.assertEqual(align_glu_tile_n(tile_n=1536, size_n=1024, chunk=256), 1024)

        S = jax.ShapeDtypeStruct
        g, k, n2 = 8, 6144, 2 * 2048
        # v7x: 64 MiB VMEM -> 3 n tiles of 1408; v6e: 128 MiB -> 2 tiles of 2048 (already aligned)
        for chip, raw_tile_n, glu_tile_n in (
            (ChipVersion.TPU_7X, 1408, 1024),
            (ChipVersion.TPU_V6E, 2048, 2048),
        ):
            info = pltpu.get_tpu_info_for_chip(chip, 1)
            vmem = int(info.vmem_capacity_bytes * 0.9)
            with (
                mock.patch.object(gmm_v2_mod.pltpu, "get_tpu_info", return_value=info),
                mock.patch(
                    "sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_tile_sizes.get_device_name",
                    return_value="TPU v7",
                ),
            ):
                dims = Dimensions(4096, k, n2, g, g, 16)
                raw = calculate_tiling(jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.bfloat16), dims, vmem)
                self.assertEqual(raw.tile_n, raw_tile_n)

                def cfg(rows, tile_info, dtype=jnp.bfloat16):
                    return make_gmm_configs(
                        S((rows, k), jnp.bfloat16),
                        S((g, k, n2), dtype),
                        None,
                        None,
                        S((g,), jnp.int32),
                        S((1,), jnp.int32),
                        tile_info=tile_info,
                        vmem_limit_bytes=vmem,
                        out_dtype=None,
                        acc_dtype=None,
                        maybe_quantize_lhs=False,
                        zero_initialize=False,
                        glu_chunk=256,
                    )

                for rows in (4096, 32):
                    c = cfg(rows, calculate_tiling)
                    self.assertEqual((c.tiles.tile_k, c.tiles.tile_n), (k, glu_tile_n))
                    self.assertEqual(c.out_tile_n, glu_tile_n // 2)
                with self.assertRaises(ValueError):
                    cfg(4096, TileSizes(128, k, 1408))
                # no glu_chunk: the auto-tiler's choice is untouched
                c0 = make_gmm_configs(
                    S((4096, k), jnp.bfloat16),
                    S((g, k, n2), jnp.bfloat16),
                    None,
                    None,
                    S((g,), jnp.int32),
                    S((1,), jnp.int32),
                    tile_info=calculate_tiling,
                    vmem_limit_bytes=vmem,
                    out_dtype=None,
                    acc_dtype=None,
                    maybe_quantize_lhs=False,
                    zero_initialize=False,
                )
                self.assertEqual(c0.tiles.tile_n, raw_tile_n)

    def test_interleave_under_explicit_sharding(self):
        # EP=4 / MoE TP=1: the size-1 "tensor" annotation on the last dim used to make the
        # reshape raise ShardingTypeError; EP=2 / TP=2 shards the interleaved dim for real.
        if len(jax.devices()) < 4:
            self.skipTest("needs 4 devices")
        e, k, n = 8, 64, 512
        rng = np.random.default_rng(0)
        w0 = rng.standard_normal((e, k, n)).astype(np.float32)
        w1 = rng.standard_normal((e, k, n)).astype(np.float32)
        s0 = rng.standard_normal((e, 1, 1, n)).astype(np.float32)
        s1 = rng.standard_normal((e, 1, 1, n)).astype(np.float32)
        for ep, tp, chunk in ((4, 1, 256), (2, 2, 128)):
            devs = np.array(jax.devices()[:4]).reshape(ep, tp)
            mesh = jax.sharding.Mesh(
                devs,
                ("expert", "tensor"),
                axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            )
            with jax.set_mesh(mesh):
                w0d = jax.device_put(w0, NamedSharding(mesh, P("expert", None, "tensor")))
                w1d = jax.device_put(w1, NamedSharding(mesh, P("expert", None, "tensor")))
                s0d = jax.device_put(s0, NamedSharding(mesh, P("expert", None, None, "tensor")))
                s1d = jax.device_put(s1, NamedSharding(mesh, P("expert", None, None, "tensor")))
                fw = interleave_gate_up(w0d, w1d, chunk)
                fs = interleave_gate_up(s0d, s1d, chunk)
            self.assertEqual(jax.typeof(fw).sharding.spec, P("expert", None, "tensor"), (ep, tp))
            self.assertEqual(
                jax.typeof(fs).sharding.spec, P("expert", None, None, "tensor"), (ep, tp)
            )
            np.testing.assert_array_equal(
                np.asarray(fw),
                np.asarray(interleave_gate_up(jnp.asarray(w0), jnp.asarray(w1), chunk)),
            )
            np.testing.assert_array_equal(
                np.asarray(fs),
                np.asarray(interleave_gate_up(jnp.asarray(s0), jnp.asarray(s1), chunk)),
            )

    def test_glu_epilogue_chunk_fallback(self):
        # per-device widths that are not a multiple of 256 (MoE TP=1 on 128- / 1408-wide
        # experts) and non-silu activations take the plain gate|up concat
        self.assertEqual(glu_epilogue_chunk("silu", 2048), 256)
        self.assertEqual(glu_epilogue_chunk("silu", 256), 256)
        self.assertEqual(glu_epilogue_chunk("silu", 128), 0)
        self.assertEqual(glu_epilogue_chunk("silu", 1408), 0)
        self.assertEqual(glu_epilogue_chunk("gelu", 2048), 0)

    def test_fused_glu_declines_integer_activation_quant(self):
        # chips with an int8 MXU quantize the lhs to int8 for int4 weights; the fused kernel's
        # tile matmul is the fp8 form, so the wrapper must hand back None (two-call path)
        from unittest import mock

        from jax._src.tpu_info import ChipVersion
        from jax.experimental.pallas import tpu as pltpu

        import sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 as gmm_v2_mod
        from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2_fused_glu import (
            fused_glu_down_configs,
        )

        S = jax.ShapeDtypeStruct
        g, k, i = 32, 6144, 2048

        def configs(chip):
            info = pltpu.get_tpu_info_for_chip(chip, 1)
            with mock.patch.object(gmm_v2_mod.pltpu, "get_tpu_info", return_value=info):
                return fused_glu_down_configs(
                    S((32, k), jnp.bfloat16),
                    S((g, k, 2 * i), jnp.int4),
                    S((g, 1, 1, 2 * i), jnp.float32),
                    S((g, i, k), jnp.int4),
                    S((g, 1, 1, k), jnp.float32),
                    S((256,), jnp.int32),
                    S((1,), jnp.int32),
                    glu_chunk=256,
                    preferred_element_type=jnp.bfloat16,
                    acc_dtype=jnp.float32,
                    zero_initialize=True,
                    vmem_limit_bytes=int(info.vmem_capacity_bytes * 0.9),
                    skip_prefix=True,
                    tile_m=32,
                )

        r = configs(ChipVersion.TPU_7X)
        self.assertIsNotNone(r)
        self.assertEqual(r[0].lhs_cfgs.quant_dtype, jnp.dtype(jnp.float8_e4m3fn))
        self.assertIsNone(configs(ChipVersion.TPU_V6E))

    def test_epilogue_chunk_constant(self):
        from sgl_jax.srt.layers import moe

        self.assertEqual(moe._GLU_EPILOGUE_CHUNK, 256)


if __name__ == "__main__":
    unittest.main()
