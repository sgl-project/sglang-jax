import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.moe import FusedEPMoE
from sgl_jax.srt.utils.quantization.quantization_utils import dequantize_tensor, quantize_tensor


def _make_mesh():
    devices = np.array(jax.devices())
    if devices.size < 1:
        raise unittest.SkipTest("No JAX devices available")
    # FusedEPMoE initializes weights with PartitionSpec(('data', 'tensor'), ...).
    return jax.sharding.Mesh(
        devices[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _make_quant_config():
    return QuantizationConfig(
        linear_rules=[{"module_path": ".*", "weight_dtype": "float8_e4m3fn"}],
        moe_weight_dtype=jnp.float8_e4m3fn,
        moe_activation_dtype=None,
        is_static_checkpoint=True,
        weight_block_size=[128, 128],
    )


class TestFusedMoeStaticQuantAdapter(unittest.TestCase):
    def _build_moe_with_static_128_weights(self):
        mesh = _make_mesh()
        quant_config = _make_quant_config()

        with jax.set_mesh(mesh):
            moe = FusedEPMoE(
                hidden_size=256,
                num_experts=2,
                num_experts_per_tok=1,
                intermediate_dim=512,
                ep_size=1,
                mesh=mesh,
                quantization_config=quant_config,
            )
            moe.quantize_weights(is_static=True)

            w1 = jax.random.normal(jax.random.key(0), (2, 256, 512), dtype=jnp.float32)
            w2 = jax.random.normal(jax.random.key(1), (2, 512, 256), dtype=jnp.float32)
            w3 = jax.random.normal(jax.random.key(2), (2, 256, 512), dtype=jnp.float32)

            w1_q, w1_s = quantize_tensor(jnp.float8_e4m3fn, w1, axis=1, block_size=128)
            w2_q, w2_s = quantize_tensor(jnp.float8_e4m3fn, w2, axis=1, block_size=128)
            w3_q, w3_s = quantize_tensor(jnp.float8_e4m3fn, w3, axis=1, block_size=128)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                moe.w1.value = w1_q
                moe.w2.value = w2_q
                moe.w3.value = w3_q
                moe.w1_scale.value = w1_s.reshape(w1_s.shape[0], w1_s.shape[1], 1, w1_s.shape[2])
                moe.w2_scale.value = w2_s.reshape(w2_s.shape[0], w2_s.shape[1], 1, w2_s.shape[2])
                moe.w3_scale.value = w3_s.reshape(w3_s.shape[0], w3_s.shape[1], 1, w3_s.shape[2])
                moe.subc_quant_wsz = 128

        return mesh, moe

    def _dequant_weights(self, moe):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            w1 = dequantize_tensor(
                moe.w1.value,
                jnp.squeeze(moe.w1_scale.value, axis=2),
                axis=1,
                out_dtype=jnp.float32,
            )
            w2 = dequantize_tensor(
                moe.w2.value,
                jnp.squeeze(moe.w2_scale.value, axis=2),
                axis=1,
                out_dtype=jnp.float32,
            )
            w3 = dequantize_tensor(
                moe.w3.value,
                jnp.squeeze(moe.w3_scale.value, axis=2),
                axis=1,
                out_dtype=jnp.float32,
            )
        return w1, w2, w3

    def test_prepare_static_block_quant_for_fused_kernel_updates_shapes(self):
        mesh, moe = self._build_moe_with_static_128_weights()

        with jax.set_mesh(mesh):
            ok = moe.prepare_static_block_quant_for_fused_kernel(target_subc_quant_wsz=256)
            self.assertTrue(ok)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.assertEqual(moe.subc_quant_wsz, 256)
            self.assertEqual(moe.w1_scale.value.shape, (2, 1, 1, 512))
            self.assertEqual(moe.w2_scale.value.shape, (2, 2, 1, 256))
            self.assertEqual(moe.w3_scale.value.shape, (2, 1, 1, 512))

        # Idempotent after one successful adaptation.
        with jax.set_mesh(mesh):
            self.assertFalse(moe.prepare_static_block_quant_for_fused_kernel(target_subc_quant_wsz=256))

    def test_prepare_static_block_quant_for_fused_kernel_preserves_dequantized_weights(self):
        mesh, moe = self._build_moe_with_static_128_weights()

        pre = self._dequant_weights(moe)

        with jax.set_mesh(mesh):
            self.assertTrue(moe.prepare_static_block_quant_for_fused_kernel(target_subc_quant_wsz=256))

        post = self._dequant_weights(moe)

        # This path requantizes (dequantize -> requantize), so it is not exact, but
        # it should preserve the decoded weights to within normal quantization error.
        for name, before, after in zip(("w1", "w2", "w3"), pre, post, strict=True):
            diff = jnp.abs(before - after)
            self.assertLess(
                float(jnp.mean(diff)),
                0.03,
                msg=f"{name} mean abs diff too large after static 128->256 requant",
            )
            self.assertLess(
                float(jnp.max(diff)),
                0.5,
                msg=f"{name} max abs diff too large after static 128->256 requant",
            )


if __name__ == "__main__":
    unittest.main()
