import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.dtype_config import DtypeConfig
from sgl_jax.srt.models.llama import LlamaMLP
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


class TestDtypeConfigConsistency(CustomTestCase):
    def test_mlp_consistency(self):
        mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        jax.sharding.set_mesh(mesh)

        hidden_size = 256
        intermediate_size = 512

        def cast_params(params, target_dtype, float32_paths=None):
            float32_paths = float32_paths or ()

            def _cast(tree, path=()):
                if hasattr(tree, "items"):
                    return type(tree)({k: _cast(v, path + (k,)) for k, v in tree.items()})

                if isinstance(tree, (list, tuple)):
                    return type(tree)(_cast(v, path + (str(i),)) for i, v in enumerate(tree))

                if hasattr(tree, "astype"):
                    name = "/".join(path)
                    if any(p in name for p in float32_paths):
                        return tree.astype(jnp.float32)
                    return tree.astype(target_dtype)

                return tree

            return _cast(params)

        # Create fp32 model and reuse its weights for the bf16 and mixed-precision models.
        model_fp32 = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=jnp.float32,
            mesh=mesh,
        )

        params_fp32 = model_fp32.params

        model_bf16 = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=jnp.bfloat16,
            mesh=mesh,
        )
        model_bf16.params = cast_params(params_fp32, jnp.bfloat16)

        dtype_config = DtypeConfig(
            default_dtype=jnp.bfloat16,
            config_dict={
                "gate_proj": "float32",
            },
        )
        model_mixed = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype_config=dtype_config,
            mesh=mesh,
        )
        model_mixed.params = cast_params(
            params_fp32,
            jnp.bfloat16,
            float32_paths=("gate_proj",),
        )

        # Generate dummy input
        x = jax.random.uniform(jax.random.PRNGKey(0), (1, hidden_size), dtype=jnp.float32)

        # Run forward passes
        out_fp32 = model_fp32(x)
        out_bf16 = model_bf16(x.astype(jnp.bfloat16))
        out_mixed = model_mixed(x.astype(jnp.bfloat16))

        # Compare outputs
        out_fp32 = np.array(out_fp32)
        out_bf16 = np.array(out_bf16.astype(jnp.float32))
        out_mixed = np.array(out_mixed.astype(jnp.float32))

        diff_fp32_bf16 = np.max(np.abs(out_fp32 - out_bf16))
        diff_bf16_mixed = np.max(np.abs(out_bf16 - out_mixed))
        diff_fp32_mixed = np.max(np.abs(out_fp32 - out_mixed))

        print(f"diff_fp32_bf16: {diff_fp32_bf16}")
        print(f"diff_bf16_mixed: {diff_bf16_mixed}")
        print(f"diff_fp32_mixed: {diff_fp32_mixed}")

        # The output of mixed precision should be between fp32 and bf16.
        self.assertGreater(diff_fp32_bf16, diff_bf16_mixed)
        self.assertGreater(diff_fp32_bf16, diff_fp32_mixed)


if __name__ == "__main__":
    unittest.main()
