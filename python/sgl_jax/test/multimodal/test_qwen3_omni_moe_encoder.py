import os
import unittest

import jax
import numpy as np
import torch
from flax import nnx
from jax import numpy as jnp
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

from sgl_jax.srt.models.qwen3_omni_moe.audio_encoder import (
    Qwen3OmniMoeAudioEncoder as JaxQwen3OmniMoeAudioEncoder,
)
from sgl_jax.srt.models.qwen3_omni_moe.weights_mapping import (
    create_audio_tower_weight_mappings,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.weight_utils import WeightLoader

# Ensure JAX uses float32 precision for matmul
jax.config.update("jax_default_matmul_precision", "highest")


class TestQwen3OmniMoeAudioEncoderPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"
        if not os.path.exists(model_path):
            raise unittest.SkipTest(f"Model path not found: {model_path}")

        cpu_devices = jax.devices("cpu")
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, len(cpu_devices)],
            dcn_parallelism=[1, 1],
            devices=cpu_devices[: len(cpu_devices)],
        )

        # Load float32 model
        model_fp32 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        cls.audio_tower_fp32: Qwen3OmniMoeAudioEncoder = model_fp32.thinker.audio_tower
        cls.audio_tower_fp32.eval()

        # Load bf16 model
        model_bf16 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        cls.audio_tower_bf16: Qwen3OmniMoeAudioEncoder = model_bf16.thinker.audio_tower
        cls.audio_tower_bf16.eval()

        # M6: build the JAX audio tower directly + load its `thinker.audio_tower.*` weights via the
        # shared create_audio_tower_weight_mappings builder (the same load path the in-model wrapper
        # uses), instead of the deleted staged Qwen3OmniMoeThinkerEmbedding scaffold. A tiny holder
        # gives the WeightLoader the `audio_tower.*` attribute the mappings target.
        def _build_jax_audio_tower(thinker_config, dtype):
            audio_config = thinker_config.audio_config
            thinker_config.model_path = model_path
            thinker_config.revision = None
            thinker_config.dtype = dtype

            class _AudioOnly(nnx.Module):
                def __init__(self):
                    self.mesh = cls.mesh
                    self.dtype = dtype
                    self.audio_tower = JaxQwen3OmniMoeAudioEncoder(
                        audio_config, mesh=cls.mesh, dtype=dtype, rngs=nnx.Rngs(0)
                    )

            with jax.set_mesh(cls.mesh):
                holder = _AudioOnly()
            loader = WeightLoader(
                model=holder, model_config=thinker_config, mesh=cls.mesh, dtype=dtype
            )
            loader.load_weights_from_safetensors(create_audio_tower_weight_mappings(audio_config))
            return holder.audio_tower

        cls.jax_audio_tower_fp32 = _build_jax_audio_tower(model_fp32.thinker.config, jnp.float32)
        cls.jax_audio_tower_bf16 = _build_jax_audio_tower(model_bf16.thinker.config, jnp.bfloat16)

    def _test_and_compare_audio_tower(
        self, input_features_np, feature_lens_np, dtype_name="float32"
    ):
        """Compare Torch and JAX outputs in different precisions

        Args:
            input_features_np: Input features as numpy array
            feature_lens_np: Feature lengths as numpy array
            dtype_name: Precision type, "float32" or "bf16"
        """
        if dtype_name == "float32":
            jax_dtype = jnp.float32
            torch_dtype = torch.float32
            atol = 1e-5
            rtol = 1e-3
            jax_audio_tower = self.jax_audio_tower_fp32
            torch_audio_tower = self.audio_tower_fp32
        elif dtype_name == "bf16":
            jax_dtype = jnp.bfloat16
            torch_dtype = torch.bfloat16
            atol = 1e-2  # bf16 has lower precision, relax error tolerance
            rtol = 1e-1
            jax_audio_tower = self.jax_audio_tower_bf16
            torch_audio_tower = self.audio_tower_bf16
        else:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        # JAX
        input_features_jax = jnp.array(input_features_np, dtype=jax_dtype)
        feature_lens_jax = jnp.array(feature_lens_np, dtype=jnp.int32)

        jax_output = jax_audio_tower(
            input_features=input_features_jax, feature_lens=feature_lens_jax
        )

        # PyTorch
        input_features_torch = torch.from_numpy(input_features_np).to(dtype=torch_dtype)
        feature_lens_torch = torch.from_numpy(feature_lens_np).to(dtype=torch.long)

        torch_output = torch_audio_tower(
            input_features=input_features_torch, feature_lens=feature_lens_torch
        ).last_hidden_state

        torch_output_np = torch_output.detach().cpu().to(torch.float32).numpy()
        jax_output_np = np.array(jax_output).astype(np.float32)

        self.assertEqual(
            torch_output_np.shape,
            jax_output_np.shape,
            f"[{dtype_name}] Output shapes don't match: torch {torch_output_np.shape} vs jax {jax_output_np.shape}",
        )

        np.testing.assert_allclose(
            torch_output_np,
            jax_output_np,
            atol=atol,
            rtol=rtol,
            err_msg=f"[{dtype_name}] Outputs don't match within tolerance",
        )

    def test_random_input_audio(self):
        input_features_np = np.random.randn(128, 666).astype(np.float32)
        feature_lens_np = np.array([666], dtype=np.int64)

        # Test float32 precision
        self._test_and_compare_audio_tower(input_features_np, feature_lens_np, dtype_name="float32")

    def test_random_input_audio_2(self):
        input_features_np = np.random.randn(128, 1791).astype(np.float32)
        feature_lens_np = np.array([1500, 291], dtype=np.int64)

        # Test float32 precision
        self._test_and_compare_audio_tower(input_features_np, feature_lens_np, dtype_name="float32")

    def test_random_input_audio_3(self):
        input_features_np = np.random.randn(128, 2005).astype(np.float32)
        feature_lens_np = np.array([1001, 596, 408], dtype=np.int64)

        # Test float32 precision
        self._test_and_compare_audio_tower(input_features_np, feature_lens_np, dtype_name="float32")


if __name__ == "__main__":
    unittest.main()
