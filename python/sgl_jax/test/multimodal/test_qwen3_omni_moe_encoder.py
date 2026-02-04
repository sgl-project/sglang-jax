import unittest

import jax
import numpy as np
import torch
from jax import numpy as jnp
from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_loader import get_model_loader
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_moe_encoder import (
    Qwen3OmniMoeThinkerEmbedding,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class TestQwen3OmniMoeAudioEncoderPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"

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

        model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=Qwen3OmniMoeThinkerEmbedding,
            ),
            mesh=cls.mesh,
        )

        # JAX float32 model
        model_config_fp32 = model_fp32.thinker.config
        model_config_fp32.model_path = model_path
        model_config_fp32.revision = None
        model_config_fp32.dtype = jnp.float32
        model_config_fp32.model_class = Qwen3OmniMoeThinkerEmbedding
        jax_model_fp32 = model_loader.load_model(
            model_config=model_config_fp32,
        )
        cls.jax_audio_tower_fp32 = jax_model_fp32.audio_tower

        # JAX bf16 model
        model_config_bf16 = model_bf16.thinker.config
        model_config_bf16.model_path = model_path
        model_config_bf16.revision = None
        model_config_bf16.dtype = jnp.bfloat16
        model_config_bf16.model_class = Qwen3OmniMoeThinkerEmbedding
        jax_model_bf16 = model_loader.load_model(
            model_config=model_config_bf16,
        )
        cls.jax_audio_tower_bf16 = jax_model_bf16.audio_tower

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
        feature_lens_jax = jnp.array(feature_lens_np, dtype=jnp.int64)

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


class TestQwen3OmniMoeThinkerEmbeddingPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"

        cpu_devices = jax.devices("cpu")
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, len(cpu_devices)],
            dcn_parallelism=[1, 1],
            devices=cpu_devices[: len(cpu_devices)],
        )

        model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=Qwen3OmniMoeThinkerEmbedding,
            ),
            mesh=cls.mesh,
        )

        # JAX float32 model
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_config_fp32 = config.thinker_config
        model_config_fp32.model_path = model_path
        model_config_fp32.revision = None
        model_config_fp32.dtype = jnp.float32
        model_config_fp32.model_class = Qwen3OmniMoeThinkerEmbedding
        jax_model_fp32 = model_loader.load_model(
            model_config=model_config_fp32,
        )
        cls.jax_model_fp32 = jax_model_fp32

    def _test_model(self, data_path: str):
        data = np.load(data_path)
        forward_batch = ForwardBatch(
            bid=0,
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=jnp.array(data["input_ids"]),
            req_pool_indices=None,
            seq_lens=None,
            out_cache_loc=None,
        )

        input_features = data.get("input_features")
        audio_feature_lengths = data.get("audio_feature_lengths")
        pixel_values = data.get("pixel_values")
        pixel_values_videos = data.get("pixel_values_videos")
        image_grid_thw = data.get("image_grid_thw")
        video_grid_thw = data.get("video_grid_thw")

        jax_output = self.jax_model_fp32(
            forward_batch=forward_batch,
            input_features=jnp.array(input_features) if input_features is not None else None,
            audio_feature_lengths=(
                jnp.array(audio_feature_lengths) if audio_feature_lengths is not None else None
            ),
            pixel_values=jnp.array(pixel_values) if pixel_values is not None else None,
            pixel_values_videos=(
                jnp.array(pixel_values_videos) if pixel_values_videos is not None else None
            ),
            image_grid_thw=jnp.array(image_grid_thw) if image_grid_thw is not None else None,
            video_grid_thw=jnp.array(video_grid_thw) if video_grid_thw is not None else None,
        )

        jax_input_embeds, jax_visual_embeds_multiscale, jax_visual_pos_masks = jax_output
        torch_input_embeds = data.get("input_embeds")
        torch_visual_pos_masks = data.get("visual_pos_masks")

        num_multiscale = sum(1 for key in data if key.startswith("visual_embeds_multiscale_"))
        torch_visual_embeds_multiscale = tuple(
            data[f"visual_embeds_multiscale_{i}"] for i in range(num_multiscale)
        )
        data.close()

        np.testing.assert_allclose(
            np.array(jax_input_embeds).astype(np.float32), torch_input_embeds, rtol=1e-3, atol=1e-5
        )
        if torch_visual_pos_masks is not None:
            np.testing.assert_allclose(
                np.array(jax_visual_pos_masks).astype(np.float32),
                torch_visual_pos_masks,
                rtol=1e-3,
                atol=1e-5,
            )

        if num_multiscale > 0:
            for i in range(num_multiscale):
                np.testing.assert_allclose(
                    np.array(jax_visual_embeds_multiscale[i]).astype(np.float32),
                    torch_visual_embeds_multiscale[i],
                    rtol=1e-3,
                    atol=1e-5,
                )

    def test_omni_modal_input(self):
        self._test_model("/models/npz_data/qwen3_omni/omni_encoder_ref.npz")

    def test_audio_modal_input(self):
        self._test_model("/models/npz_data/qwen3_omni/audio_encoder_ref.npz")


if __name__ == "__main__":
    unittest.main()
