"""Tests for EmbedModelRunner._prepare_input (P1: assembles kwargs from mm_items).

jax is stubbed with numpy so _prepare_input runs without the model stack.
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


def _install_import_stubs():
    jax_stub = types.ModuleType("jax")
    jax_stub.Array = np.ndarray
    jax_stub.sharding = types.SimpleNamespace(Mesh=object)
    jax_numpy_stub = types.ModuleType("jax.numpy")
    jax_numpy_stub.asarray = np.asarray
    jax_numpy_stub.array = np.array
    jax_numpy_stub.zeros = np.zeros
    jax_numpy_stub.int32 = np.int32
    jax_numpy_stub.bool = np.bool_
    jax_numpy_stub.bfloat16 = np.float32
    jax_stub.numpy = jax_numpy_stub
    jax_stub.tree_util = types.SimpleNamespace(
        tree_flatten=lambda value: ([], None),
        tree_unflatten=lambda tree_def, leaves: None,
    )
    sys.modules.setdefault("jax", jax_stub)
    sys.modules.setdefault("jax.numpy", jax_numpy_stub)

    flax_stub = types.ModuleType("flax")
    nnx_stub = types.ModuleType("flax.nnx")
    nnx_stub.split = lambda model: (None, None)
    nnx_stub.merge = lambda model_def, model_state: None
    flax_stub.nnx = nnx_stub
    sys.modules.setdefault("flax", flax_stub)
    sys.modules.setdefault("flax.nnx", nnx_stub)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = type(
        "AutoConfig",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: SimpleNamespace())},
    )
    sys.modules.setdefault("transformers", transformers_stub)

    load_config_stub = types.ModuleType("sgl_jax.srt.configs.load_config")
    load_config_stub.LoadConfig = type("LoadConfig", (), {"__init__": lambda self, *a, **k: None})
    sys.modules.setdefault("sgl_jax.srt.configs.load_config", load_config_stub)

    base_runner_stub = types.ModuleType("sgl_jax.srt.model_executor.base_model_runner")
    base_runner_stub.BaseModelRunner = object
    sys.modules.setdefault("sgl_jax.srt.model_executor.base_model_runner", base_runner_stub)

    loader_stub = types.ModuleType("sgl_jax.srt.model_loader.loader")
    loader_stub.get_model_loader = lambda *args, **kwargs: None
    sys.modules.setdefault("sgl_jax.srt.model_loader.loader", loader_stub)

    server_args_stub = types.ModuleType("sgl_jax.srt.multimodal.common.ServerArgs")
    server_args_stub.MultimodalServerArgs = type("MultimodalServerArgs", (), {})
    sys.modules.setdefault("sgl_jax.srt.multimodal.common.ServerArgs", server_args_stub)

    modality_stub = types.ModuleType("sgl_jax.srt.multimodal.common.modality_enum")

    class _MultimodalDataItem:
        def __init__(self, modality=None, feature=None, model_specific_data=None, **kwargs):
            self.modality = modality
            self.feature = feature
            self.model_specific_data = model_specific_data or {}

        @classmethod
        def from_dict(cls, data):
            return cls(**data)

        def is_image(self):
            return self.modality == "image"

        def is_video(self):
            return self.modality == "video"

        def is_audio(self):
            return self.modality == "audio"

    modality_stub.MultimodalDataItem = _MultimodalDataItem
    sys.modules.setdefault("sgl_jax.srt.multimodal.common.modality_enum", modality_stub)

    schedule_batch_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.schedule_batch")
    schedule_batch_stub.Req = object
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.schedule_batch", schedule_batch_stub)


_install_import_stubs()

from sgl_jax.srt.multimodal.model_executor.embed.embed_model_runner import (  # noqa: E402
    EmbedModelRunner,
)


class _Item:
    """Duck-typed mm_item (assembler only uses these)."""

    def __init__(self, modality, feature, model_specific_data=None):
        self._m = modality
        self.feature = feature
        self.model_specific_data = model_specific_data or {}

    def is_image(self):
        return self._m == "image"

    def is_video(self):
        return self._m == "video"

    def is_audio(self):
        return self._m == "audio"


class TestEmbedModelRunnerPrepareInput(unittest.TestCase):
    audio_token_id = 151669

    def _runner(self):
        runner = EmbedModelRunner.__new__(EmbedModelRunner)
        runner.model_class = object
        runner.model_config = SimpleNamespace()
        return runner

    def _batch(self, input_ids, omni_inputs):
        return SimpleNamespace(
            input_ids=input_ids, origin_input_ids=input_ids, omni_inputs=omni_inputs
        )

    def test_audio_codes_item_routes_to_audio_codes_kwarg(self):
        item = _Item(
            "audio", np.zeros((8, 20), dtype=np.int32), {"is_codes": True, "token_lengths": [2]}
        )
        batch = self._batch(
            [1, self.audio_token_id, self.audio_token_id, 2],
            {"audio_token_id": self.audio_token_id, "mm_items": [item]},
        )
        out = self._runner()._prepare_input(batch)
        self.assertEqual(np.asarray(out["audio_codes"]).shape, (8, 20))
        self.assertIsNone(out["input_features"])
        self.assertEqual(
            np.asarray(out["input_ids"]).tolist(),
            [1, self.audio_token_id, self.audio_token_id, 2],
        )

    def test_audio_placeholder_mismatch_raises_before_embed(self):
        item = _Item(
            "audio", np.zeros((8, 20), dtype=np.int32), {"is_codes": True, "token_lengths": [2]}
        )
        batch = self._batch(
            [1, self.audio_token_id, 2],
            {"audio_token_id": self.audio_token_id, "mm_items": [item]},
        )

        with self.assertRaisesRegex(ValueError, "audio placeholder count mismatch"):
            self._runner()._prepare_input(batch)

    def test_audio_placeholder_mismatch_raises_for_transport_dict_item(self):
        item = {
            "modality": "audio",
            "feature": np.zeros((8, 20), dtype=np.int32),
            "model_specific_data": {"is_codes": True, "token_lengths": [2]},
        }
        batch = self._batch(
            [1, self.audio_token_id, 2],
            {"audio_token_id": self.audio_token_id, "mm_items": [item]},
        )

        with self.assertRaisesRegex(ValueError, "audio placeholder count mismatch"):
            self._runner()._prepare_input(batch)

    def test_audio_codes_without_audio_token_id_raises(self):
        # Audio codes present but no scatter target id: must fail loudly, not silently
        # drop the audio at the embedding scatter (review R2-6).
        item = _Item(
            "audio", np.zeros((8, 20), dtype=np.int32), {"is_codes": True, "token_lengths": [2]}
        )
        batch = self._batch([1, 2, 3], {"mm_items": [item]})  # no audio_token_id key
        with self.assertRaisesRegex(ValueError, "no audio_token_id"):
            self._runner()._prepare_input(batch)

    def test_text_only_no_mm_inputs(self):
        out = self._runner()._prepare_input(self._batch([1, 2, 3], None))
        self.assertIsNone(out["audio_codes"])
        self.assertIsNone(out["input_features"])
        self.assertIsNone(out["pixel_values"])
        self.assertEqual(np.asarray(out["input_ids"]).tolist(), [1, 2, 3])

    def test_image_item_routes_to_pixel_values(self):
        item = _Item("image", np.zeros((4, 8), dtype=np.float32))
        out = self._runner()._prepare_input(
            self._batch([1, 2], {"mm_items": [item], "image_grid_thw": [(1, 2, 2)]})
        )
        self.assertEqual(np.asarray(out["pixel_values"]).shape, (4, 8))
        self.assertIsNone(out["audio_codes"])
        self.assertIsNotNone(out["image_grid_thw"])

    def test_continuous_audio_densifies_with_mask(self):
        # continuous-audio (e.g. Qwen3-Omni mel): no is_codes flag -> input_features path
        item = _Item("audio", np.ones((1, 3, 2), dtype=np.float32))  # [B, n_mels, T]
        mm = {"mm_items": [item], "audio_feature_attention_mask": np.ones((1, 2), dtype=np.int32)}
        out = self._runner()._prepare_input(self._batch([1, 2], mm))
        self.assertIsNone(out["audio_codes"])
        self.assertIsNotNone(out["input_features"])
        self.assertIsNotNone(out["audio_feature_lengths"])


if __name__ == "__main__":
    unittest.main()
