from __future__ import annotations

import asyncio
import dataclasses
import sys
import types
import unittest
from enum import Enum, auto
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
)


def _install_import_stubs():
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.Request = type("Request", (), {})
    sys.modules.setdefault("fastapi", fastapi_stub)

    imageio_stub = types.ModuleType("imageio")
    imageio_v3_stub = types.ModuleType("imageio.v3")
    sys.modules.setdefault("imageio", imageio_stub)
    sys.modules.setdefault("imageio.v3", imageio_v3_stub)

    librosa_stub = types.ModuleType("librosa")
    librosa_stub.load = lambda *args, **kwargs: (
        np.zeros(1, dtype=np.float32),
        kwargs.get("sr", 24000),
    )
    sys.modules.setdefault("librosa", librosa_stub)

    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_image_stub.Image = type("Image", (), {})
    pil_image_stub.open = lambda *args, **kwargs: pil_image_stub.Image()
    pil_stub.Image = pil_image_stub
    sys.modules.setdefault("PIL", pil_stub)
    sys.modules.setdefault("PIL.Image", pil_image_stub)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = type("AutoConfig", (), {})
    transformers_stub.AutoProcessor = type("AutoProcessor", (), {})
    sys.modules.setdefault("transformers", transformers_stub)

    managers_io_stub = types.ModuleType("sgl_jax.srt.managers.io_struct")
    for name in (
        "AbortReq",
        "BatchEmbeddingOut",
        "BatchStrOut",
        "BatchTokenIDOut",
        "ProfileReqOutput",
    ):
        setattr(managers_io_stub, name, type(name, (), {}))
    sys.modules.setdefault("sgl_jax.srt.managers.io_struct", managers_io_stub)

    mm_io_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.io_struct")

    class _DataType(Enum):
        IMAGE = auto()
        VIDEO = auto()
        AUDIO = auto()

    @dataclasses.dataclass
    class _GenerateMMReqInput:
        input_ids: list[int] | None = None
        prompt: str | None = None
        text: str | None = None
        image_data: list[str] | str | None = None
        video_data: list[str] | str | None = None
        audio_data: list[str] | str | None = None
        input_reference: str | None = None
        data_type: _DataType | None = None
        sampling_params: dict | None = None

    @dataclasses.dataclass
    class _GenerateOmniReqInput:
        rid: str | None = None
        prompt: str | None = None
        input_ids: list[int] | None = None
        image_data: list[str] | str | None = None
        video_data: list[str] | str | None = None
        audio_data: list[str] | str | None = None
        stream: bool = False
        n: int | None = 1
        sampling_params: dict | None = None
        stop: str | list[str] | None = None

    @dataclasses.dataclass
    class _TokenizedGenerateMMReqInput:
        rid: str | None = None
        input_ids: list[int] | None = None

    @dataclasses.dataclass
    class _TokenizedGenerateOmniReqInput:
        rid: str | None = None
        prompt: str | None = None
        input_ids: list[int] | None = None
        mm_inputs: dict | None = None
        stream: bool = False
        n: int | None = 1
        sampling_params: dict | None = None
        stop: str | list[str] | None = None

    for name in (
        "AudioSpeechRequest",
        "AudioTranscriptionRequest",
        "AudioTranscriptionResponse",
    ):
        setattr(mm_io_stub, name, type(name, (), {}))
    mm_io_stub.DataType = _DataType
    mm_io_stub.GenerateMMReqInput = _GenerateMMReqInput
    mm_io_stub.GenerateOmniReqInput = _GenerateOmniReqInput
    mm_io_stub.TokenizedGenerateMMReqInput = _TokenizedGenerateMMReqInput
    mm_io_stub.TokenizedGenerateOmniReqInput = _TokenizedGenerateOmniReqInput
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.io_struct", mm_io_stub)

    tokenizer_manager_stub = types.ModuleType("sgl_jax.srt.managers.tokenizer_manager")
    tokenizer_manager_stub.ReqState = type("ReqState", (), {})
    tokenizer_manager_stub.TokenizerManager = type("TokenizerManager", (), {})
    sys.modules.setdefault("sgl_jax.srt.managers.tokenizer_manager", tokenizer_manager_stub)

    modality_stub = types.ModuleType("sgl_jax.srt.multimodal.common.modality_enum")

    class _Modality:
        IMAGE = "image"
        VIDEO = "video"
        AUDIO = "audio"

    class _MultimodalDataItem:
        def __init__(self, modality=None, feature=None, offsets=None, model_specific_data=None):
            self.modality = modality
            self.feature = feature
            self.offsets = offsets
            self.model_specific_data = model_specific_data or {}
            self.pad_value = None

        def is_image(self):
            return self.modality == "image"

        def is_video(self):
            return self.modality == "video"

        def is_audio(self):
            return self.modality == "audio"

        def set_pad_value(self):
            self.pad_value = 1

    modality_stub.Modality = _Modality
    modality_stub.MultimodalDataItem = _MultimodalDataItem
    sys.modules.setdefault("sgl_jax.srt.multimodal.common.modality_enum", modality_stub)

    mrope_utils_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.mrope_utils")
    mrope_utils_stub.compute_mrope_positions = lambda **kwargs: (None, None)
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.mrope_utils", mrope_utils_stub)

    prompt_builder_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.prompt_builder")
    prompt_builder_stub.MultimodalPromptBuilder = type(
        "MultimodalPromptBuilder",
        (),
        {"__init__": lambda self, tokenizer=None: None},
    )
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.prompt_builder", prompt_builder_stub)

    server_args_stub = types.ModuleType("sgl_jax.srt.server_args")
    server_args_stub.PortArgs = type("PortArgs", (), {})
    server_args_stub.ServerArgs = type("ServerArgs", (), {})
    sys.modules.setdefault("sgl_jax.srt.server_args", server_args_stub)

    srt_utils_stub = types.ModuleType("sgl_jax.srt.utils")
    srt_utils_stub.configure_logger = lambda *args, **kwargs: None
    srt_utils_stub.dataclass_to_string_truncated = lambda obj, *args, **kwargs: str(obj)
    srt_utils_stub.kill_itself_when_parent_died = lambda *args, **kwargs: None
    sys.modules.setdefault("sgl_jax.srt.utils", srt_utils_stub)

    utils_stub = types.ModuleType("sgl_jax.utils")
    utils_stub.TypeBasedDispatcher = type(
        "TypeBasedDispatcher", (), {"__init__": lambda self, *a, **k: None}
    )
    utils_stub.get_exception_traceback = lambda: ""
    sys.modules.setdefault("sgl_jax.utils", utils_stub)


_install_import_stubs()

from sgl_jax.srt.multimodal.manager.io_struct import GenerateOmniReqInput  # noqa: E402
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import (  # noqa: E402
    MultimodalTokenizer,
)


class _FakeCodec:
    def __init__(self, audio_token_id):
        self.audio_token_id = audio_token_id
        self.encode_calls = []

    def encode(self, audio_data):
        self.encode_calls.append(audio_data)
        return MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((5, 20), dtype=np.int32),
            audio_token_id=self.audio_token_id,
            source="fake_raw_audio",
        )


class _FakeHF:
    """Vision-only HF processor stand-in: returns fixed input_ids, no audio."""

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __call__(self, images=None, videos=None, text="", return_tensors=None, **kwargs):
        return {"input_ids": np.asarray([self.input_ids], dtype=np.int64)}


def _audio_item(mm_inputs):
    for item in mm_inputs["mm_items"]:
        if item.is_audio():
            return item
    return None


class TestMiMoV25MultimodalTokenizerAudio(unittest.TestCase):
    audio_token_id = 151669

    def _new_tokenizer(self, *, is_mimo_v25=True, hf_input_ids=None):
        from sgl_jax.srt.multimodal.models.mimo_v2_5.processor import MiMoV25Processor

        tokenizer = MultimodalTokenizer.__new__(MultimodalTokenizer)
        tokenizer.server_args = SimpleNamespace(
            model_path="local/MiMo-V2.5" if is_mimo_v25 else "local/other",
            trust_remote_code=True,
        )
        tokenizer.mm_config = SimpleNamespace(
            model_type="mimo_v2_5" if is_mimo_v25 else "qwen3_omni",
            audio_token_id=self.audio_token_id,
            audio_config={"audio_channels": 20, "group_size": 4, "speech_vocab_size": 1280},
        )
        tokenizer.tokenizer = None
        if is_mimo_v25:
            tokenizer.mm_processor = MiMoV25Processor(
                "local/MiMo-V2.5",
                audio_token_id=self.audio_token_id,
                hf_processor=_FakeHF(hf_input_ids or [1, self.audio_token_id, 2]),
                codec=_FakeCodec(self.audio_token_id),
            )
        else:
            tokenizer.mm_processor = None
        return tokenizer

    def test_mimo_v25_raw_audio_yields_audio_codes_mm_item(self):
        tokenizer = self._new_tokenizer(is_mimo_v25=True)
        request = GenerateOmniReqInput(
            input_ids=[1, self.audio_token_id, 2],
            audio_data=["raw-audio"],
        )
        tokenized = asyncio.run(tokenizer._tokenize_one_request(request))

        self.assertEqual(tokenizer.mm_processor._codec.encode_calls, [["raw-audio"]])
        self.assertEqual(tokenized.input_ids, [1, self.audio_token_id, self.audio_token_id, 2])
        # audio rides as a first-class mm_item carrying codes + meta (no payload side-channel)
        self.assertNotIn("mimo_v25_audio_payload", tokenized.mm_inputs)
        item = _audio_item(tokenized.mm_inputs)
        self.assertIsNotNone(item)
        self.assertEqual(np.asarray(item.feature).shape, (8, 20))
        self.assertTrue(item.model_specific_data["is_codes"])
        self.assertEqual(item.model_specific_data["token_lengths"], [2])
        self.assertEqual(item.model_specific_data["group_size"], 4)

    def test_mimo_v25_audio_without_placeholder_raises(self):
        tokenizer = self._new_tokenizer(is_mimo_v25=True, hf_input_ids=[1, 2, 3])
        request = GenerateOmniReqInput(input_ids=[1, 2, 3], audio_data=["raw-audio"])
        with self.assertRaises(ValueError):
            asyncio.run(tokenizer._tokenize_one_request(request))

    def test_non_mimo_v25_audio_still_requires_mm_processor(self):
        tokenizer = self._new_tokenizer(is_mimo_v25=False)
        request = GenerateOmniReqInput(input_ids=[1, 2, 3], audio_data=["raw-audio"])
        with self.assertRaisesRegex(ValueError, "processor/config"):
            asyncio.run(tokenizer._tokenize_one_request(request))


if __name__ == "__main__":
    unittest.main()
