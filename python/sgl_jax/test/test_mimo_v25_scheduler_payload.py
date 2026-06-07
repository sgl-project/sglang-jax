from __future__ import annotations

import dataclasses
import sys
import types
import unittest

import numpy as np


def _install_import_stubs():
    jax_stub = types.ModuleType("jax")
    jax_stub.Array = np.ndarray
    jax_numpy_stub = types.ModuleType("jax.numpy")
    for name in ("array", "asarray", "concatenate", "full", "stack", "zeros"):
        setattr(jax_numpy_stub, name, getattr(np, name))
    jax_numpy_stub.int32 = np.int32
    jax_stub.numpy = jax_numpy_stub
    sys.modules.setdefault("jax", jax_stub)
    sys.modules.setdefault("jax.numpy", jax_numpy_stub)

    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_image_stub.Image = type("Image", (), {})
    pil_stub.Image = pil_image_stub
    sys.modules.setdefault("PIL", pil_stub)
    sys.modules.setdefault("PIL.Image", pil_image_stub)

    @dataclasses.dataclass
    class _BatchTokenIDOut:
        rids: list[str] | None = None
        output_hidden_states_for_mm: list | None = None

    @dataclasses.dataclass
    class _TokenizedGenerateReqInput:
        rid: str | None = None
        input_ids: list[int] | None = None
        sampling_params: object | None = None
        stream: bool = False
        return_hidden_states: bool = False
        mm_inputs: dict | None = None

    managers_io_stub = types.ModuleType("sgl_jax.srt.managers.io_struct")
    managers_io_stub.BatchTokenIDOut = _BatchTokenIDOut
    managers_io_stub.TokenizedGenerateReqInput = _TokenizedGenerateReqInput
    for name in (
        "AbortReq",
        "BatchStrOut",
        "ProfileReq",
        "ProfileReqOutput",
    ):
        setattr(managers_io_stub, name, type(name, (), {}))
    sys.modules.setdefault("sgl_jax.srt.managers.io_struct", managers_io_stub)

    sampling_stub = types.ModuleType("sgl_jax.srt.sampling.sampling_params")

    class _SamplingParams:
        def __init__(self, max_new_tokens=128, stop=None, **kwargs):
            self.max_new_tokens = max_new_tokens
            self.stop_strs = stop
            for key, value in kwargs.items():
                setattr(self, key, value)

    sampling_stub.SamplingParams = _SamplingParams
    sys.modules.setdefault("sgl_jax.srt.sampling.sampling_params", sampling_stub)

    mm_io_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.io_struct")

    @dataclasses.dataclass
    class _TokenizedGenerateOmniReqInput:
        rid: str
        prompt: str | None = None
        input_ids: list[int] | None = None
        mm_inputs: dict | None = None
        stream: bool = False
        n: int | None = 1
        sampling_params: dict | None = None
        stop: str | list[str] | None = None

    mm_io_stub.DataType = type("DataType", (), {})
    mm_io_stub.OmniInputs = dict
    mm_io_stub.TokenizedGenerateOmniReqInput = _TokenizedGenerateOmniReqInput
    mm_io_stub.TokenizedGenerateMMReqInput = type("TokenizedGenerateMMReqInput", (), {})
    mm_io_stub.TokenizedGenerateAudioReqInput = type("TokenizedGenerateAudioReqInput", (), {})
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.io_struct", mm_io_stub)

    modality_stub = types.ModuleType("sgl_jax.srt.multimodal.common.modality_enum")

    class _MultimodalDataItem:
        @classmethod
        def from_dict(cls, item):
            return cls()

        def is_image(self):
            return False

        def is_video(self):
            return False

        def is_audio(self):
            return False

    modality_stub.MultimodalDataItem = _MultimodalDataItem
    modality_stub.pad_input_tokens = lambda input_ids, **kwargs: list(input_ids)
    sys.modules.setdefault("sgl_jax.srt.multimodal.common.modality_enum", modality_stub)

    for module_name in (
        "psutil",
        "setproctitle",
        "zmq",
        "sgl_jax.srt.multimodal.manager.device_manager",
        "sgl_jax.srt.multimodal.manager.stage",
        "sgl_jax.srt.multimodal.manager.utils",
        "sgl_jax.srt.multimodal.models.static_configs",
        "sgl_jax.srt.server_args",
        "sgl_jax.srt.utils",
        "sgl_jax.srt.utils.common_utils",
        "sgl_jax.utils",
    ):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    sys.modules["psutil"].Process = lambda *args, **kwargs: None
    sys.modules["setproctitle"].setproctitle = lambda *args, **kwargs: None
    sys.modules["zmq"].Context = lambda *args, **kwargs: None
    sys.modules["zmq"].PULL = object()
    sys.modules["zmq"].PUSH = object()
    sys.modules["zmq"].POLLIN = object()
    sys.modules["zmq"].NOBLOCK = object()
    sys.modules["zmq"].ZMQError = Exception
    sys.modules["sgl_jax.srt.multimodal.manager.device_manager"].DeviceManager = type(
        "DeviceManager", (), {}
    )
    sys.modules["sgl_jax.srt.multimodal.manager.stage"].Stage = type("Stage", (), {})
    sys.modules["sgl_jax.srt.multimodal.manager.utils"].load_stage_configs_from_yaml = (
        lambda *args, **kwargs: []
    )
    sys.modules["sgl_jax.srt.multimodal.models.static_configs"].get_stage_config_path = (
        lambda *args, **kwargs: ""
    )
    sys.modules["sgl_jax.srt.server_args"].PortArgs = type("PortArgs", (), {})
    sys.modules["sgl_jax.srt.server_args"].ServerArgs = type("ServerArgs", (), {})
    sys.modules["sgl_jax.srt.utils"].configure_logger = lambda *args, **kwargs: None
    sys.modules["sgl_jax.srt.utils"].kill_itself_when_parent_died = lambda *args, **kwargs: None
    sys.modules["sgl_jax.srt.utils"].get_zmq_socket = lambda *args, **kwargs: None
    sys.modules["sgl_jax.srt.utils.common_utils"].get_zmq_socket = lambda *args, **kwargs: None
    sys.modules["sgl_jax.utils"].TypeBasedDispatcher = type(
        "TypeBasedDispatcher", (), {"__init__": lambda self, *args, **kwargs: None}
    )
    sys.modules["sgl_jax.utils"].get_exception_traceback = lambda: ""


_install_import_stubs()

from sgl_jax.srt.multimodal.manager.global_scheduler import (  # noqa: E402
    GlobalScheduler,
)
from sgl_jax.srt.multimodal.manager.io_struct import (  # noqa: E402
    TokenizedGenerateOmniReqInput,
)


class TestMiMoV25SchedulerConvert(unittest.TestCase):
    audio_token_id = 151669

    def _scheduler(self):
        scheduler = GlobalScheduler.__new__(GlobalScheduler)
        scheduler.req_store = {}
        return scheduler

    def test_convert_carries_mm_inputs_and_ar_params(self):
        # P1: scheduler only transports mm_inputs (incl mm_items) + orchestration params;
        # it does NOT normalize payloads or flatten features into req fields.
        mm_inputs = {"audio_token_id": self.audio_token_id, "mm_items": []}
        tokenized = TokenizedGenerateOmniReqInput(
            rid="rid-1",
            prompt="prompt",
            input_ids=[10, self.audio_token_id, self.audio_token_id, 11],
            mm_inputs=mm_inputs,
            sampling_params={"max_new_tokens": 7, "temperature": 0.3},
            stop=["done"],
            stream=True,
        )
        req = self._scheduler().convert_omni_request(tokenized)

        self.assertEqual(req.rid, "rid-1")
        self.assertIs(req.omni_inputs, mm_inputs)  # carried as-is, single source of truth
        self.assertEqual(req.extra["sampling_params"]["max_new_tokens"], 7)
        self.assertEqual(req.extra["stop"], ["done"])
        self.assertTrue(req.extra["stream"])

        ar_req = req.to_stage_reqs("auto_regressive")[0]
        self.assertEqual(ar_req.rid, "rid-1")
        self.assertEqual(ar_req.input_ids, tokenized.input_ids)
        self.assertIs(ar_req.mm_inputs, req.omni_inputs)
        self.assertTrue(ar_req.stream)
        self.assertEqual(ar_req.sampling_params.max_new_tokens, 7)
        self.assertAlmostEqual(ar_req.sampling_params.temperature, 0.3)

    def test_convert_no_payload_normalization(self):
        # the mimo payload side-channel / req.audio_payload are gone; nothing is normalized
        mm_inputs = {"audio_token_id": self.audio_token_id, "mm_items": []}
        tokenized = TokenizedGenerateOmniReqInput(
            rid="rid-2",
            prompt="p",
            input_ids=[1, 2],
            mm_inputs=mm_inputs,
            sampling_params={"max_new_tokens": 3},
        )
        req = self._scheduler().convert_omni_request(tokenized)
        self.assertNotIn("mimo_v25_audio_payload", req.omni_inputs)
        self.assertEqual(req.extra["sampling_params"]["max_new_tokens"], 3)

    def test_stage0_to_ar_carries_multimodal_embedding(self):
        mm_inputs = {"audio_token_id": self.audio_token_id, "mm_items": []}
        tokenized = TokenizedGenerateOmniReqInput(
            rid="rid-3",
            prompt="p",
            input_ids=[10, self.audio_token_id, 11],
            mm_inputs=mm_inputs,
            sampling_params={"max_new_tokens": 6},
        )
        req = self._scheduler().convert_omni_request(tokenized)
        req.omni_inputs["multimodal_embedding"] = np.ones((4, 6), dtype=np.float32)

        ar_req = req.to_stage_reqs("auto_regressive")[0]
        self.assertIs(ar_req.mm_inputs, req.omni_inputs)
        self.assertIn("multimodal_embedding", ar_req.mm_inputs)
        self.assertEqual(ar_req.sampling_params.max_new_tokens, 6)


if __name__ == "__main__":
    unittest.main()
