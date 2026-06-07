from __future__ import annotations

import dataclasses
import sys
import types
import unittest
from types import SimpleNamespace


def _install_import_stubs():
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.Request = type("Request", (), {})
    fastapi_responses_stub = types.ModuleType("fastapi.responses")
    fastapi_responses_stub.ORJSONResponse = type("ORJSONResponse", (), {})
    fastapi_responses_stub.StreamingResponse = type("StreamingResponse", (), {})
    sys.modules.setdefault("fastapi", fastapi_stub)
    sys.modules.setdefault("fastapi.responses", fastapi_responses_stub)

    protocol_stub = types.ModuleType("sgl_jax.srt.entrypoints.openai.protocol")

    @dataclasses.dataclass
    class _MessageProcessingResult:
        prompt: str
        prompt_ids: str | list[int]
        image_data: list | None
        audio_data: list | None
        video_data: list | None
        modalities: list[str]
        stop: list[str] | None
        tool_call_constraint: object | None = None

    protocol_stub.MessageProcessingResult = _MessageProcessingResult
    for name in (
        "ChatCompletionRequest",
        "ChatCompletionResponse",
        "ChatCompletionResponseChoice",
        "ChatCompletionResponseStreamChoice",
        "ChatCompletionStreamResponse",
        "ChatCompletionTokenLogprob",
        "ChatMessage",
        "ChoiceLogprobs",
        "DeltaMessage",
        "ErrorResponse",
        "FunctionResponse",
        "LogProbs",
        "ToolCall",
        "ToolChoice",
        "TopLogprob",
    ):
        setattr(protocol_stub, name, type(name, (), {}))
    sys.modules.setdefault("sgl_jax.srt.entrypoints.openai.protocol", protocol_stub)

    serving_base_stub = types.ModuleType("sgl_jax.srt.entrypoints.openai.serving_base")
    serving_base_stub.OpenAIServingBase = type(
        "OpenAIServingBase",
        (),
        {
            "__init__": lambda self, tokenizer_manager: setattr(
                self, "tokenizer_manager", tokenizer_manager
            )
        },
    )
    sys.modules.setdefault("sgl_jax.srt.entrypoints.openai.serving_base", serving_base_stub)

    usage_stub = types.ModuleType("sgl_jax.srt.entrypoints.openai.usage_processor")
    usage_stub.UsageProcessor = type("UsageProcessor", (), {})
    sys.modules.setdefault("sgl_jax.srt.entrypoints.openai.usage_processor", usage_stub)

    utils_stub = types.ModuleType("sgl_jax.srt.entrypoints.openai.utils")
    utils_stub.process_hidden_states_from_ret = lambda *args, **kwargs: None
    utils_stub.to_openai_style_logprobs = lambda *args, **kwargs: None
    sys.modules.setdefault("sgl_jax.srt.entrypoints.openai.utils", utils_stub)

    function_parser_stub = types.ModuleType("sgl_jax.srt.function_call.function_call_parser")
    function_parser_stub.FunctionCallParser = type("FunctionCallParser", (), {})
    sys.modules.setdefault("sgl_jax.srt.function_call.function_call_parser", function_parser_stub)

    managers_io_stub = types.ModuleType("sgl_jax.srt.managers.io_struct")

    @dataclasses.dataclass
    class _ImageData:
        url: str
        detail: str = "auto"

    @dataclasses.dataclass
    class _GenerateReqInput:
        text: str | None = None
        input_ids: list[int] | None = None
        image_data: list | None = None
        video_data: list | None = None
        audio_data: list | None = None
        sampling_params: dict | None = None
        return_logprob: bool = False
        logprob_start_len: int = -1
        top_logprobs_num: int = 0
        stream: bool = False
        extra_key: str | None = None
        rid: str | None = None

    managers_io_stub.ImageData = _ImageData
    managers_io_stub.GenerateReqInput = _GenerateReqInput
    sys.modules.setdefault("sgl_jax.srt.managers.io_struct", managers_io_stub)

    template_manager_stub = types.ModuleType("sgl_jax.srt.managers.template_manager")
    template_manager_stub.TemplateManager = type("TemplateManager", (), {})
    sys.modules.setdefault("sgl_jax.srt.managers.template_manager", template_manager_stub)

    tokenizer_manager_stub = types.ModuleType("sgl_jax.srt.managers.tokenizer_manager")
    tokenizer_manager_stub.TokenizerManager = type("TokenizerManager", (), {})
    sys.modules.setdefault("sgl_jax.srt.managers.tokenizer_manager", tokenizer_manager_stub)

    mm_io_stub = types.ModuleType("sgl_jax.srt.multimodal.manager.io_struct")

    @dataclasses.dataclass
    class _GenerateOmniReqInput:
        prompt: str | None = None
        input_ids: list[int] | None = None
        image_data: list | None = None
        video_data: list | None = None
        audio_data: list | None = None
        sampling_params: dict | None = None
        return_logprob: bool = False
        logprob_start_len: int = -1
        top_logprobs_num: int = 0
        extra_key: str | None = None
        stream: bool = False
        rid: str | None = None
        stop: str | list[str] | None = None

    mm_io_stub.GenerateOmniReqInput = _GenerateOmniReqInput
    sys.modules.setdefault("sgl_jax.srt.multimodal.manager.io_struct", mm_io_stub)

    conversation_stub = types.ModuleType("sgl_jax.srt.conversation")
    conversation_stub.generate_chat_conv = lambda *args, **kwargs: None
    sys.modules.setdefault("sgl_jax.srt.conversation", conversation_stub)

    reasoning_stub = types.ModuleType("sgl_jax.srt.reasoning_parser")
    reasoning_stub.ReasoningParser = type("ReasoningParser", (), {})
    sys.modules.setdefault("sgl_jax.srt.reasoning_parser", reasoning_stub)

    sgl_utils_stub = types.ModuleType("sgl_jax.utils")
    sgl_utils_stub.convert_json_schema_to_str = lambda value: value
    sys.modules.setdefault("sgl_jax.utils", sgl_utils_stub)


_install_import_stubs()

from sgl_jax.srt.entrypoints.openai.protocol import (  # noqa: E402
    MessageProcessingResult,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sgl_jax.srt.multimodal.manager.io_struct import GenerateOmniReqInput  # noqa: E402


class TestOpenAIAudioToOmniRequest(unittest.TestCase):
    def test_multimodal_chat_conversion_preserves_audio_data(self):
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
            server_args=SimpleNamespace(multimodal=True),
        )
        serving._process_messages = lambda request, is_multimodal: MessageProcessingResult(
            prompt="<audio>",
            prompt_ids=[1, 2, 3],
            image_data=None,
            video_data=None,
            audio_data=["UklGRg=="],
            modalities=["audio"],
            stop=["</s>"],
        )
        serving._build_sampling_params = lambda request, stop, tool_call_constraint: {
            "max_new_tokens": 9,
            "stop": stop,
        }
        request = SimpleNamespace(
            stream=True,
            rid="chat-rid",
            stop=["done"],
            logprobs=False,
            top_logprobs=None,
            extra_key=None,
        )

        internal_request, returned_request = serving._convert_to_internal_request(request)

        self.assertIs(returned_request, request)
        self.assertIsInstance(internal_request, GenerateOmniReqInput)
        self.assertEqual(internal_request.prompt, "<audio>")
        self.assertEqual(internal_request.audio_data, ["UklGRg=="])
        self.assertEqual(internal_request.sampling_params["max_new_tokens"], 9)
        self.assertTrue(internal_request.stream)
        self.assertEqual(internal_request.rid, "chat-rid")
        self.assertEqual(internal_request.stop, ["done"])

    def test_multimodal_chat_rejects_logprobs(self):
        # logprobs/top_logprobs are not plumbed through the omni AR stage; the
        # multimodal path must raise instead of silently dropping them (D5-5).
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True),
            server_args=SimpleNamespace(multimodal=True),
        )
        serving._process_messages = lambda request, is_multimodal: MessageProcessingResult(
            prompt="<audio>",
            prompt_ids=[1, 2, 3],
            image_data=None,
            video_data=None,
            audio_data=["UklGRg=="],
            modalities=["audio"],
            stop=None,
        )
        serving._build_sampling_params = lambda request, stop, tool_call_constraint: {}
        for logprobs, top_logprobs in ((True, None), (False, 5)):
            request = SimpleNamespace(
                stream=False,
                rid="r",
                stop=None,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                extra_key=None,
            )
            with (
                self.subTest(logprobs=logprobs, top_logprobs=top_logprobs),
                self.assertRaises(ValueError),
            ):
                serving._convert_to_internal_request(request)


if __name__ == "__main__":
    unittest.main()
