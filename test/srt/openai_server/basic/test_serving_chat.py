"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import asyncio
import json
import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.managers.io_struct import GenerateReqInput


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.mm_processor = None

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with (
            patch("sgl_jax.srt.entrypoints.openai.serving_chat.generate_chat_conv") as conv_mock,
            patch.object(self.chat, "_process_messages") as proc_mock,
        ):
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch("sgl_jax.srt.entrypoints.openai.serving_chat.generate_chat_conv") as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = initial_stop_str.copy()  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_multimodal_jinja_prompt_skips_encode_decode_round_trip(self):
        self.template_manager.chat_template_name = None
        self.tm.mm_processor = Mock()
        self.tm.mm_processor.apply_chat_template.return_value = "<image>Test prompt"
        self.tm.tokenizer.encode.reset_mock()
        self.tm.tokenizer.decode.reset_mock()

        result = self.chat._apply_jinja_template(self.basic_req, tools=None, is_multimodal=True)

        self.assertEqual(result.prompt, "<image>Test prompt")
        self.assertEqual(result.prompt_ids, [])
        self.tm.tokenizer.encode.assert_not_called()
        self.tm.tokenizer.decode.assert_not_called()

    def test_multimodal_jinja_assistant_prefix_preserves_token_round_trip(self):
        self.template_manager.chat_template_name = None
        self.tm.mm_processor = Mock()
        self.tm.mm_processor.apply_chat_template.return_value = "<image>Test prompt"
        self.tm.tokenizer.encode.reset_mock()
        self.tm.tokenizer.encode.side_effect = ([1, 2], [1, 3])
        self.tm.tokenizer.decode.reset_mock()
        self.tm.tokenizer.decode.return_value = "<image>Test prompt partial"
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Hi?"},
                {"role": "assistant", "content": "partial"},
            ],
            continue_final_message=True,
        )

        result = self.chat._apply_jinja_template(request, tools=None, is_multimodal=True)

        self.assertEqual(result.prompt, "<image>Test prompt partial")
        self.assertEqual(result.prompt_ids, [1, 2, 3])
        self.assertEqual(
            self.tm.tokenizer.encode.call_args_list,
            [unittest.mock.call("<image>Test prompt"), unittest.mock.call("partial")],
        )
        self.tm.tokenizer.decode.assert_called_once_with([1, 2, 3])

    # ------------- sampling-params -------------
    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            stop=["</s>"],
        )
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1], None, None, [], ["</s>"], None),
        ):
            params = self.chat._build_sampling_params(req, ["</s>"], None)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["max_new_tokens"], 150)
            self.assertEqual(params["min_new_tokens"], 5)
            self.assertEqual(params["stop"], ["</s>"])

    async def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            finish_reason_type="stop",
            index=0,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")
        self.assertIn('"arguments":', result, "Should contain arguments field")
        self.assertIn(', "unit": "celsius"}', result, "Should contain remaining arguments")
        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    async def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            finish_reason_type="stop",
            index=0,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    async def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            finish_reason_type="stop",
            index=0,
        )

        # Should return None since there's no parser data
        self.assertIsNone(result, "Should return None when parser has no tool call data")


_KIMI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
_KIMI_CALL_HEAD = (
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    "<|tool_call_argument_begin|>"
)
_KIMI_CALL_TAIL = "<|tool_call_end|><|tool_calls_section_end|>"
_KIMI_SECTION = _KIMI_CALL_HEAD + '{"city": "Paris"}' + _KIMI_CALL_TAIL


def _kimi_ret(text: str, finish_type: str | None = "stop") -> dict:
    return {
        "text": text,
        "meta_info": {
            "id": "chatcmpl-kimi",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cached_tokens": 0,
            "finish_reason": {"type": finish_type, "matched": None} if finish_type else None,
        },
        "index": 0,
    }


class ServingChatKimiK2TestCase(unittest.TestCase):
    """`--reasoning-parser kimi_k2 --tool-call-parser kimi_k2` serving behavior."""

    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat = OpenAIServingChat(self.tm, _MockTemplateManager())

    def _request(self, **kwargs) -> ChatCompletionRequest:
        kwargs.setdefault("messages", [{"role": "user", "content": "Weather in Paris?"}])
        return ChatCompletionRequest(model="x", tools=_KIMI_TOOLS, **kwargs)

    def _stream(self, request: ChatCompletionRequest, deltas: list[str]):
        """Run _generate_chat_stream over cumulative outputs; reassemble the client view."""

        async def _generate():
            text = ""
            for i, delta in enumerate(deltas):
                text += delta
                yield _kimi_ret(text, "stop" if i == len(deltas) - 1 else None)

        async def _collect():
            self.tm.generate_request = Mock(return_value=_generate())
            return [c async for c in self.chat._generate_chat_stream(Mock(), request, Mock())]

        reasoning, content, calls = "", "", {}
        for line in asyncio.run(_collect()):
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line[len("data: ") :])["choices"]:
                delta = choice["delta"]
                reasoning += delta.get("reasoning_content") or ""
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    call = calls.setdefault(
                        tc["index"], {"id": None, "name": None, "arguments": ""}
                    )
                    call["id"] = call["id"] or tc["id"]
                    call["name"] = call["name"] or tc["function"]["name"]
                    call["arguments"] += tc["function"]["arguments"] or ""
        return reasoning, content, [calls[i] for i in sorted(calls)]

    # ------------- reasoning gate / chat_template_kwargs -------------
    def test_reasoning_gate_reads_thinking_not_enable_thinking(self):
        gate = self.chat._get_reasoning_from_request
        self.assertTrue(gate(self._request()))
        self.assertFalse(gate(self._request(chat_template_kwargs={"thinking": False})))
        self.assertTrue(gate(self._request(chat_template_kwargs={"enable_thinking": False})))

    def test_forced_tool_choice_defaults_thinking_off(self):
        result = MessageProcessingResult("p", [1], None, None, None, [], [])
        with patch.object(self.chat, "_apply_conversation_template", return_value=result):
            required = self._request(tool_choice="required")
            self.chat._process_messages(required, is_multimodal=False)
            self.assertEqual(required.chat_template_kwargs, {"thinking": False})

            explicit = self._request(
                tool_choice="required", chat_template_kwargs={"thinking": True}
            )
            self.chat._process_messages(explicit, is_multimodal=False)
            self.assertEqual(explicit.chat_template_kwargs, {"thinking": True})

            auto = self._request(tool_choice="auto")
            self.chat._process_messages(auto, is_multimodal=False)
            self.assertIsNone(auto.chat_template_kwargs)

            self.tm.server_args.reasoning_parser = "qwen3"
            other = self._request(tool_choice="required")
            self.chat._process_messages(other, is_multimodal=False)
            self.assertIsNone(other.chat_template_kwargs)

    # ------------- tool-call IDs -------------
    def test_history_tool_calls_cnt(self):
        call = {"id": "functions.get_weather:0", "type": "function"}
        call["function"] = {"name": "get_weather", "arguments": "{}"}
        request = self._request(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [call, call]},
                {"role": "tool", "content": "sunny", "tool_call_id": "functions.get_weather:0"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "user", "content": "again"},
            ]
        )
        self.assertEqual(self.chat._get_history_tool_calls_cnt(request), 3)

    def test_tool_call_ids_continue_history_counter(self):
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_weather:7<|tool_call_argument_begin|>{"city": "A"}'
            '<|tool_call_end|><|tool_call_begin|>get_weather:8<|tool_call_argument_begin|>{"city": "B"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        tools = self._request().tools
        calls, _, finish = self.chat._process_tool_calls(
            text, tools, "kimi_k2", {"type": "stop"}, history_tool_calls_cnt=2
        )
        self.assertEqual(
            [c.id for c in calls], ["functions.get_weather:2", "functions.get_weather:3"]
        )
        self.assertEqual(finish["type"], "tool_calls")

        self.tm.server_args.tool_call_parser = "qwen25"
        calls, _, _ = self.chat._process_tool_calls(
            '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
            tools,
            "qwen25",
            {"type": "stop"},
        )
        self.assertTrue(calls[0].id.startswith("call_"))

    # ------------- non-streaming end to end -------------
    def test_non_streaming_think_end_after_tool_section(self):
        request = self._request()
        response = self.chat._build_chat_response(
            request, [_kimi_ret("plan" + _KIMI_SECTION + "</think>")], 0
        )
        choice = response.choices[0]
        self.assertEqual(choice.message.reasoning_content, "plan")
        self.assertIsNone(choice.message.content)
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].id, "functions.get_weather:0")
        self.assertEqual(
            json.loads(choice.message.tool_calls[0].function.arguments), {"city": "Paris"}
        )

    def test_non_streaming_forced_json_fallback(self):
        # _process_messages already set thinking=False for tool_choice="required".
        request = self._request(tool_choice="required", chat_template_kwargs={"thinking": False})
        response = self.chat._build_chat_response(
            request, [_kimi_ret('[{"name": "get_weather", "parameters": {"city": "Paris"}}]')], 0
        )
        choice = response.choices[0]
        self.assertIsNone(choice.message.reasoning_content)
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].id, "functions.get_weather:0")

    # ------------- streaming end to end -------------
    def test_streaming_think_end_after_tool_section_per_token(self):
        deltas = ["pl", "an", "<|tool_calls_section_begin|>", "<|tool_call_begin|>"]
        deltas += ["functions.get_weather:0", "<|tool_call_argument_begin|>", '{"city": ']
        deltas += ['"Paris"}', "<|tool_call_end|>", "<|tool_calls_section_end|>", "</think>"]
        reasoning, content, calls = self._stream(self._request(stream=True), deltas)
        self.assertEqual((reasoning, content), ("plan", ""))
        self.assertEqual(
            calls,
            [
                {
                    "id": "functions.get_weather:0",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                }
            ],
        )

    def test_streaming_argument_tail_in_final_chunk_is_not_corrupted(self):
        deltas = ["plan", _KIMI_CALL_HEAD + '{"city": ', '"Paris"}' + _KIMI_CALL_TAIL]
        _, _, calls = self._stream(self._request(stream=True), deltas)
        self.assertEqual(json.loads(calls[0]["arguments"]), {"city": "Paris"})

    def test_streaming_whole_call_in_final_chunk_is_not_corrupted(self):
        _, _, calls = self._stream(self._request(stream=True), ["plan", _KIMI_SECTION])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"city": "Paris"})

    def test_streaming_thinking_off_and_history_offset(self):
        call = {"id": "functions.get_weather:0", "type": "function"}
        call["function"] = {"name": "get_weather", "arguments": '{"city": "Oslo"}'}
        request = self._request(
            stream=True,
            chat_template_kwargs={"thinking": False},
            messages=[
                {"role": "user", "content": "Oslo?"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "content": "rain", "tool_call_id": "functions.get_weather:0"},
                {"role": "user", "content": "Paris?"},
            ],
        )
        reasoning, content, calls = self._stream(request, ["Sure.", _KIMI_SECTION, ""])
        self.assertEqual((reasoning, content), ("", "Sure."))
        self.assertEqual(calls[0]["id"], "functions.get_weather:1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
