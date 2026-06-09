"""Tests for OpenAI serving layer PD field passthrough."""

from unittest.mock import MagicMock, patch

from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


class TestProtocolFields:
    def test_chat_completion_request_has_bootstrap_fields(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=12345,
            disagg_transfer_id="tid-001",
        )
        assert req.bootstrap_host == "10.0.0.1"
        assert req.bootstrap_port == 8998
        assert req.bootstrap_room == 12345
        assert req.disagg_transfer_id == "tid-001"

    def test_completion_request_has_bootstrap_fields(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            bootstrap_host="10.0.0.2",
            bootstrap_port=9998,
            bootstrap_room=67890,
            disagg_transfer_id="tid-002",
        )
        assert req.bootstrap_host == "10.0.0.2"
        assert req.bootstrap_port == 9998
        assert req.bootstrap_room == 67890
        assert req.disagg_transfer_id == "tid-002"

    def test_bootstrap_fields_default_none(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.bootstrap_host is None
        assert req.bootstrap_port is None
        assert req.bootstrap_room is None
        assert req.disagg_transfer_id is None


class TestServingChatPassthrough:
    def test_chat_passes_bootstrap_fields(self):
        from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        mock_tokenizer_manager = MagicMock()
        mock_tokenizer_manager.model_config = None
        mock_tokenizer_manager.server_args = MagicMock()
        mock_tokenizer_manager.server_args.multimodal = False

        mock_template_manager = MagicMock()
        serving = OpenAIServingChat(mock_tokenizer_manager, mock_template_manager)

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=42,
        )

        with patch.object(serving, "_process_messages") as mock_process:
            mock_process.return_value = MagicMock(
                prompt="hello",
                prompt_ids="hello",
                image_data=None,
                video_data=None,
                audio_data=None,
                stop=[],
                tool_call_constraint=None,
            )
            with patch.object(serving, "_build_sampling_params", return_value={}):
                adapted, _ = serving._convert_to_internal_request(request)

        assert adapted.bootstrap_host == "10.0.0.1"
        assert adapted.bootstrap_port == 8998
        assert adapted.bootstrap_room == 42


class TestServingCompletionsPassthrough:
    def test_completions_passes_bootstrap_fields(self):
        from sgl_jax.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )

        mock_tokenizer_manager = MagicMock()
        mock_template_manager = MagicMock()
        mock_template_manager.completion_template_name = None
        serving = OpenAIServingCompletion(mock_tokenizer_manager, mock_template_manager)

        request = CompletionRequest(
            model="test",
            prompt="hello",
            bootstrap_host="10.0.0.2",
            bootstrap_port=9998,
            bootstrap_room=99,
        )

        adapted, _ = serving._convert_to_internal_request(request)

        assert adapted.bootstrap_host == "10.0.0.2"
        assert adapted.bootstrap_port == 9998
        assert adapted.bootstrap_room == 99
