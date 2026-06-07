import dataclasses
import sys
import types
import unittest


@dataclasses.dataclass
class _ImageData:
    url: str
    detail: str = "auto"


io_struct_stub = types.ModuleType("sgl_jax.srt.managers.io_struct")
io_struct_stub.ImageData = _ImageData
sys.modules.setdefault("sgl_jax.srt.managers.io_struct", io_struct_stub)

from sgl_jax.srt.jinja_template_utils import (  # noqa: E402
    process_content_for_template_format,
)


class TestOpenAIAudioContentParts(unittest.TestCase):
    def test_chat_request_accepts_input_audio_part(self):
        try:
            from sgl_jax.srt.entrypoints.openai.protocol import ChatCompletionRequest
        except TypeError as exc:
            raise unittest.SkipTest(
                "protocol.py uses Python 3.10 union annotations; local pydantic/Python "
                "cannot evaluate them without eval_type_backport"
            ) from exc

        request = ChatCompletionRequest.model_validate(
            {
                "model": "mimo-v2.5",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "transcribe this"},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "UklGRg==", "format": "wav"},
                            },
                        ],
                    }
                ],
            }
        )

        audio_part = request.messages[0].content[1]
        self.assertEqual(audio_part.type, "input_audio")
        self.assertEqual(audio_part.input_audio.data, "UklGRg==")
        self.assertEqual(audio_part.input_audio.format, "wav")

    def test_template_processing_extracts_input_audio_data(self):
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        processed = process_content_for_template_format(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this audio"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRg==", "format": "wav"},
                    },
                ],
            },
            "openai",
            image_data,
            video_data,
            audio_data,
            modalities,
        )

        self.assertEqual(audio_data, ["UklGRg=="])
        self.assertEqual(
            processed["content"],
            [{"type": "text", "text": "describe this audio"}, {"type": "audio"}],
        )


if __name__ == "__main__":
    unittest.main()
