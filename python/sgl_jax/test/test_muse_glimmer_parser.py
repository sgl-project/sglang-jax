import asyncio
import json
from types import SimpleNamespace

from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    Tool,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    normalize_message_reasoning,
)
from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser


def _weather_tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )


def test_required_tool_choice_uses_native_muse_format():
    parser = FunctionCallParser([_weather_tool()], "muse_glimmer")

    assert parser.get_structure_constraint("required") is None


def test_request_reasoning_alias_is_preserved_for_chat_templates():
    request = ChatCompletionRequest(
        model="muse_glimmer_30b",
        messages=[
            {
                "role": "assistant",
                "content": "final answer",
                "reasoning": "prior reasoning",
            }
        ],
    )

    message = normalize_message_reasoning(request.messages[0].model_dump())

    assert message["reasoning"] == "prior reasoning"
    assert message["reasoning_content"] == "prior reasoning"


def test_streaming_tool_call_finishes_without_detector_state():
    parser = FunctionCallParser([_weather_tool()], "muse_glimmer")
    service = object.__new__(OpenAIServingChat)
    request = SimpleNamespace(model="muse_glimmer_30b", stream_options=None)
    content = {"meta_info": {"id": "request-id"}}
    delta = (
        '<atem:function_calls><atem:invoke name="get_weather">'
        '<atem:parameter name="city">"Paris"</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )

    async def collect_chunks() -> list[str]:
        return [
            chunk
            async for chunk in service._process_tool_call_stream(
                0,
                delta,
                {0: parser},
                content,
                request,
                "stop",
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert len(chunks) == 1
    payload = json.loads(chunks[0].removeprefix("data: "))
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["delta"]["tool_calls"][0]
    assert tool_call["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
