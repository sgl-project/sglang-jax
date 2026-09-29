"""CPU contract tests for embedded multimodal generation."""

import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.entrypoints.engine import Engine


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("media_field", ["image_data", "video_data"])
@pytest.mark.parametrize("batched", [False, True])
def test_engine_preserves_media(asynchronous, stream, media_field, batched):
    # Multiple images/frames per prompt must remain grouped through normalization.
    media = [["first-a", "first-b"], ["second"]]
    prompts = ["first prompt", "second prompt"]
    expected = [{"text": "first answer"}, {"text": "second answer"}]
    if not batched:
        media, prompts, expected = media[0], prompts[0], expected[0]

    async def generate_request(request, _):
        assert getattr(request, media_field) is media
        request.normalize_batch_and_arguments()
        for i in range(2 if batched else 1):
            item = request[i] if batched else request
            assert item.text == (prompts[i] if batched else prompts)
            assert getattr(item, media_field) == (media[i] if batched else media)
            assert item.return_logprob
        if request.stream:
            for result in expected if batched else [expected]:
                yield result
        else:
            yield expected

    loop = asyncio.new_event_loop()
    engine = SimpleNamespace(
        loop=loop,
        tokenizer_manager=SimpleNamespace(generate_request=generate_request),
    )
    kwargs = dict(
        prompt=prompts,
        sampling_params={"max_new_tokens": 2},
        return_logprob=True,
        stream=stream,
        **{media_field: media},
    )

    async def run_async():
        result = await Engine.async_generate(engine, **kwargs)
        return [item async for item in result] if stream else result

    try:
        if asynchronous:
            result = loop.run_until_complete(run_async())
        else:
            result = Engine.generate(engine, **kwargs)
            if stream:
                result = list(result)
        assert result == ([expected] if stream and not batched else expected)
    finally:
        loop.close()
