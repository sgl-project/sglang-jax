"""Small shared helpers for deterministic VLM serving tests."""

import base64
import io

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def image_content(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def complete(client, messages, **generation):
    response = client.chat.completions.create(model=MODEL_ID, messages=messages, **generation)
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if choice.finish_reason != "stop" or not content:
        raise ValueError(
            f"Incomplete VLM response: {choice.finish_reason}; "
            f"content={(choice.message.content or '')[:200]!r}"
        )
    return content
