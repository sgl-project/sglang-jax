---
title: "Basic API Usage"
---

# Basic API Usage

This page is the shared reference for sending requests to a running sglang-jax server. Every recipe's §3.1 Basic Usage links here for the cURL / Python / streaming patterns; the recipe pages then add only the model-specific bits (recommended sampling parameters, parser keys, multi-turn examples).

For installing the server see [`../../get_started/install.md`](../../get_started/install.md). For the full launch-flag reference see [`launch-flags-reference.md`](launch-flags-reference.md).

## 1. Launch a Server

A minimum single-host launch on TPU (substitute the model and topology your recipe specifies):

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-8B \
  --trust-remote-code \
  --tp-size 4 \
  --device tpu \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 --port 30000
```

Once the server is up the OpenAI-compatible endpoints listen on `http://<host>:30000/v1/...`. The raw native endpoint is `http://<host>:30000/generate`.

The server also exposes API documentation at:

- `http://<host>:30000/docs` — Swagger UI
- `http://<host>:30000/redoc` — ReDoc
- `http://<host>:30000/openapi.json` — OpenAPI spec

## 2. Using cURL

Send an OpenAI-compatible chat completion request:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 3. Using Python `requests`

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 64,
        "temperature": 0,
    },
)
print(resp.json()["choices"][0]["message"]["content"])
```

## 4. Using the OpenAI Python Client

Non-streaming:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Streaming — set `stream=True` and iterate `delta.content`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
    temperature=0,
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## 5. Using the Native `/generate` Endpoint

The native endpoint is more flexible than the OpenAI shim — pass raw `text` plus a `sampling_params` dict.

Non-streaming:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    },
)
print(resp.json())
```

Streaming — SSE-style; parse `data:` lines and break on `[DONE]`:

```python
import json
import requests

resp = requests.post(
    "http://127.0.0.1:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        "stream": True,
    },
    stream=True,
)

prev = 0
for line in resp.iter_lines(decode_unicode=False):
    if not line:
        continue
    chunk = line.decode("utf-8")
    if chunk == "data: [DONE]":
        break
    if chunk.startswith("data:"):
        payload = json.loads(chunk[len("data:") :].strip())
        text = payload.get("text", "")
        print(text[prev:], end="", flush=True)
        prev = len(text)
print()
```

## 6. What's Next

For model-specific behavior (recommended sampling parameters, reasoning parser, tool-call parser, multi-host launch flags) see the recipe in [`../autoregressive/`](../autoregressive/). The recipe's §3.1 will name the model path and recommended parameters; everything else on this page applies unchanged.
