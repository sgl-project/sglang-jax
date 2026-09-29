---
title: "MiMo-V2.5"
---

# MiMo-V2.5 on SGL-JAX

> **Starter recipe** — image, audio, and video accuracy evaluated on a single TPU v7x-8 host. Serving throughput has not been benchmarked.

## 1. Model Introduction

`XiaomiMiMo/MiMo-V2.5` supports image, audio, and video understanding with text responses, including thinking and NEXTN/MTP decoding. For the separate Pro checkpoint, see [MiMo-V2.5-Pro](./MiMo-V2.5-Pro.md).

## 2. Deployment

Hardware: one TPU v7x-8 host (4 chips, 8 JAX devices), TP=8, DP=2, EP=8. Other topologies have not been evaluated for this recipe.

Install per the [installation guide](../../get_started/install.md), including multimodal dependencies:

```bash
pip install -e "./python[tpu,multimodal]"
```

Evaluation runtime: JAX/JAXLIB 0.11.1 and libtpu 0.0.46.1. The final audio fixes are included in commit `4695b3986`; image/video runs preceded those audio-only fixes. The launch configuration below is based on these runs. See the [launch flags reference](../../base/launch-flags-reference.md) for flag definitions.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
python -m sgl_jax.launch_server \
  --model-path XiaomiMiMo/MiMo-V2.5 \
  --trust-remote-code --device tpu \
  --tp-size 8 --dp-size 2 --ep-size 8 \
  --enable-sequence-parallel --moe-backend fused_v2 \
  --page-size 256 --context-length 262144 --max-seq-len 262144 \
  --chunked-prefill-size 4096 --max-prefill-tokens 16384 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.3 \
  --disable-radix-cache --skip-server-warmup \
  --max-running-requests 16 --dp-schedule-policy round_robin \
  --vision-encoder-parallel dp --mm-processor-worker-num 1 \
  --reasoning-parser mimo \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --speculative-accept-threshold-single 1.0 \
  --speculative-accept-threshold-acc 1.0 \
  --host 0.0.0.0 --port 30000
```

## 3. Invocation

Send each media file as a base64 data URL. This example makes three separate requests; replace the filenames with your own files. Audio uses the server's `audio_url` content type.

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

for kind, filename, mime, prompt in [
    ("image", "image.jpg", "image/jpeg", "Describe this image."),
    ("audio", "audio.wav", "audio/wav", "Transcribe this audio."),
    ("video", "video.mp4", "video/mp4", "Describe what happens in this video."),
]:
    encoded = base64.b64encode(Path(filename).read_bytes()).decode("ascii")
    media_type = f"{kind}_url"
    response = client.chat.completions.create(
        model="XiaomiMiMo/MiMo-V2.5",
        messages=[{
            "role": "user",
            "content": [
                {"type": media_type, media_type: {"url": f"data:{mime};base64,{encoded}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=1.0,
        top_p=0.95,
        max_tokens=32768,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    print(kind, response.choices[0].message.content)
```
