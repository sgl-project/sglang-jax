"""End-to-end tests for Qwen3-VL Dense on the standard LLM path.

Two test cases:
- `test_logit_alignment_vs_hf`: load HF transformers Qwen3-VL on CPU, run forward
  on a synthetic 4-circle image, compare the argmax (top-1) next-token id
  against the sgl-jax server's greedy first-token. The richer top-K logprobs
  comparison is blocked on a pre-existing shard_map issue in the logprobs path
  (`logits_processor._select_logits` expects sample_indices to be P("data")
  but `device_array(...)` returns it replicated).
- `test_multi_image_prefill`: send a request with two distinct images and assert
  the response cites both. Smoke-test for the multi-image splice path.

Both tests share one server boot (sgl-jax with TP=4 + page_size=16 + radix cache
off, as required by the pilot).

Resolves model path via `SGLANG_JAX_MODEL_CACHE` env (e.g. `/models`), falling
back to `Qwen/Qwen3-VL-8B-Instruct` HF id (downloads on miss).
"""

from __future__ import annotations

import base64
import io
import json
import os
import unittest
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_VL_8B_INSTRUCT,
    CustomTestCase,
    popen_launch_server,
)


def _resolve_model_path() -> str:
    """Try plain `<cache>/Qwen3-VL-8B-Instruct` (no org prefix) before falling
    back to whatever `_local_or_hf` resolved."""
    cache = os.environ.get("SGLANG_JAX_MODEL_CACHE")
    if cache:
        candidate = Path(cache) / "Qwen3-VL-8B-Instruct"
        if (candidate / "config.json").is_file():
            return str(candidate)
    return QWEN3_VL_8B_INSTRUCT


def _make_4circle_2x2() -> Image.Image:
    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    for cx, cy in [(112, 112), (336, 112), (112, 336), (336, 336)]:
        d.ellipse((cx - 60, cy - 60, cx + 60, cy + 60), fill="red", outline="black", width=3)
    return img


def _make_solid(color: str, side: int = 448) -> Image.Image:
    return Image.new("RGB", (side, side), color)


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _post_chat(base_url: str, content: list[dict], max_tokens: int = 16, **extra) -> dict:
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **extra,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


class TestQwen3VLE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _resolve_model_path()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            # --disable-precompile produces cache misses on first inference;
            # don't fail the run on that.
            check_cache_miss=False,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--context-length",
                "4096",
                "--page-size",
                "16",
                "--disable-precompile",
                # radix cache keys on raw input_ids which would incorrectly share KV
                # across different images sharing <|image_pad|> token ids
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # #2 - logit alignment vs HF transformers
    # ------------------------------------------------------------------
    def test_logit_alignment_vs_hf(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as e:
            self.skipTest(f"transformers/torch not available for HF reference: {e}")

        img = _make_4circle_2x2()
        prompt = "How many red circles? Just a number."

        # ---- HF reference ----
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            self.model, trust_remote_code=True, dtype=torch.bfloat16
        ).cpu()
        hf_model.eval()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
        with torch.no_grad():
            out = hf_model(**inputs)
        last_logits = out.logits[0, -1, :].float()
        hf_top1_id = int(torch.argmax(last_logits).item())
        hf_top1_text = proc.tokenizer.decode([hf_top1_id])

        # ---- sgl-jax request ----
        body = _post_chat(
            self.base_url,
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_img_to_b64(img)}"},
                },
                {"type": "text", "text": prompt},
            ],
            max_tokens=1,
        )
        choice = body["choices"][0]
        sgl_text = choice["message"]["content"]
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        sgl_ids = tok.encode(sgl_text, add_special_tokens=False)
        sgl_top1_id = sgl_ids[0] if sgl_ids else -1

        diag = (
            f"\n  HF      top-1: id={hf_top1_id:7d}  text={hf_top1_text!r}"
            f"\n  sgl-jax top-1: id={sgl_top1_id:7d}  text={sgl_text!r}"
        )
        self.assertEqual(
            hf_top1_id,
            sgl_top1_id,
            f"top-1 token id mismatch{diag}",
        )

    # ------------------------------------------------------------------
    # #1 - multi-image E2E
    # ------------------------------------------------------------------
    def test_multi_image_prefill(self) -> None:
        red = _make_solid("red")
        blue = _make_solid("blue")
        body = _post_chat(
            self.base_url,
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_img_to_b64(red)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_img_to_b64(blue)}"},
                },
                {
                    "type": "text",
                    "text": (
                        "There are two images. What color is the first? What color is the second? "
                        "Answer in the form 'First: X. Second: Y.'"
                    ),
                },
            ],
            max_tokens=32,
        )
        answer = body["choices"][0]["message"]["content"].lower()
        self.assertIn("red", answer, f"first image color missing from response: {answer!r}")
        self.assertIn("blue", answer, f"second image color missing from response: {answer!r}")


if __name__ == "__main__":
    unittest.main()
