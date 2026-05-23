"""End-to-end tests for Qwen3-VL Dense on the standard LLM path.

Two test cases:
- `test_logit_alignment_vs_hf`: load HF transformers Qwen3-VL on CPU, run forward
  on a synthetic 4-circle image, compare first-token top-K logprobs against the
  sgl-jax server's `top_logprobs`. Top-1 token id must match; top-K id set must
  overlap by >= K-1; top-1 logprob must be within 0.5 nats.
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
import math
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


TOP_K = 5
LOGPROB_ABS_TOL = 0.5  # nats; allows for bf16 noise + JAX vs PyTorch numerical drift


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
        hf_logprobs = torch.log_softmax(last_logits, dim=-1)
        hf_top_vals, hf_top_ids = torch.topk(hf_logprobs, TOP_K)
        hf_top: list[tuple[int, float, str]] = [
            (int(tid), float(lp), proc.tokenizer.decode([int(tid)]))
            for tid, lp in zip(hf_top_ids.tolist(), hf_top_vals.tolist())
        ]

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
            logprobs=True,
            top_logprobs=TOP_K,
        )
        choice = body["choices"][0]
        lp_block = choice.get("logprobs")
        self.assertIsNotNone(lp_block, f"server returned no logprobs: {body}")
        content = lp_block.get("content") or []
        self.assertTrue(content, f"server returned empty content logprobs: {lp_block}")
        top_block = content[0].get("top_logprobs") or []
        self.assertTrue(top_block, f"server returned empty top_logprobs: {content[0]}")

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        sgl_top: list[tuple[int, float, str]] = []
        for entry in top_block:
            decoded = entry.get("token", "")
            logprob = float(entry["logprob"])
            tid = entry.get("id")
            if tid is None:
                ids = tok.encode(decoded, add_special_tokens=False)
                tid = ids[0] if ids else -1
            sgl_top.append((int(tid), logprob, decoded))

        # ---- diagnostics ----
        def _fmt(label: str, rows: list[tuple[int, float, str]]) -> str:
            lines = [f"  {label} top-{TOP_K}:"]
            for i, (tid, lp, dec) in enumerate(rows):
                lines.append(
                    f"    {i + 1}. id={tid:7d}  logprob={lp:+.4f}  p={math.exp(lp):.4f}  text={dec!r}"
                )
            return "\n".join(lines)

        diag = "\n" + _fmt("HF", hf_top) + "\n" + _fmt("sgl-jax", sgl_top)

        self.assertEqual(
            hf_top[0][0],
            sgl_top[0][0],
            f"top-1 token id mismatch\nHF={hf_top[0][2]!r}  sgl={sgl_top[0][2]!r}{diag}",
        )

        overlap = len({r[0] for r in hf_top} & {r[0] for r in sgl_top})
        self.assertGreaterEqual(
            overlap,
            TOP_K - 1,
            f"top-{TOP_K} id overlap = {overlap}, need >= {TOP_K - 1}{diag}",
        )

        lp_diff = abs(hf_top[0][1] - sgl_top[0][1])
        self.assertLessEqual(
            lp_diff,
            LOGPROB_ABS_TOL,
            f"top-1 logprob diff = {lp_diff:.4f} > tolerance {LOGPROB_ABS_TOL}{diag}",
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
