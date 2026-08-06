"""Small deterministic generation sanity check for GLM-5.2 serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

PROMPTS = (
    "Complete this sentence in one short sentence: The purpose of a compiler is",
    "请用一句简短的中文说明什么是缓存。",
    "A farmer has 12 apples and gives away 5. How many apples remain? Answer briefly.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results = []
    for prompt_id, prompt in enumerate(PROMPTS):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "rid": f"basic-eval-{prompt_id}",
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": args.max_new_tokens,
                },
            },
            timeout=(30, None),
        )
        response.raise_for_status()
        payload = response.json()
        generated = payload.get("text")
        meta = payload.get("meta_info") or {}
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError(f"empty generation for prompt {prompt_id}: {payload}")
        if "\ufffd" in generated:
            raise RuntimeError(f"invalid replacement character for prompt {prompt_id}")
        if int(meta.get("completion_tokens", 0)) < 1:
            raise RuntimeError(f"no completion tokens for prompt {prompt_id}: {payload}")
        results.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generation": generated,
                "completion_tokens": int(meta["completion_tokens"]),
                "finish_reason": meta.get("finish_reason"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
