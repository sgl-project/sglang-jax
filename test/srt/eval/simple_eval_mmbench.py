"""MMBench V1.1: pinned questions, Qwen3-VL prompts and CircularEval."""

# Image reference resolution / CircularEval adapted from VLMEvalKit's
# vlmeval/dataset/image_base.py and dataset/utils/multiple_choice.py at REVISION.
# Copyright 2023 VLMEvalKit Authors. SPDX-License-Identifier: Apache-2.0.
# Changes: fixed single-image dataset, in-memory scoring, no judge/result cache.
# See repository LICENSE.

import base64
import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from eval.mmbench_utils import can_infer, choices_for, prompt_for
from PIL import Image
from vlm_utils import MODEL_REVISION, complete, image_content

DATASET = "MMBench_DEV_EN_V11"
DATASET_SHA256 = "27a54ae2e8f54502361cc5a41ed5fc35f4253af74734b5593d7b1de545eb5656"
REVISION = "d21c5e969983dea6741a98eb2f7ec566ab82ee91"


def load_data():
    # Preserve VLMEvalKit's existing cache environment variable.
    root = os.environ.get("LMUData", Path.home() / "LMUData")  # noqa: SIM112
    path = Path(root) / f"{DATASET}.tsv"
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://opencompass.openxlab.space/utils/benchmarks/MMBench/{DATASET}.tsv"
        try:
            response = requests.get(url, timeout=120)
        except requests.exceptions.SSLError:
            # The public dataset host has an expired certificate. Authenticate the
            # bytes with the pinned SHA256 below; never disable TLS globally.
            response = requests.get(url, timeout=120, verify=False)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != DATASET_SHA256:
            raise ValueError("MMBench V1.1 checksum mismatch")
        path.write_bytes(response.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("MMBench V1.1 checksum mismatch")
    data = pd.read_csv(path, sep="\t")
    # The pinned TSV stores a base64 image or the index of another image row.
    images = dict(zip(data["index"].astype(str), data["image"].astype(str)))
    data["image"] = [
        images[value] if len(value) <= 64 else value for value in data["image"].astype(str)
    ]
    return data


def select_indices(data, limit):
    originals = data.loc[data["index"].between(0, 999_999), "index"].tolist()
    if limit is None:
        ids = originals
    elif 0 < limit <= len(originals):
        ids = random.Random(0).sample(originals, limit)
    else:
        raise ValueError("MMBench limit must be between 1 and the number of original questions")
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(originals):
        raise ValueError("Missing or duplicate MMBench question IDs")
    selected = data[data["index"].mod(1_000_000).isin(ids)].copy()
    # Preserve every row from the checksum-pinned source for these originals.
    # Some four-option questions have only three rotations ("all of the above"
    # stays fixed), so option count must not determine group completeness.
    if not selected["index"].is_unique:
        raise ValueError("Duplicate MMBench permutation IDs")
    return selected


def messages_for(row):
    with Image.open(io.BytesIO(base64.b64decode(row["image"]))) as image:
        content = [image_content(image), {"type": "text", "text": prompt_for(row)}]
    return [{"role": "user", "content": content}]


def run_mmbench(args):
    from openai import OpenAI

    started = time.monotonic()
    data = select_indices(load_data(), args.num_examples)
    generation = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    with tempfile.TemporaryDirectory(prefix="mmbench-v11-") as temporary:
        output = (
            Path(os.environ.get("RESULTS_DIR", temporary))
            / f"mmbench-v11-{args.num_examples or 'full'}"
        )
        output.mkdir(parents=True, exist_ok=True)
        prepared = time.monotonic()
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        with OpenAI(
            base_url=base_url + "/v1", api_key="EMPTY", timeout=120, max_retries=0
        ) as client:

            def evaluate(row):
                try:
                    answer = complete(client, messages_for(row), **generation)
                except Exception as exc:
                    raise RuntimeError(f"MMBench {row['index']}: {exc}") from exc
                return {
                    "id": int(row["index"]),
                    "answer": row["answer"],
                    "response": answer,
                }

            samples = []
            with (
                ThreadPoolExecutor(max_workers=args.num_threads) as pool,
                (output / "responses.jsonl").open("w") as stream,
            ):
                try:
                    for sample in pool.map(evaluate, (row for _, row in data.iterrows())):
                        samples.append(sample)
                        stream.write(json.dumps(sample) + "\n")
                        stream.flush()
                except Exception:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        inferred = time.monotonic()
        if len(samples) != len(data):
            raise ValueError("Missing MMBench responses")
        predictions = data.drop(columns="image", errors="ignore").copy()
        predictions["prediction"] = [sample["response"] for sample in samples]
        predictions["parsed_answer"] = predictions.apply(
            lambda row: can_infer(row["prediction"], choices_for(row)) or "Z", axis=1
        )
        predictions["correct"] = predictions["parsed_answer"] == predictions["answer"]
        # All source permutations must be correct; unmatched answers fail the group.
        questions = predictions.groupby(predictions["index"].mod(1_000_000))["correct"].all()
        score = questions.mean()
        if score is None or not math.isfinite(score):
            raise ValueError("MMBench produced no finite score")
        predictions.to_csv(output / "predictions.tsv", sep="\t", index=False)
        questions.to_csv(output / "questions.csv")
        ids = [str(sample["id"]) for sample in samples]
        return {
            "score": float(score),
            "num_examples": int(data["index"].mod(1_000_000).nunique()),
            "num_requests": len(samples),
            "dataset_id": DATASET,
            "dataset_sha256": DATASET_SHA256,
            "vlmevalkit_revision": REVISION,
            "model_revision": MODEL_REVISION,
            "split": "dev",
            "evaluation_protocol": "circulareval-exact-matching",
            "prompt_version": "qwen3-vl-vlmevalkit",
            "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "timing_seconds": {
                "prepare": prepared - started,
                "inference": inferred - prepared,
                "grading": time.monotonic() - inferred,
            },
            "samples": samples,
        }
