"""Validate the existing epic/dsv4 static expert-FP8 publication at model load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT = "sglang-jax-deepseek-v4-expert-fp8-per-channel-v1"
CONFIG_KEY = "sglang_jax_expert_format"
COMPLETE = "static-fp8-complete.json"
CHUNK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def validate_static_checkpoint(checkpoint, *, verify_files=False):
    """Require a complete publication and its exact config/index, optionally hash all shards."""
    path = Path(checkpoint)
    proof = json.loads((path / COMPLETE).read_text())
    if proof.get("format") != FORMAT:
        raise ValueError("unsupported static checkpoint format")
    for name, digest in proof["metadata_sha256"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"static checkpoint metadata mismatch: {name}")
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index.values()) != set(proof["shards"]):
        raise ValueError("static checkpoint shard coverage mismatch")
    for name, record in proof["shards"].items():
        if (
            Path(name).name != name
            or not (path / name).is_file()
            or (path / name).stat().st_size != record["bytes"]
        ):
            raise ValueError(f"static checkpoint shard missing or wrong size: {name}")
        if verify_files and sha256(path / name) != record["sha256"]:
            raise ValueError(f"static checkpoint shard checksum mismatch: {name}")
    return proof
