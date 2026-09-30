#!/usr/bin/env python3
"""Stage a THIN Kimi-K3 checkpoint: everything except the per-expert tensors.

The full release is 1.42 TiB across 96 shards and a `tpu7x-standard-4t` node has ~919 GB of
tmpfs, so it cannot be staged. But the split is extremely lopsided:

    per-expert tensors   1347 GiB   92.7 %   <- streamed at load, by byte range, EP-filtered
    everything else       107 GiB    7.3 %   <- what this script stages

107 GiB fits comfortably. So the full model becomes loadable without touching sglang-jax's generic
WeightLoader: it reads a thin local directory, and `KIMI_K3_WEIGHTS_URI=gs://...` points the MoE
expert load at the release in GCS.

Nothing is downloaded whole. Tensors are read by range straight out of the shard headers, and
written into one local safetensors file per source shard, preserving names, dtypes and shapes
exactly -- including bf16, which is why the writer is hand-rolled rather than going through
numpy (no bfloat16) or torch (an extra dependency and an extra copy).

Usage:
    python3 scripts/kimi_k3/stage_thin_checkpoint.py --src gs://<bucket>/<kimi-k3> --out /dev/shm/k3_thin
    ... --limit-shards 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

AUX = [
    "config.json",
    "generation_config.json",
    "configuration_kimi_k3.py",
    "encoding_k3.py",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "tokenization_kimi.py",
    "tokenizer_config.json",
    "tiktoken.model",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
]


def write_safetensors(path: str, tensors: list[tuple[str, str, tuple, bytes]]) -> int:
    """Write a safetensors file from ``(name, dtype_str, shape, raw_bytes)`` tuples.

    Hand-rolled on purpose: the point is to move bytes through UNINTERPRETED, so a dtype the
    local numpy/torch cannot represent (bf16, and fp8/fp4 variants) round-trips exactly instead of
    being widened or refused.
    """
    header, offset = {}, 0
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8  # safetensors wants the data block 8-byte aligned
    blob += b" " * pad

    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, _, _, raw in tensors:
            fh.write(raw)
    return 8 + len(blob) + offset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dev/shm/k3_thin")
    ap.add_argument("--src", required=True, help="gs:// directory of the released checkpoint")
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all (smoke-test knob)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".repo", "python"))
    from google.cloud import storage

    from sgl_jax.srt.layers.quantization.mxfp4_streaming import (
        EXPERT_SUFFIXES,
        ShardReader,
        list_shards,
        parse_expert_id,
        parse_gs_uri,
    )

    os.makedirs(args.out, exist_ok=True)
    client = storage.Client()
    bucket, prefix = parse_gs_uri(args.src)

    for name in AUX:
        blob = client.bucket(bucket).blob(f"{prefix}/{name}")
        if blob.exists():
            blob.download_to_filename(os.path.join(args.out, name))

    shards = list_shards(bucket, prefix, client=client)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"  {len(shards)} shards to scan", flush=True)

    kept_total = skipped_total = 0
    for i, shard in enumerate(shards, 1):
        reader = ShardReader(bucket, shard, client=client)
        tensors, skipped = [], 0
        for span in sorted(reader.spans.values(), key=lambda s: s.start):
            is_expert = parse_expert_id(span.name) is not None and span.name.endswith(
                EXPERT_SUFFIXES
            )
            if is_expert:
                skipped += span.nbytes
                continue
            raw = (
                reader._blob.download_as_bytes(start=span.start, end=span.end - 1)
                if span.nbytes
                else b""
            )
            tensors.append((span.name, span.dtype, span.shape, raw))

        kept = sum(len(t[3]) for t in tensors)
        kept_total += kept
        skipped_total += skipped
        out = os.path.join(args.out, os.path.basename(shard))
        if tensors:
            write_safetensors(out, tensors)
        print(
            f"  [{i}/{len(shards)}] {os.path.basename(shard)}: "
            f"kept {len(tensors)} tensors / {kept / 2**30:.2f} GiB, "
            f"skipped {skipped / 2**30:.2f} GiB of experts",
            flush=True,
        )

    total = kept_total + skipped_total
    print(
        f"  thin checkpoint: {kept_total / 2**30:.0f} GiB staged, "
        f"{skipped_total / 2**30:.0f} GiB of experts left in GCS "
        f"({100 * kept_total / total:.1f}% of the release)"
    )
    print(f"  serve with: --model-path {args.out}  KIMI_K3_WEIGHTS_URI={args.src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
