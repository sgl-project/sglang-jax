"""Offline converter: FP8 block-quantized HF checkpoint -> int4 per-channel routed experts.

Only ``model.layers.<L>.mlp.experts.<e>.{gate,up,down}_proj`` tensors of the
selected layers change; every other tensor (attention, dense MLP, shared
experts, gates, norms, embeddings, MTP draft layers) is copied byte-for-byte,
so the result still loads with ``quant_method: fp8`` for the rest of the model.

Per converted projection the output shard holds ``<p>.weight`` as int8 codes
``[out, in]`` in ``[-7, 7]`` and ``<p>.weight_scale`` f32 ``[out, 1]`` (the
``weight_scale_inv`` block sidecar is dropped). See
``sgl_jax.srt.utils.quantization.int4_experts`` for the exact rounding.

Shards are independent, so the work can be split across processes/pods:

    python -m sgl_jax.tools.convert_moe_experts_int4 --input-dir IN --output-dir OUT \
        --shard-index 0 --num-shards 8        # each worker converts every 8th shard
    python -m sgl_jax.tools.convert_moe_experts_int4 --input-dir IN --output-dir OUT --finalize

``--finalize`` merges the per-shard index fragments into
``model.safetensors.index.json``, copies the non-tensor files (config.json,
tokenizer, ...) and records the expert format under
``config.json:quantization_config.routed_experts``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# safetensors' numpy backend resolves F8 dtypes through these numpy attributes
# (same shim as sgl_jax.srt.utils.weight_utils).
if not hasattr(np, "float8_e4m3fn"):
    np.float8_e4m3fn = ml_dtypes.float8_e4m3fn
if not hasattr(np, "float8_e5m2"):
    np.float8_e5m2 = ml_dtypes.float8_e5m2

from sgl_jax.srt.utils.quantization.int4_experts import (
    FP8_BLOCK_SCALE_SUFFIX,
    INT4_SCALE_SUFFIX,
    convert_fp8_block_expert_to_int4,
    match_routed_expert_weight,
)

INDEX_NAME = "model.safetensors.index.json"
PARTS_DIR = "_int4_index_parts"
ROUTED_EXPERTS_CONFIG_KEY = "routed_experts"


def _load_index(input_dir: str) -> dict:
    with open(os.path.join(input_dir, INDEX_NAME)) as f:
        return json.load(f)


def _load_config(input_dir: str) -> dict:
    with open(os.path.join(input_dir, "config.json")) as f:
        return json.load(f)


def _block_size(config: dict) -> tuple[int, int]:
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "fp8":
        raise ValueError(f"expected an FP8 checkpoint, got quantization_config={qc}")
    wbs = qc.get("weight_block_size")
    if not (isinstance(wbs, list) and len(wbs) == 2):
        raise ValueError(f"quantization_config.weight_block_size must be [n, k], got {wbs}")
    return int(wbs[0]), int(wbs[1])


def _shard_files(index: dict) -> list[str]:
    return sorted(set(index["weight_map"].values()))


def convert_shard(
    input_dir: str,
    output_dir: str,
    filename: str,
    *,
    max_layer: int,
    block_size: tuple[int, int],
    workers: int,
) -> dict:
    """Converts one shard; returns ``{tensor_name: nbytes}`` for the index."""
    src = os.path.join(input_dir, filename)
    dst = os.path.join(output_dir, filename)
    t0 = time.monotonic()
    tensors: dict[str, np.ndarray] = {}
    jobs = []
    with safe_open(src, framework="np") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        key_set = set(keys)
        for key in keys:
            matched = match_routed_expert_weight(key)
            if matched is not None and matched[1] < max_layer:
                prefix = matched[0]
                scale_key = prefix + FP8_BLOCK_SCALE_SUFFIX
                if scale_key not in key_set:
                    raise KeyError(f"{key}: missing {scale_key} in {filename}")
                jobs.append((prefix, f.get_tensor(key), f.get_tensor(scale_key)))
            elif key.endswith(FP8_BLOCK_SCALE_SUFFIX):
                base = key[: -len(FP8_BLOCK_SCALE_SUFFIX)] + ".weight"
                matched = match_routed_expert_weight(base)
                if matched is not None and matched[1] < max_layer:
                    continue  # replaced by the per-channel scale below
                tensors[key] = f.get_tensor(key)
            else:
                tensors[key] = f.get_tensor(key)

    def _convert(job):
        prefix, weight, scale_inv = job
        codes, scale = convert_fp8_block_expert_to_int4(weight, scale_inv, block_size)
        return prefix, codes, scale

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for prefix, codes, scale in ex.map(_convert, jobs):
            tensors[prefix + ".weight"] = codes
            tensors[prefix + INT4_SCALE_SUFFIX] = scale

    os.makedirs(output_dir, exist_ok=True)
    save_file(tensors, dst, metadata=metadata)
    sizes = {k: int(v.nbytes) for k, v in tensors.items()}
    parts_dir = os.path.join(output_dir, PARTS_DIR)
    os.makedirs(parts_dir, exist_ok=True)
    with open(os.path.join(parts_dir, filename + ".json"), "w") as f:
        json.dump({"file": filename, "sizes": sizes}, f)
    print(
        f"[convert] {filename}: {len(jobs)} expert projections -> int4, "
        f"{len(tensors)} tensors, {sum(sizes.values()) / 1e9:.2f} GB, {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return sizes


def default_max_layer(config: dict) -> int:
    """Default conversion bound (exclusive): every routed-expert layer the serving
    quantization config covers -- the decoder layers plus the NextN (MTP) draft layers.
    The draft model loads its experts through the same routed-expert quantization
    (``routed_expert_scale_suffix``), so with the int4 YAML it expects int4 codes and
    ``.weight_scale`` sidecars; leaving the draft layers in FP8 made the MTP load fail."""
    return int(config["num_hidden_layers"]) + int(config.get("num_nextn_predict_layers") or 0)


def finalize(input_dir: str, output_dir: str, *, max_layer: int, min_layer: int) -> None:
    index = _load_index(input_dir)
    files = _shard_files(index)
    parts_dir = os.path.join(output_dir, PARTS_DIR)
    weight_map: dict[str, str] = {}
    total = 0
    missing = []
    for filename in files:
        part = os.path.join(parts_dir, filename + ".json")
        if not os.path.exists(part):
            missing.append(filename)
            continue
        with open(part) as f:
            sizes = json.load(f)["sizes"]
        for k, n in sizes.items():
            weight_map[k] = filename
            total += n
    if missing:
        raise FileNotFoundError(f"{len(missing)} shard(s) not converted yet, e.g. {missing[:3]}")

    with open(os.path.join(output_dir, INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".safetensors") or name == INDEX_NAME or name == "config.json":
            continue
        src = os.path.join(input_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))

    config = _load_config(input_dir)
    config["quantization_config"][ROUTED_EXPERTS_CONFIG_KEY] = {
        "quant_method": "int4",
        "scheme": "per_channel_symmetric",
        "storage_dtype": "int8",
        "scale_key": INT4_SCALE_SUFFIX.lstrip("."),
        "layers": [min_layer, max_layer - 1],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(
        f"[finalize] {len(files)} shards, {len(weight_map)} tensors, {total / 1e9:.2f} GB, "
        f"routed experts int4 for layers [{min_layer}, {max_layer - 1}]",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="split the shard list across workers")
    ap.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="convert routed experts of layers < MAX_LAYER (default: num_hidden_layers + "
        "num_nextn_predict_layers, i.e. the NextN/MTP draft layers too: the serving int4 "
        "YAML applies to every routed-expert layer, so the draft loader expects int4 codes "
        "and .weight_scale sidecars as well)",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="convert exactly these shard files (overrides --shard-index/--num-shards)",
    )
    ap.add_argument("--finalize", action="store_true", help="write the merged index and config")
    ap.add_argument(
        "--list", action="store_true", help="print the shards this worker would convert"
    )
    args = ap.parse_args(argv)

    config = _load_config(args.input_dir)
    max_layer = args.max_layer if args.max_layer is not None else default_max_layer(config)
    min_layer = int(config.get("first_k_dense_replace", 0))

    if args.finalize:
        finalize(args.input_dir, args.output_dir, max_layer=max_layer, min_layer=min_layer)
        return 0

    block_size = _block_size(config)
    files = _shard_files(_load_index(args.input_dir))
    if args.files:
        unknown = sorted(set(args.files) - set(files))
        if unknown:
            raise ValueError(f"not in {INDEX_NAME}: {unknown}")
        mine = list(args.files)
    else:
        mine = files[args.shard_index :: args.num_shards]
    if args.list:
        print("\n".join(mine))
        return 0
    for filename in mine:
        convert_shard(
            args.input_dir,
            args.output_dir,
            filename,
            max_layer=max_layer,
            block_size=block_size,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
