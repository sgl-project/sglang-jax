"""Load a checkpoint without serving, then fingerprint every addressable shard.

Run baseline and candidate in separate processes on the same topology. Fingerprint
D2H is outside the loading timer and does not gather weights across hosts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import resource
import threading
import time
from pathlib import Path

import jax
import numpy as np
import psutil
from flax import nnx
from jax.experimental import multihost_utils

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import set_global_server_args
from sgl_jax.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sgl_jax.srt.model_loader import loader as loaders
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--fingerprint", choices=("full", "none"), default="full")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--backend", default="fused")
    parser.add_argument("--coordinator-port-offset", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    range_io = None
    if args.load_format == "runai_streamer":
        # The adapter moved in the refactor. Count the same public read boundary
        # in both revisions; these are requested bytes, not network traffic.
        try:
            from sgl_jax.srt.model_loader.weights.source import RunaiWeightSource
        except ImportError:
            from sgl_jax.srt.utils.runai_utils import RunaiWeightSource
        range_io = {"calls": 0, "ranges": 0, "requested_bytes": 0}
        lock = threading.Lock()
        original_read = RunaiWeightSource.read_ranges

        def counted_read(self, ranges, **kwargs):
            with lock:
                range_io["calls"] += 1
                range_io["ranges"] += len(ranges)
                range_io["requested_bytes"] += sum(size for _, _, size in ranges)
            return original_read(self, ranges, **kwargs)

        RunaiWeightSource.read_ranges = counted_read
    rank = int(os.environ.get("FALCON_RANK", "0"))
    world = int(os.environ.get("FALCON_WORLD_SIZE", "1"))
    if world > 1:
        host, port = os.environ["FALCON_JAX_COORDINATOR_ADDRESS"].rsplit(":", 1)
        jax.distributed.initialize(
            coordinator_address=f"{host}:{int(port) + args.coordinator_port_offset}",
            num_processes=world,
            process_id=rank,
            initialization_timeout=600,
        )
    assert jax.device_count() == args.tp
    mesh = create_device_mesh([args.dp, args.tp // args.dp], [1, 1])
    out = Path(args.output) / args.label / f"rank-{rank}"
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    process = psutil.Process()
    peak = [process.memory_info().rss]

    def monitor():
        while not stop.wait(0.1):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    multihost_utils.sync_global_devices("before_weight_loading")
    start = time.perf_counter()
    extra = (
        {"concurrency": 32, "memory_limit": 4 << 30} if args.load_format == "runai_streamer" else {}
    )
    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=True,
        device="tpu",
        dtype="bfloat16",
        tp_size=args.tp,
        dp_size=args.dp,
        ep_size=args.tp,
        moe_backend=args.backend,
        page_size=256,
        context_length=None,
        chunked_prefill_size=4096,
        mem_fraction_static=0.95,
        swa_full_tokens_ratio=0.25,
        max_running_requests=128,
        attention_backend="fa",
        skip_server_warmup=True,
        nnodes=world,
        node_rank=rank,
        dist_init_addr=os.environ.get("FALCON_JAX_COORDINATOR_ADDRESS"),
        download_dir="/tmp/tpu_logs/metadata",
        load_format=args.load_format,
        model_loader_extra_config=extra,
    )
    global_server_args_dict.update(
        {key: getattr(server_args, key) for key in GLOBAL_SERVER_ARGS_KEYS}
    )
    set_global_server_args(server_args)
    config = ModelConfig.from_server_args(server_args)
    config.configure_for_serving(server_args)
    loaders.print_parameter_shardings = lambda model: None
    loader = loaders.get_model_loader(
        LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=extra,
        ),
        mesh,
    )
    load_start = time.perf_counter()
    model = loader.load_model(model_config=config)
    leaves = jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
    arrays = [
        (jax.tree_util.keystr(path), value)
        for path, value in leaves
        if isinstance(value, jax.Array)
    ]
    abstract = [
        jax.tree_util.keystr(path)
        for path, value in leaves
        if isinstance(value, jax.ShapeDtypeStruct)
    ]
    if abstract:
        raise RuntimeError(f"Unloaded abstract leaves: {abstract[:20]}")
    assert arrays
    jax.block_until_ready([value for _, value in arrays])
    end = time.perf_counter()
    stop.set()
    thread.join()
    memory = [device.memory_stats() for device in jax.local_devices()]
    times = multihost_utils.process_allgather(
        np.asarray([end - start, end - load_start], np.float32)
    )
    result = {
        "label": args.label,
        "model_path": args.model_path,
        "rank": rank,
        "hosts": world,
        "backend": jax.default_backend(),
        "devices": jax.device_count(),
        "timings_by_rank": np.asarray(times).tolist(),
        "peak_rss_bytes": peak[0],
        "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device_memory": memory,
        "array_count": len(arrays),
        "runai_range_io": range_io,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "numpy",
                "safetensors",
                "transformers",
            )
        },
        "source_sha256": hashlib.sha256(
            b"".join(
                path.read_bytes()
                for path in sorted(Path(loaders.__file__).parents[1].rglob("*.py"))
            )
        ).hexdigest(),
        "checkpoint_metadata_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(config.model_path).glob("*.json")
        },
    }
    (out / "timing.json").write_text(json.dumps(result, indent=2))
    print("WEIGHT_LOADING_TIMING " + json.dumps(result), flush=True)
    fingerprint_start = time.perf_counter()
    with (out / "manifest.jsonl").open("w") as stream:
        for name, array in arrays if args.fingerprint == "full" else []:
            for shard in array.addressable_shards:
                data = shard.data
                digest = hashlib.sha256()
                # A bounded row chunk avoids materializing a whole expert group.
                row_bytes = max(1, int(np.prod(data.shape[1:])) * data.dtype.itemsize)
                rows = max(1, (256 << 20) // row_bytes)
                for begin in range(0, data.shape[0] if data.ndim else 1, rows):
                    chunk = data if data.nbytes <= (256 << 20) else data[begin : begin + rows]
                    digest.update(np.asarray(chunk).tobytes())
                record = {
                    "name": name,
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                    "spec": str(array.sharding.spec),
                    "index": str(shard.index),
                    "sha256": digest.hexdigest(),
                }
                stream.write(json.dumps(record) + "\n")
    result["fingerprint_seconds"] = time.perf_counter() - fingerprint_start
    (out / "timing.json").write_text(json.dumps(result, indent=2))
    multihost_utils.sync_global_devices("weight_fingerprints_complete")
    print("WEIGHT_LOADING_COMPLETE " + json.dumps(result), flush=True)
    if world > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
