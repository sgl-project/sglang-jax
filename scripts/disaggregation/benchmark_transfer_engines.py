#!/usr/bin/env python3
"""Two-host KV transfer microbenchmark for JAX transfer and TPU Raiden.

The producer and consumer run this script simultaneously. A separate TCP
control connection carries readiness and completion barriers; timings start
after the producer has registered each payload and end when the consumer
observes device-ready completion, so control coordination is excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import time
import zlib
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("jax", "raiden"), required=True)
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--producer-host", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--blocks", default="1,4,16,32,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--packing", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--control-port", type=int, default=31000)
    parser.add_argument("--jax-port", type=int, default=31001)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--raiden-control-port", type=int, default=0)
    return parser.parse_args()


ARGS = parse_args()

# Raiden embeds XLA and must be loaded before jax/jaxlib.
if ARGS.engine == "raiden":
    from tpu_raiden.frameworks.jax import _tpu_raiden_jax  # noqa: F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _send_control(control: Any, payload: dict[str, Any]) -> None:
    control.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def _recv_control(control: Any) -> dict[str, Any]:
    line = control.readline()
    if not line:
        raise RuntimeError("control connection closed unexpectedly")
    return json.loads(line)


def _accept_control(host_ip: str) -> tuple[socket.socket, Any]:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host_ip, ARGS.control_port))
    listener.listen(1)
    listener.settimeout(ARGS.timeout_seconds)
    connection, _ = listener.accept()
    listener.close()
    connection.settimeout(ARGS.timeout_seconds)
    return connection, connection.makefile("rwb", buffering=0)


def _connect_control() -> tuple[socket.socket, Any]:
    deadline = time.monotonic() + ARGS.timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection(
                (ARGS.producer_host, ARGS.control_port),
                timeout=min(5.0, ARGS.timeout_seconds),
            )
            connection.settimeout(ARGS.timeout_seconds)
            return connection, connection.makefile("rwb", buffering=0)
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(
        f"timed out connecting to {ARGS.producer_host}:{ARGS.control_port}: {last_error}"
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _host_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


def _uuid(req_id: str) -> int:
    return zlib.crc32(req_id.encode("utf-8")) & 0xFFFFFFFF


def _make_sharding() -> jax.sharding.NamedSharding:
    devices = np.asarray(jax.devices("tpu"))
    mesh = jax.sharding.Mesh(devices.reshape((1, len(devices))), ("data", "model"))
    return jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, None, "model", None, None),
    )


def _make_stacked_sharding(
    sharding: jax.sharding.NamedSharding,
) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(
        sharding.mesh,
        jax.sharding.PartitionSpec(None, None, None, "model", None, None),
    )


def _make_caches(
    max_blocks: int,
    sharding: jax.sharding.NamedSharding,
    *,
    source: bool,
) -> list[jax.Array]:
    shape = (
        max_blocks,
        ARGS.page_size,
        ARGS.kv_heads,
        ARGS.packing,
        ARGS.head_dim,
    )
    caches = []
    for layer in range(ARGS.num_layers):
        fill_value = layer + 1 if source else 0
        caches.append(
            jax.device_put(
                jnp.full(shape, fill_value, dtype=jnp.bfloat16),
                sharding,
            )
        )
    jax.block_until_ready(caches)
    return caches


def _assert_values(arrays: Any, num_blocks: int) -> None:
    if isinstance(arrays, jax.Array):
        actual = np.asarray(
            jax.device_get(
                jnp.stack(
                    (
                        arrays[:, 0, 0, 0, 0, 0],
                        arrays[:, num_blocks - 1, 0, 0, 0, 0],
                    ),
                    axis=1,
                )
            ),
            dtype=np.float32,
        ).reshape(-1)
        expected = np.repeat(np.arange(1, ARGS.num_layers + 1, dtype=np.float32), 2)
        np.testing.assert_array_equal(actual, expected)
        return

    samples: list[Any] = []
    for array in arrays:
        samples.extend((array[0, 0, 0, 0, 0], array[num_blocks - 1, 0, 0, 0, 0]))
    actual = np.asarray(jax.device_get(samples), dtype=np.float32)
    expected = np.repeat(np.arange(1, ARGS.num_layers + 1, dtype=np.float32), 2)
    np.testing.assert_array_equal(actual, expected)


def _logical_bytes(num_blocks: int) -> int:
    return (
        ARGS.num_layers
        * num_blocks
        * ARGS.page_size
        * ARGS.kv_heads
        * ARGS.packing
        * ARGS.head_dim
        * 2
    )


def _run_producer(
    blocks: list[int],
    caches: list[jax.Array],
    sharding: jax.sharding.NamedSharding,
    host_ip: str,
    engine: Any,
) -> None:
    connection, control = _accept_control(host_ip)
    if ARGS.engine == "raiden":
        endpoints = engine.get_local_endpoints()
        _send_control(
            control,
            {"type": "endpoint", "value": endpoints, "host_ip": host_ip},
        )
    else:
        _send_control(control, {"type": "endpoint", "value": host_ip})

    stacked_caches = None
    if ARGS.engine == "jax":
        # jax.experimental.transfer's native API is exercised with one array,
        # matching the production manager's one "kv" entry. Stacking the layer
        # dimension also avoids relying on undocumented pytree support in the
        # native transfer binding.
        stacked_caches = jax.device_put(
            jnp.stack(caches, axis=0),
            _make_stacked_sharding(sharding),
        )
        stacked_caches.block_until_ready()

    for num_blocks in blocks:
        for iteration in range(ARGS.warmup + ARGS.iterations):
            req_id = f"{ARGS.engine}-{num_blocks}-{iteration}"
            if ARGS.engine == "raiden":
                engine.register_read(req_id, _uuid(req_id), list(range(num_blocks)))
                payload = caches
            else:
                assert stacked_caches is not None
                payload = stacked_caches[:, :num_blocks]
                payload.block_until_ready()
                engine.await_pull(_uuid(req_id), payload)

            _send_control(control, {"type": "ready", "req_id": req_id})
            done = _recv_control(control)
            if done != {"type": "done", "req_id": req_id}:
                raise RuntimeError(f"unexpected control response: {done}")

            if ARGS.engine == "raiden":
                deadline = time.monotonic() + ARGS.timeout_seconds
                while time.monotonic() < deadline:
                    done_sending, _, _ = engine.poll_stats()
                    if req_id in done_sending:
                        break
                    time.sleep(0.001)
                else:
                    raise TimeoutError(f"producer did not finish {req_id}")

    _send_control(control, {"type": "producer_done"})
    control.close()
    connection.close()


def _run_consumer(
    blocks: list[int],
    caches: list[jax.Array],
    sharding: jax.sharding.NamedSharding,
    engine: Any,
) -> None:
    connection, control = _connect_control()
    endpoint_message = _recv_control(control)
    if endpoint_message.get("type") != "endpoint":
        raise RuntimeError(f"unexpected endpoint message: {endpoint_message}")
    endpoint_value = endpoint_message["value"]
    if ARGS.engine == "raiden":
        endpoints = endpoint_value
        port = int(str(endpoints[0]["endpoint"]).rsplit(":", 1)[1])
        remote_endpoint: Any = f"{endpoint_message['host_ip']}:{port}"
        link = None
    else:
        remote_endpoint = None
        link = engine.connect(f"{endpoint_value}:{ARGS.jax_port}")

    rows: list[dict[str, Any]] = []
    for num_blocks in blocks:
        logical_bytes = _logical_bytes(num_blocks)
        for iteration in range(ARGS.warmup + ARGS.iterations):
            req_id = f"{ARGS.engine}-{num_blocks}-{iteration}"
            ready = _recv_control(control)
            if ready != {"type": "ready", "req_id": req_id}:
                raise RuntimeError(f"unexpected control request: {ready}")

            start = time.perf_counter()
            if ARGS.engine == "raiden":
                engine.start_read(
                    req_id=req_id,
                    uuid=_uuid(req_id),
                    remote_endpoint=remote_endpoint,
                    remote_block_ids=list(range(num_blocks)),
                    local_block_ids=list(range(num_blocks)),
                    parallelism=1,
                )
                deadline = time.monotonic() + ARGS.timeout_seconds
                while time.monotonic() < deadline:
                    _, done_recving, failed_recving = engine.poll_stats()
                    if req_id in failed_recving:
                        raise RuntimeError(f"Raiden transfer failed for {req_id}")
                    if req_id in done_recving:
                        break
                    time.sleep(0.001)
                else:
                    raise TimeoutError(f"consumer did not finish {req_id}")
                received = caches
            else:
                received = link.pull(
                    _uuid(req_id),
                    jax.ShapeDtypeStruct(
                        (
                            ARGS.num_layers,
                            num_blocks,
                            ARGS.page_size,
                            ARGS.kv_heads,
                            ARGS.packing,
                            ARGS.head_dim,
                        ),
                        jnp.bfloat16,
                        sharding=_make_stacked_sharding(sharding),
                    ),
                )
                received.block_until_ready()
            elapsed_s = time.perf_counter() - start

            measured = iteration >= ARGS.warmup
            if iteration == ARGS.warmup:
                _assert_values(received, num_blocks)
            if measured:
                rows.append(
                    {
                        "engine": ARGS.engine,
                        "blocks": num_blocks,
                        "tokens": num_blocks * ARGS.page_size,
                        "logical_bytes": logical_bytes,
                        "iteration": iteration - ARGS.warmup,
                        "latency_ms": elapsed_s * 1000.0,
                        "bandwidth_gbps": logical_bytes / elapsed_s / 1e9,
                    }
                )
            _send_control(control, {"type": "done", "req_id": req_id})

    samples_path = ARGS.artifact_root / "samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    summary: list[dict[str, Any]] = []
    for num_blocks in blocks:
        selected = [row for row in rows if row["blocks"] == num_blocks]
        latencies = [float(row["latency_ms"]) for row in selected]
        bandwidths = [float(row["bandwidth_gbps"]) for row in selected]
        summary.append(
            {
                "engine": ARGS.engine,
                "blocks": num_blocks,
                "tokens": num_blocks * ARGS.page_size,
                "logical_bytes": _logical_bytes(num_blocks),
                "iterations": len(selected),
                "latency_mean_ms": statistics.mean(latencies),
                "latency_p50_ms": _percentile(latencies, 0.50),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "latency_p99_ms": _percentile(latencies, 0.99),
                "bandwidth_mean_gbps": statistics.mean(bandwidths),
                "bandwidth_p50_gbps": _percentile(bandwidths, 0.50),
            }
        )

    (ARGS.artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (ARGS.artifact_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    producer_done = _recv_control(control)
    if producer_done != {"type": "producer_done"}:
        raise RuntimeError(f"unexpected final control message: {producer_done}")
    control.close()
    connection.close()


def main() -> None:
    ARGS.artifact_root.mkdir(parents=True, exist_ok=True)
    blocks = [int(value) for value in ARGS.blocks.split(",")]
    if not blocks or min(blocks) <= 0:
        raise ValueError("--blocks must contain positive integers")

    sharding = _make_sharding()
    caches = _make_caches(
        max(blocks),
        sharding,
        source=ARGS.role == "producer",
    )
    host_ip = _host_ip()

    if ARGS.engine == "raiden":
        from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager

        engine = KVCacheManager(
            kv_caches=caches,
            local_control_port=ARGS.raiden_control_port,
            max_blocks=max(blocks),
            num_slots=1,
            timeout_s=ARGS.timeout_seconds,
            parallelism=1,
            unsafe_skip_buffer_lock=True,
        )
    else:
        from jax.experimental.transfer import start_transfer_server

        engine = start_transfer_server(
            jax.local_devices()[0].client,
            f"{host_ip}:{ARGS.jax_port}",
            [f"{host_ip}:0"],
            max_num_parallel_copies=1,
            transfer_size=64 * 1024 * 1024,
            use_raw_buffers=False,
        )

    print(
        json.dumps(
            {
                "engine": ARGS.engine,
                "role": ARGS.role,
                "jax": jax.__version__,
                "devices": [str(device) for device in jax.devices("tpu")],
                "host_ip": host_ip,
                "blocks": blocks,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if ARGS.role == "producer":
        _run_producer(blocks, caches, sharding, host_ip, engine)
    else:
        _run_consumer(blocks, caches, sharding, engine)


if __name__ == "__main__":
    main()
