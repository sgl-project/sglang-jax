"""Minimal exploratory probe: which read paths survive the metadata bug?

Run on D side after a pull. Tries several extraction strategies and
prints which ones work vs which raise. Lives next to the round-trip test
so we can keep it as a reproducer for the JAX 0.8.1 transfer behavior.

Usage (on a pair of pods, like pd_transfer_matrix.py):

  Pod A:
    python test/srt/multi_host/disaggregation/pd_transfer_probe.py \\
      --role producer --my-host $(hostname -i) --ctl-port 31000 \\
      --transfer-port 31001

  Pod B:
    python test/srt/multi_host/disaggregation/pd_transfer_probe.py \\
      --role consumer --my-host $(hostname -i) --remote <pod-A-ip> \\
      --ctl-port 31000 --transfer-port 31001
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.jax_transfer.wrapper import get_or_create_wrapper

NELEM = 4096
UUID = "probe-0"


def _device_sharding() -> NamedSharding:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices).reshape(len(devices)), axis_names=("x",))
    return NamedSharding(mesh, P("x"))


def _producer(args):
    wrapper = get_or_create_wrapper(args.my_host, args.transfer_port)
    wrapper.start()
    print(f"[P] up {args.my_host}:{args.transfer_port}", flush=True)

    sh = _device_sharding()
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 200, size=(NELEM,), dtype=np.int32).astype(np.int16)
    # int16 bytes are exactly what bf16 would look like at the bit level.
    payload = jax.device_put(jnp.asarray(raw).view(jnp.bfloat16), sh)
    payload.block_until_ready()
    print(
        f"[P] payload ready shape={payload.shape} dtype={payload.dtype} "
        f"sharding={payload.sharding}",
        flush=True,
    )

    wrapper.register_pull(UUID, payload)
    print(f"[P] await_pull registered uuid={UUID}", flush=True)

    listen = socket.socket()
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.my_host, args.ctl_port))
    listen.listen(1)
    conn, _ = listen.accept()
    addr_msg = f"{args.my_host}:{args.transfer_port}"
    conn.sendall((addr_msg + "\n").encode("utf-8"))
    print("[P] sent addr, waiting for DONE", flush=True)
    conn.settimeout(300.0)
    ack = conn.recv(64).decode().strip()
    print(f"[P] D acked: {ack!r}", flush=True)
    return 0


def _consumer(args):
    wrapper = get_or_create_wrapper(args.my_host, args.transfer_port)
    wrapper.start()
    print(f"[D] up {args.my_host}:{args.transfer_port}", flush=True)

    s = socket.socket()
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        try:
            s.connect((args.remote, args.ctl_port))
            break
        except ConnectionRefusedError:
            time.sleep(1)
    buf = b""
    while b"\n" not in buf:
        buf += s.recv(4096)
    p_addr = buf.decode().strip()
    print(f"[D] got P addr {p_addr}", flush=True)

    sh = _device_sharding()
    spec = jax.ShapeDtypeStruct((NELEM,), jnp.bfloat16, sharding=sh)
    arr = wrapper.pull(UUID, spec, remote_addr=p_addr)
    print(
        f"[D] pulled arr type={type(arr).__name__} shape={arr.shape} "
        f"dtype={arr.dtype} sharding={arr.sharding}",
        flush=True,
    )
    deadline = time.perf_counter() + 60
    while not arr.is_ready():
        if time.perf_counter() > deadline:
            print("[D] FAIL is_ready timeout", flush=True)
            s.sendall(b"FAIL\n")
            return 1
        time.sleep(0.001)
    print("[D] arr is_ready=True", flush=True)

    def _try(label, fn):
        try:
            v = fn()
            print(f"[probe] PASS {label}: {v!r}", flush=True)
        except Exception as e:
            print(
                f"[probe] FAIL {label}: {type(e).__name__}: " f"{str(e)[:200]}",
                flush=True,
            )

    num_shards = len(arr.addressable_shards)
    shard_size = NELEM // num_shards
    print(f"[D] num_shards={num_shards} shard_size={shard_size}", flush=True)

    # 1. addressable_data(0) shape and nbytes (just metadata reads)
    _try("addressable_data(0).shape", lambda: arr.addressable_data(0).shape)
    _try("addressable_data(0).nbytes", lambda: arr.addressable_data(0).nbytes)

    # 2. addressable_data(0)[:1] -> jax slice -> device_get
    def _slice_one():
        sub = arr.addressable_data(0)[:1]
        return np.asarray(jax.device_get(sub))

    _try("addressable_data(0)[:1] device_get", _slice_one)

    # 3. addressable_data(0)[:shard_size] -> slice to actual size
    def _slice_full_shard():
        sub = arr.addressable_data(0)[:shard_size]
        return np.asarray(jax.device_get(sub)).shape

    _try(
        "addressable_data(0)[:shard_size] device_get .shape",
        _slice_full_shard,
    )

    # 4. addressable_data(0)[:shard_size].tobytes len
    def _slice_full_shard_bytes():
        sub = arr.addressable_data(0)[:shard_size]
        return len(np.asarray(jax.device_get(sub)).tobytes())

    _try(
        "addressable_data(0)[:shard_size] tobytes len",
        _slice_full_shard_bytes,
    )

    # 5. jax.jit + slice
    def _jit_slice():
        sub = jax.jit(lambda x: x[:shard_size])(arr.addressable_data(0))
        return np.asarray(jax.device_get(sub)).shape

    _try("jax.jit(x[:shard_size]) device_get", _jit_slice)

    # 6. arr[:shard_size] on the global array
    def _global_slice():
        sub = arr[:shard_size]
        return np.asarray(jax.device_get(sub)).shape

    _try("arr[:shard_size] device_get", _global_slice)

    # 7. jnp.sum scalar
    def _scalar_sum():
        s_ = jnp.sum(arr.astype(jnp.float32))
        return float(s_)

    _try("jnp.sum(arr.astype(f32)) as float", _scalar_sum)

    # 8. addressable_data(0) + dummy op + device_get
    def _add_zero():
        sub = arr.addressable_data(0) + jnp.zeros((), dtype=arr.dtype)
        return np.asarray(jax.device_get(sub)).shape

    _try("addressable_data + 0 device_get .shape", _add_zero)

    s.sendall(b"DONE\n")
    s.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["producer", "consumer"], required=True)
    ap.add_argument("--my-host", default="0.0.0.0")
    ap.add_argument("--remote", default="")
    ap.add_argument("--ctl-port", type=int, default=31000)
    ap.add_argument("--transfer-port", type=int, default=31001)
    args = ap.parse_args()
    if args.role == "producer":
        return _producer(args)
    return _consumer(args)


if __name__ == "__main__":
    sys.exit(main())
