"""Minimal cross-process smoke test for PD transfer.

This script is manual-only and is intentionally not wired into CI yet.
It exercises one real transfer end to end:

1. prefill publishes one payload through ``JaxTransferKVManager``
2. decode pulls it with the real transfer wrapper
3. decode compares the received bytes to a locally reconstructed reference
4. prefill waits for the pull-done ack and verifies cleanup

Run it on one host with two shells or on two different hosts/pods.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from pd_transfer_matrix import (
    _accept_one,
    _arr_host_bytes,
    _connect,
    _device_sharding,
    _payload_numpy,
    _read_line,
    _replicated_sharding,
)

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager, PMetadata
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import get_or_create_wrapper
from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool

REQ_ID = "pd-smoke"
NUM_ELEMS = 4096
DTYPE_NAME = "bf16"
DTYPE = jnp.bfloat16
SEED = 42


def _wait_for_terminal(handle, timeout_s: float) -> KVPoll:
    deadline = time.perf_counter() + timeout_s
    while True:
        state = handle.poll()
        if state in (KVPoll.SUCCESS, KVPoll.FAILED):
            return state
        if time.perf_counter() > deadline:
            raise TimeoutError(f"{REQ_ID} stuck in {state.value}")
        time.sleep(0.001)


def _make_host_pool() -> QueueHostKVPool:
    mesh = Mesh(
        np.asarray(jax.local_devices()).reshape(len(jax.local_devices())),
        axis_names=("x",),
    )
    return QueueHostKVPool(
        pool_size=1,
        max_tokens_per_buffer=NUM_ELEMS,
        layer_num=1,
        kv_head_per_rank=1,
        head_dim=1,
        dtype=DTYPE,
        mesh=mesh,
        partition_spec=P(),
    )


def _prefill(args: argparse.Namespace) -> int:
    wrapper = get_or_create_wrapper(args.my_host, args.transfer_port)
    wrapper.start()
    notifier = ZmqPullNotifier("prefill", args.my_host, args.side_channel_port)
    notifier.start()
    ctl = _accept_one(args.my_host, args.ctl_port)
    rx_buf = bytearray()
    try:
        ctl.sendall(
            f"{args.my_host} {args.transfer_port} {args.side_channel_port} {int(args.use_d2h_staging)}\n".encode()
        )
        payload_sharding = _replicated_sharding() if args.use_d2h_staging else _device_sharding()
        host_pool = _make_host_pool() if args.use_d2h_staging else None
        mgr = JaxTransferKVManager(wrapper, notifier, host_pool=host_pool)
        sender = mgr.create_sender(REQ_ID)
        sender.init(kv_indices=None)
        payload_flat = jax.device_put(
            jnp.asarray(_payload_numpy(SEED, DTYPE_NAME, NUM_ELEMS)).astype(DTYPE),
            payload_sharding,
        )
        payload_flat.block_until_ready()
        payload = (
            payload_flat.reshape((NUM_ELEMS, 1, 1, 1)) if args.use_d2h_staging else payload_flat
        )
        sender.attach_payload({"kv": payload}, use_d2h_staging=args.use_d2h_staging)
        sender.send()
        status = _read_line(ctl, rx_buf).strip()
        if status != "PASS":
            print(f"[P] decode reported {status!r}", flush=True)
            return 1
        if _wait_for_terminal(sender, timeout_s=30.0) != KVPoll.SUCCESS:
            return 1
        if host_pool is not None and host_pool.available_size() != host_pool.total_size():
            print("[P] host pool leak detected", flush=True)
            return 1
        print("[P] PASS sender reached SUCCESS and cleanup completed", flush=True)
        return 0
    finally:
        ctl.close()
        notifier.stop()


def _decode(args: argparse.Namespace) -> int:
    ctl = _connect(args.remote, args.ctl_port)
    rx_buf = bytearray()
    handshake = _read_line(ctl, rx_buf)
    p_host, p_transfer_port, p_side_channel_port, p_use_d2h = handshake.split()
    p_transfer_addr = f"{p_host}:{p_transfer_port}"
    use_d2h = bool(int(p_use_d2h))
    bind_host = args.my_host or "0.0.0.0"
    wrapper = get_or_create_wrapper(bind_host, args.transfer_port)
    wrapper.start()
    notifier = ZmqPullNotifier("decode", bind_host, args.side_channel_port)
    notifier.start()
    try:
        spec = jax.ShapeDtypeStruct(
            (NUM_ELEMS, 1, 1, 1) if use_d2h else (NUM_ELEMS,),
            DTYPE,
            sharding=_replicated_sharding() if use_d2h else _device_sharding(),
        )
        mgr = JaxTransferKVManager(wrapper, notifier)
        receiver = mgr.create_receiver(REQ_ID)
        receiver.init(
            PMetadata(
                remote_addr=p_transfer_addr,
                uuid=REQ_ID,
                specs={"kv": spec},
                p_side_channel_host=p_host,
                p_side_channel_port=int(p_side_channel_port),
            )
        )
        state = _wait_for_terminal(receiver, timeout_s=180.0)
        if state != KVPoll.SUCCESS or receiver.result is None:
            ctl.sendall(f"FAIL state={state.value}\n".encode())
            return 1
        got = _arr_host_bytes(receiver.result["kv"])
        expected = _payload_numpy(SEED, DTYPE_NAME, NUM_ELEMS).tobytes()
        if got != expected:
            ctl.sendall(b"FAIL bytes-mismatch\n")
            print(
                f"[D] FAIL bytes mismatch: got={len(got)} expected={len(expected)}",
                flush=True,
            )
            return 1
        ctl.sendall(b"PASS\n")
        print("[D] PASS byte-equal transfer and ack completed", flush=True)
        return 0
    finally:
        ctl.close()
        notifier.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["prefill", "decode"], required=True)
    ap.add_argument("--my-host", required=True)
    ap.add_argument("--remote", default="")
    ap.add_argument("--ctl-port", type=int, default=31000)
    ap.add_argument("--transfer-port", type=int, default=31001)
    ap.add_argument("--side-channel-port", type=int, default=31002)
    ap.add_argument("--use-d2h-staging", action="store_true")
    args = ap.parse_args()
    if args.role == "prefill":
        return _prefill(args)
    if not args.remote:
        raise ValueError("--remote is required for decode role")
    return _decode(args)


if __name__ == "__main__":
    sys.exit(main())
