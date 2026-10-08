"""Minimal cross-process smoke test for PD transfer.

This script is manual-only and is intentionally not wired into CI yet.
It exercises one real transfer end to end:

1. prefill publishes one payload through ``JaxTransferKVManager``
2. decode pulls it with the real transfer wrapper
3. decode compares the received bytes to a locally reconstructed reference
4. prefill waits for the pull-done ack and verifies cleanup

With ``--hybrid`` (pass it on both sides) the payload is the hybrid-model dual
payload: ``kv`` plus one request's linear-attention ``recurrent_state`` and
``conv_state`` gathered from a real ``RecurrentStatePool`` (Qwen3.5 GDN shapes
by default). Decode scatters the received state into a *different* slot of its
own pool and checks that slot byte-for-byte and that no other slot changed.

Run it on one host with two shells or on two different hosts/pods.
"""

from __future__ import annotations

import argparse
import faulthandler
import ipaddress
import logging
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
    _host_port,
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

# Hybrid (--hybrid) recurrent pool: P sends its slot P_SLOT, D receives into
# D_SLOT so a wrong slot mapping cannot pass the byte check.
HYBRID_POOL_SIZE = 4
HYBRID_P_SLOT = 2
HYBRID_D_SLOT = 3


def _wait_for_terminal(handle, timeout_s: float) -> KVPoll:
    deadline = time.perf_counter() + timeout_s
    while True:
        state = handle.poll()
        if state in (KVPoll.SUCCESS, KVPoll.FAILED):
            return state
        if time.perf_counter() > deadline:
            # Show where each thread is (e.g. the pull worker blocked in the
            # native connect()/pull()) before giving up.
            print(f"[diag] {REQ_ID} stuck in {state.value}; thread stacks:", flush=True)
            faulthandler.dump_traceback(all_threads=True)
            raise TimeoutError(f"{REQ_ID} stuck in {state.value}")
        time.sleep(0.001)


def _make_host_pool() -> QueueHostKVPool:
    mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1), axis_names=("x",))
    return QueueHostKVPool(
        pool_size=1,
        max_padded_pages=NUM_ELEMS,
        layer_num=1,
        per_layer_shape=(1, 1, 1),
        dtype=DTYPE,
        mesh=mesh,
        partition_spec=P(),
        pool_name="pd_smoke",
    )


# ---------------------------------------------------------------------------
# Hybrid recurrent state helpers
# ---------------------------------------------------------------------------


def _make_recurrent_pool(args: argparse.Namespace):
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

    devices = np.asarray(jax.local_devices())
    mesh = Mesh(
        devices.reshape(1, len(devices)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,  # as in production
    )
    return RecurrentStatePool(
        linear_recurrent_layer_ids=list(range(args.recurrent_layers)),
        size=HYBRID_POOL_SIZE,
        num_heads=args.recurrent_num_heads,
        head_dim=args.recurrent_head_dim,
        conv_kernel_size=args.conv_kernel_size,
        mesh=mesh,
        dp_size=1,
    )


def _slot_reference(shape, dtype, layer: int, salt: int) -> np.ndarray:
    """Deterministic per-(layer, group) slot content, reproducible on both sides."""

    rng = np.random.default_rng(SEED * 1000 + layer * 10 + salt)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype=dtype)


def _fill_prefill_slot(rsp) -> None:
    """Write the reference state into P's slot; every other slot stays zero."""

    from sgl_jax.srt.disaggregation.common.recurrent_state import (
        global_recurrent_slot,
        scatter_recurrent_slot,
    )

    rec = [
        _slot_reference(buf.shape[1:], buf.dtype, layer, 0)
        for layer, buf in enumerate(rsp.recurrent_buffers)
    ]
    conv = [
        [_slot_reference(c.shape[1:], c.dtype, layer, 1 + i) for i, c in enumerate(inner)]
        for layer, inner in enumerate(rsp.conv_buffers)
    ]
    scatter_recurrent_slot(rsp, global_recurrent_slot(rsp, HYBRID_P_SLOT, 0), rec, conv)


def _host_bytes(arr) -> bytes:
    return np.asarray(jax.device_get(arr)).tobytes()


def _check_decode_pool(rsp) -> str | None:
    """Return an error string, or ``None`` if D's pool holds exactly the reference."""

    from sgl_jax.srt.disaggregation.common.recurrent_state import global_recurrent_slot

    g = global_recurrent_slot(rsp, HYBRID_D_SLOT, 0)
    groups = [
        ("recurrent_state", layer, 0, buf) for layer, buf in enumerate(rsp.recurrent_buffers)
    ] + [
        ("conv_state", layer, 1 + i, c)
        for layer, inner in enumerate(rsp.conv_buffers)
        for i, c in enumerate(inner)
    ]
    for name, layer, salt, buf in groups:
        host = np.asarray(jax.device_get(buf))
        expected = _slot_reference(buf.shape[1:], buf.dtype, layer, salt)
        if host[g].tobytes() != _host_bytes(expected):
            return f"{name} layer {layer}: slot bytes mismatch"
        if np.delete(host, g, axis=0).astype(np.float32).any():
            return f"{name} layer {layer}: a non-target slot was modified"
    return None


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


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
        # D2H staging stages a per-layer list of (pages, *per_layer_shape) arrays.
        kv = [payload_flat.reshape((NUM_ELEMS, 1, 1, 1))] if args.use_d2h_staging else payload_flat
        payload = {"kv": kv}
        if args.hybrid:
            from sgl_jax.srt.disaggregation.common.recurrent_state import (
                gather_recurrent_slot,
                global_recurrent_slot,
            )

            rsp = _make_recurrent_pool(args)
            _fill_prefill_slot(rsp)
            payload.update(gather_recurrent_slot(rsp, global_recurrent_slot(rsp, HYBRID_P_SLOT, 0)))
            jax.block_until_ready(payload)
        buffer_id = host_pool.reserve() if host_pool is not None else None
        sender.attach_payload(payload, use_d2h_staging=args.use_d2h_staging, buffer_id=buffer_id)
        sender.send()
        status = _read_line(ctl, rx_buf).strip()
        if status != "PASS":
            print(f"[P] decode reported {status!r}", flush=True)
            return 1
        if _wait_for_terminal(sender, timeout_s=30.0) != KVPoll.SUCCESS:
            return 1
        if host_pool is not None:
            # In the server the prefill terminal callback owns the release.
            host_pool.release(buffer_id)
            if host_pool.available_size() != host_pool.total_size():
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
    p_transfer_addr = _host_port(p_host, p_transfer_port)
    use_d2h = bool(int(p_use_d2h))
    print(
        f"[D] handshake: pulling from {p_transfer_addr} "
        f"(use_d2h={use_d2h}, hybrid={args.hybrid}, local_devices={jax.local_device_count()})",
        flush=True,
    )
    bind_host = args.my_host or "0.0.0.0"
    wrapper = get_or_create_wrapper(bind_host, args.transfer_port)
    wrapper.start()
    notifier = ZmqPullNotifier("decode", bind_host, args.side_channel_port)
    notifier.start()
    try:
        kv_spec = jax.ShapeDtypeStruct(
            (NUM_ELEMS, 1, 1, 1) if use_d2h else (NUM_ELEMS,),
            DTYPE,
            sharding=_replicated_sharding() if use_d2h else _device_sharding(),
        )
        specs = {"kv": [kv_spec] if use_d2h else kv_spec}
        rsp = None
        if args.hybrid:
            from sgl_jax.srt.disaggregation.common.recurrent_state import (
                build_recurrent_state_specs,
            )

            rsp = _make_recurrent_pool(args)
            specs.update(build_recurrent_state_specs(rsp))
        mgr = JaxTransferKVManager(wrapper, notifier)
        receiver = mgr.create_receiver(REQ_ID)
        receiver.init(
            PMetadata(
                remote_addr=p_transfer_addr,
                uuid=REQ_ID,
                specs=specs,
                p_side_channel_host=p_host,
                p_side_channel_port=int(p_side_channel_port),
            )
        )
        state = _wait_for_terminal(receiver, timeout_s=180.0)
        if state != KVPoll.SUCCESS or receiver.result is None:
            ctl.sendall(f"FAIL state={state.value}\n".encode())
            return 1
        result = receiver.result
        kv = result["kv"][0] if use_d2h else result["kv"]
        got = _arr_host_bytes(kv.reshape((NUM_ELEMS,)) if use_d2h else kv)
        expected = _payload_numpy(SEED, DTYPE_NAME, NUM_ELEMS).tobytes()
        if got != expected:
            ctl.sendall(b"FAIL bytes-mismatch\n")
            print(
                f"[D] FAIL bytes mismatch: got={len(got)} expected={len(expected)}",
                flush=True,
            )
            return 1
        if rsp is not None:
            from sgl_jax.srt.disaggregation.common.recurrent_state import (
                global_recurrent_slot,
                scatter_recurrent_slot,
            )

            scatter_recurrent_slot(
                rsp,
                global_recurrent_slot(rsp, HYBRID_D_SLOT, 0),
                result["recurrent_state"],
                result["conv_state"],
            )
            error = _check_decode_pool(rsp)
            if error is not None:
                ctl.sendall(b"FAIL recurrent-state-mismatch\n")
                print(f"[D] FAIL {error}", flush=True)
                return 1
        ctl.sendall(b"PASS\n")
        what = "kv + recurrent/conv state" if rsp is not None else "kv"
        print(f"[D] PASS byte-equal transfer ({what}) and ack completed", flush=True)
        return 0
    finally:
        ctl.close()
        notifier.stop()


def _warn_if_loopback(host: str) -> None:
    """Flag loopback hosts, e.g. ``hostname -i`` resolving to ``127.0.1.1``."""

    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return  # DNS name; assume routable
    if addr.is_loopback:
        print(
            f"[warn] --my-host {host} is a loopback address; a peer on another "
            "host cannot reach it. Pass a routable IP instead, e.g. "
            "--my-host $(hostname -I | awk '{print $1}').",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["prefill", "decode"], required=True)
    ap.add_argument("--my-host", required=True)
    ap.add_argument("--remote", default="")
    ap.add_argument("--ctl-port", type=int, default=31000)
    ap.add_argument("--transfer-port", type=int, default=31001)
    ap.add_argument("--side-channel-port", type=int, default=31002)
    ap.add_argument("--use-d2h-staging", action="store_true")
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="also transfer recurrent/conv state (pass on both roles)",
    )
    # Qwen3.5 GDN defaults.
    ap.add_argument("--recurrent-layers", type=int, default=2)
    ap.add_argument("--recurrent-num-heads", type=int, default=16)
    ap.add_argument("--recurrent-head-dim", type=int, default=128)
    ap.add_argument("--conv-kernel-size", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(name)s] %(message)s")
    _warn_if_loopback(args.my_host)
    if args.role == "prefill":
        return _prefill(args)
    if not args.remote:
        raise ValueError("--remote is required for decode role")
    return _decode(args)


if __name__ == "__main__":
    sys.exit(main())
