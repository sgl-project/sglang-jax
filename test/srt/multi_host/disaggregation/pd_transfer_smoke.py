"""Minimal cross-process smoke test for PD transfer.

This script is manual-only and is intentionally not wired into CI yet.
It exercises one real transfer end to end with the selected ``--engine``
(pass the same value on both sides):

``jax_transfer`` (default):

1. prefill publishes one payload through ``JaxTransferKVManager``
2. decode pulls it with the real transfer wrapper
3. decode compares the received bytes to a locally reconstructed reference
4. prefill waits for the pull-done ack and verifies cleanup

``raiden``:

1. both sides register a paged KV cache (production fused-buffer layout) with
   ``RaidenTransferKVManager``; prefill fills a few pages with reference data
2. prefill starts a real ``BootstrapServer`` and publishes the transfer
   (block IDs) through ``start_prefill``
3. decode looks the transfer up via the bootstrap server and reads the pages
   into *different* pages of its own cache with ``try_start_decode``
4. decode checks those pages byte-for-byte and that no other page changed;
   prefill waits for the sender to reach SUCCESS

With ``--hybrid`` (pass it on both sides) the payload also carries one
request's linear-attention ``recurrent_state`` and ``conv_state`` from a real
``RecurrentStatePool`` (Qwen3.5 GDN shapes by default). Decode receives the
state into a *different* slot of its own pool and checks that slot
byte-for-byte and that no other slot changed.

Run it on one host with two shells or on two different hosts/pods.
"""

from __future__ import annotations

import argparse
import faulthandler
import ipaddress
import json
import logging
import select
import socket
import sys
import time


def _requested_engine(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--engine" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--engine="):
            return arg.split("=", 1)[1]
    return "jax_transfer"


if __name__ == "__main__" and _requested_engine(sys.argv[1:]) == "raiden":
    # tpu-raiden's native runtime must be loaded before jax/jaxlib.
    from sgl_jax.raiden import preload_raiden

    preload_raiden()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
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
            _print_result_readiness(handle)
            raise TimeoutError(f"{REQ_ID} stuck in {state.value}")
        time.sleep(0.001)


def _print_result_readiness(handle) -> None:
    """Decode only: did ``pull()`` return, and which arrays are still not ready?"""

    if not hasattr(handle, "_results"):
        return
    results = handle._results
    if results is None:
        print("[diag] pull() has not returned results yet", flush=True)
        return
    for name, tree in results.items():
        leaves = jax.tree.leaves(tree)
        ready = sum(leaf.is_ready() for leaf in leaves)
        print(
            f"[diag] pull() returned; {name}: {ready}/{len(leaves)} arrays ready",
            flush=True,
        )


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


# ---------------------------------------------------------------------------
# Raiden engine (--engine raiden)
# ---------------------------------------------------------------------------

# Paged KV cache in the production fused layout
# (pages, page_size, heads * 2 // packing, packing, head_dim). P exposes pages
# RAIDEN_P_PAGES; D reads them into RAIDEN_D_PAGES so a wrong page mapping
# cannot pass the byte check.
RAIDEN_NUM_PAGES = 8
RAIDEN_PAGE_SIZE = 16
RAIDEN_KV_LAYERS = 2
RAIDEN_HEAD_DIM = 128
RAIDEN_P_PAGES = (1, 2, 3)
RAIDEN_D_PAGES = (4, 5, 6)
RAIDEN_NUM_SLOTS = 2  # concurrent transfer slots per engine
RAIDEN_BOOTSTRAP_ROOM = 4242


def _raiden_kv_shape() -> tuple[int, ...]:
    packing = 2  # bf16
    # Dim 2 is sharded over "tensor", so make it one head pair per local device.
    heads = jax.local_device_count()
    return (RAIDEN_NUM_PAGES, RAIDEN_PAGE_SIZE, heads * 2 // packing, packing, RAIDEN_HEAD_DIM)


def _raiden_page_reference(layer: int) -> np.ndarray:
    """Deterministic content of the transferred pages, reproducible on both sides."""

    shape = (len(RAIDEN_P_PAGES), *_raiden_kv_shape()[1:])
    rng = np.random.default_rng(SEED * 7919 + layer)
    return np.asarray(jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype=DTYPE))


def _make_raiden_kv_caches(*, fill: bool) -> list[jax.Array]:
    """Zeroed per-layer KV caches; ``fill`` writes the reference into P's pages."""

    devices = np.asarray(jax.local_devices())
    mesh = Mesh(devices.reshape(1, len(devices)), ("data", "tensor"))
    sharding = NamedSharding(mesh, P("data", None, "tensor", None, None))
    caches = []
    for layer in range(RAIDEN_KV_LAYERS):
        host = np.zeros(_raiden_kv_shape(), dtype=jnp.dtype(DTYPE))
        if fill:
            host[list(RAIDEN_P_PAGES)] = _raiden_page_reference(layer)
        caches.append(jax.device_put(host, sharding))
    # Raiden reads/writes the raw device buffers, so they must be materialized
    # before registration and must not be replaced afterwards.
    jax.block_until_ready(caches)
    return caches


def _check_raiden_kv(caches: list[jax.Array]) -> str | None:
    """Return an error string, or ``None`` if D's cache holds exactly the reference."""

    for layer, cache in enumerate(caches):
        host = np.asarray(jax.device_get(cache))
        if host[list(RAIDEN_D_PAGES)].tobytes() != _raiden_page_reference(layer).tobytes():
            return f"kv layer {layer}: page bytes mismatch"
        if np.delete(host, list(RAIDEN_D_PAGES), axis=0).astype(np.float32).any():
            return f"kv layer {layer}: a non-target page was modified"
    return None


def _start_raiden(args: argparse.Namespace, kv_caches: list[jax.Array], rsp):
    """Register the KV caches (and recurrent buffers) with Raiden, as the factory does."""

    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import (
        get_or_create_raiden_wrapper,
    )

    wrapper = get_or_create_raiden_wrapper(args.my_host, args.transfer_port)
    wrapper.start(
        kv_caches,
        max_blocks=len(RAIDEN_P_PAGES),
        num_slots=RAIDEN_NUM_SLOTS,
        dp_size=1,
    )
    if rsp is not None:
        from sgl_jax.srt.disaggregation.common.recurrent_state import (
            CONV_STATE_KEY,
            RECURRENT_STATE_KEY,
        )

        wrapper.start_recurrent(
            {
                RECURRENT_STATE_KEY: list(rsp.recurrent_buffers),
                CONV_STATE_KEY: [buf for inner in rsp.conv_buffers for buf in inner],
            },
            num_slots=RAIDEN_NUM_SLOTS,
            dp_size=1,
        )
    return wrapper


def _prefill_raiden(args: argparse.Namespace) -> int:
    from sgl_jax.srt.disaggregation.base.transfer import PrefillTransferContext
    from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient, BootstrapServer
    from sgl_jax.srt.disaggregation.raiden_transfer.conn import RaidenTransferKVManager

    rsp = None
    if args.hybrid:
        rsp = _make_recurrent_pool(args)
        # Fill before registration: Raiden aliases the buffers it is given.
        _fill_prefill_slot(rsp)
        jax.block_until_ready((rsp.recurrent_buffers, rsp.conv_buffers))
    kv_caches = _make_raiden_kv_caches(fill=True)
    wrapper = _start_raiden(args, kv_caches, rsp)

    # "::" is dual-stack, and BootstrapServer probes readiness on loopback.
    bootstrap = BootstrapServer(host="::", port=args.bootstrap_port)
    bootstrap.start()
    client = BootstrapClient(f"http://{_host_port(args.my_host, args.bootstrap_port)}")
    mgr = RaidenTransferKVManager(wrapper, client)

    transfer = mgr.start_prefill(
        PrefillTransferContext(
            req_id=REQ_ID,
            transfer_id=REQ_ID,
            bootstrap_room=RAIDEN_BOOTSTRAP_ROOM,
            dp_rank=0,
            buffer_id=None,
            payload_factory=lambda: {"kv": kv_caches},
            block_ids_factory=lambda: list(RAIDEN_P_PAGES),
            recurrent_slot_idx=HYBRID_P_SLOT if args.hybrid else None,
        )
    )
    sender = transfer.sender

    ctl = _accept_one(args.my_host, args.ctl_port)
    rx_buf = bytearray()
    try:
        handshake = {
            "host": args.my_host,
            "bootstrap_port": args.bootstrap_port,
            "transport_metadata": mgr.prefill_transport_metadata(0),
            "hybrid": bool(args.hybrid),
        }
        ctl.sendall((json.dumps(handshake) + "\n").encode())
        # The sender settles once decode's read completes; stop early if decode
        # reports a failure first.
        status: str | None = None
        sender_state = sender.poll()
        deadline = time.perf_counter() + 180.0
        while sender_state not in (KVPoll.SUCCESS, KVPoll.FAILED):
            if status is None and select.select([ctl], [], [], 0)[0]:
                status = _read_line(ctl, rx_buf).strip()
                if status != "PASS":
                    break
            if time.perf_counter() > deadline:
                print(f"[diag] {REQ_ID} sender stuck in {sender_state.value}", flush=True)
                break
            time.sleep(0.001)
            sender_state = sender.poll()
        if status is None:
            status = _read_line(ctl, rx_buf).strip()
        if status != "PASS":
            print(f"[P] decode reported {status!r} (sender={sender_state.value})", flush=True)
            return 1
        if sender_state != KVPoll.SUCCESS:
            print(f"[P] FAIL sender ended in {sender_state.value}", flush=True)
            return 1
        print("[P] PASS sender reached SUCCESS (raiden)", flush=True)
        _wait_for_peer_exit(ctl, timeout_s=60.0)
        return 0
    finally:
        ctl.close()
        bootstrap.stop()


def _wait_for_peer_exit(sock: socket.socket, timeout_s: float) -> None:
    """Block until decode's process exits (its control socket reaches EOF).

    Prefill holds a Raiden connection into decode's transport server. If
    prefill exits first, decode's server logs ``ProcessPeerRequest failed:
    recv eof`` after its PASS line. Decode leaves its control socket for the
    kernel to close at process exit, so EOF here means decode is gone.
    """

    sock.settimeout(timeout_s)
    try:
        while sock.recv(4096):
            pass
    except (TimeoutError, OSError) as exc:
        print(f"[P] decode did not exit within {timeout_s:.0f}s ({exc}); exiting", flush=True)


def _decode_raiden(args: argparse.Namespace) -> int:
    from sgl_jax.srt.disaggregation.base.transfer import (
        AdmissionState,
        DecodeMetadataContext,
        DecodeTransferContext,
    )
    from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient
    from sgl_jax.srt.disaggregation.raiden_transfer.conn import RaidenTransferKVManager

    rsp = _make_recurrent_pool(args) if args.hybrid else None
    kv_caches = _make_raiden_kv_caches(fill=False)
    wrapper = _start_raiden(args, kv_caches, rsp)

    ctl = _connect(args.remote, args.ctl_port)
    rx_buf = bytearray()
    try:
        handshake = json.loads(_read_line(ctl, rx_buf))
        if bool(handshake["hybrid"]) != bool(args.hybrid):
            ctl.sendall(b"FAIL hybrid-flag-mismatch\n")
            print("[D] FAIL pass --hybrid on both sides or on neither", flush=True)
            return 1
        p_host = str(handshake["host"])
        bootstrap_url = f"http://{_host_port(p_host, handshake['bootstrap_port'])}"
        print(
            f"[D] handshake: prefill={p_host} bootstrap={bootstrap_url} "
            f"endpoints={handshake['transport_metadata'].get('endpoints')} "
            f"hybrid={args.hybrid} local_devices={jax.local_device_count()}",
            flush=True,
        )
        mgr = RaidenTransferKVManager(wrapper, BootstrapClient(bootstrap_url))
        peer_info = {"host": p_host, "transport_metadata": handshake["transport_metadata"]}

        meta_ctx = DecodeMetadataContext(
            req_id=REQ_ID,
            transfer_id=REQ_ID,
            bootstrap_room=RAIDEN_BOOTSTRAP_ROOM,
            prefill_dp_rank=0,
            peer_info=peer_info,
        )
        deadline = time.perf_counter() + 60.0
        while not mgr.poll_decode_metadata(meta_ctx):
            if time.perf_counter() > deadline:
                ctl.sendall(b"FAIL metadata-timeout\n")
                print("[D] FAIL transfer metadata not found in bootstrap", flush=True)
                return 1
            time.sleep(0.01)

        # Token slots of D's destination pages, as the scheduler would pass them.
        kv_indices = np.concatenate(
            [np.arange(p * RAIDEN_PAGE_SIZE, (p + 1) * RAIDEN_PAGE_SIZE) for p in RAIDEN_D_PAGES]
        )
        admission = mgr.try_start_decode(
            DecodeTransferContext(
                req_id=REQ_ID,
                transfer_id=REQ_ID,
                bootstrap_room=RAIDEN_BOOTSTRAP_ROOM,
                decode_dp_rank=0,
                prefill_dp_rank=0,
                peer_info=peer_info,
                kv_indices=kv_indices,
                page_size=RAIDEN_PAGE_SIZE,
                prompt_tokens=int(kv_indices.size),
                spec_factory=lambda: None,
                recurrent_slot_idx=HYBRID_D_SLOT if args.hybrid else None,
            )
        )
        if admission.state != AdmissionState.ADMITTED or admission.receiver is None:
            ctl.sendall(f"FAIL admission={admission.reason}\n".encode())
            print(f"[D] FAIL decode not admitted: {admission.reason}", flush=True)
            return 1
        state = _wait_for_terminal(admission.receiver, timeout_s=180.0)
        if state != KVPoll.SUCCESS:
            ctl.sendall(f"FAIL state={state.value}\n".encode())
            return 1
        error = _check_raiden_kv(kv_caches)
        if error is None and rsp is not None:
            error = _check_decode_pool(rsp)
        if error is not None:
            ctl.sendall(b"FAIL bytes-mismatch\n")
            print(f"[D] FAIL {error}", flush=True)
            return 1
        ctl.sendall(b"PASS\n")
        what = "kv + recurrent/conv state" if rsp is not None else "kv"
        print(f"[D] PASS byte-equal transfer ({what}) via raiden", flush=True)
        return 0
    finally:
        # Do not close: the kernel closes the fd at process exit, after the
        # Raiden threads are gone. Prefill waits for that EOF before exiting
        # (see _wait_for_peer_exit).
        ctl.detach()


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
    ap.add_argument(
        "--engine",
        choices=["jax_transfer", "raiden"],
        default="jax_transfer",
        help="transfer engine under test (pass the same value on both roles)",
    )
    ap.add_argument("--my-host", required=True)
    ap.add_argument("--remote", default="")
    ap.add_argument("--ctl-port", type=int, default=31000)
    ap.add_argument(
        "--transfer-port",
        type=int,
        default=31001,
        help="jax_transfer server port, or the Raiden KV engine control port",
    )
    ap.add_argument(
        "--side-channel-port",
        type=int,
        default=31002,
        help="jax_transfer only: ZMQ pull-done ack port on prefill",
    )
    ap.add_argument(
        "--bootstrap-port",
        type=int,
        default=8998,
        help="raiden only: port of the bootstrap server prefill starts",
    )
    ap.add_argument("--use-d2h-staging", action="store_true", help="jax_transfer only")
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
    if args.engine == "raiden" and args.use_d2h_staging:
        ap.error("--use-d2h-staging selects the jax_transfer path; drop it for --engine raiden")
    if args.role == "decode" and not args.remote:
        ap.error("--remote is required for the decode role")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(name)s] %(message)s")
    _warn_if_loopback(args.my_host)
    if args.engine == "raiden":
        return _prefill_raiden(args) if args.role == "prefill" else _decode_raiden(args)
    return _prefill(args) if args.role == "prefill" else _decode(args)


if __name__ == "__main__":
    sys.exit(main())
