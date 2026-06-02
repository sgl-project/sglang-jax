"""Cross-pod matrix script for manual PD transfer validation.

Exercises the current sender/receiver contract:

  * Event-driven sender: ``send()`` registers a ZMQ callback and the
    receiver acks completion via ``send_done``.
  * Path A (D2H staging via :class:`QueueHostKVPool`) vs path B
    (direct from HBM), selectable via ``--use-d2h-staging``. Both
    paths produce byte-equal transfers.
  * Pipelined concurrency within each cell: P fires ``ITERATIONS``
    senders back-to-back without waiting for individual acks, D drains
    them all, then P collects all acks and verifies states.
  * Pool leak check (path A): ``available_size()`` must return to the
    initial value once every transfer in the script completes.

Usage:

  Pod A (prefill, path B):
    python test/srt/multi_host/disaggregation/pd_transfer_matrix.py \\
      --role prefill --my-host $(hostname -i) --ctl-port 31000 \\
      --transfer-port 31001 --side-channel-port 31002

  Pod B (decode, path B):
    python test/srt/multi_host/disaggregation/pd_transfer_matrix.py \\
      --role decode --my-host $(hostname -i) --remote <pod-A-ip> \\
      --ctl-port 31000 --transfer-port 31001

  For path A add ``--use-d2h-staging`` on the prefill invocation; the
  decoder picks the path up from the control-channel handshake.

Exit 0 only if all cells produce byte-equal results AND (path A only)
the host pool ends with no leaked buffers.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVReceiver,
    JaxTransferKVSender,
    PMetadata,
)
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import get_or_create_wrapper
from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool

PAGE_SIZE_TOKENS = int(os.environ.get("PD_PAGE_SIZE_TOKENS", "128"))
ITERATIONS = int(os.environ.get("PD_ROUNDTRIP_ITERS", "8"))
POOL_SIZE = int(os.environ.get("PD_POOL_SIZE", "16"))
KV_LAYER_NUM = int(os.environ.get("PD_KV_LAYER_NUM", "36"))
KV_HEADS_PER_RANK = int(os.environ.get("PD_KV_HEADS_PER_RANK", "2"))
KV_HEAD_DIM = int(os.environ.get("PD_KV_HEAD_DIM", "128"))
KV_PACKING = int(os.environ.get("PD_KV_PACKING", "2"))


def _page_counts() -> tuple[int, ...]:
    raw = os.environ.get("PD_PAGE_COUNTS")
    if not raw:
        return (1, 16)
    counts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"invalid PD_PAGE_COUNTS={raw!r}")
    return counts


@dataclass(frozen=True)
class Cell:
    dtype_name: str
    dtype: jnp.dtype
    page_count: int


def _dtypes() -> list[tuple[str, jnp.dtype]]:
    out: list[tuple[str, jnp.dtype]] = [
        ("bf16", jnp.bfloat16),
        ("fp16", jnp.float16),
    ]
    fp8 = getattr(jnp, "float8_e4m3fn", None)
    if fp8 is not None:
        out.append(("fp8_e4m3fn", fp8))
    else:
        print(
            "[warn] jnp.float8_e4m3fn not available, skipping fp8 cell",
            flush=True,
        )
    return out


def _all_cells() -> list[Cell]:
    return [Cell(name, dt, pc) for name, dt in _dtypes() for pc in _page_counts()]


def _payload_shape(num_tokens: int) -> tuple[int, int, int, int]:
    # The current path-A host-pool contract is 4-D, so fold K/V packing
    # into the last dimension while keeping the logical bytes/token close
    # to the real per-rank KV payload.
    return (
        num_tokens,
        KV_LAYER_NUM,
        KV_HEADS_PER_RANK,
        KV_HEAD_DIM * KV_PACKING,
    )


def _device_sharding() -> NamedSharding:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices).reshape(len(devices)), axis_names=("x",))
    return NamedSharding(mesh, P("x"))


def _replicated_sharding() -> NamedSharding:
    devices = jax.local_devices()
    mesh = Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))
    return NamedSharding(mesh, P())


def _payload_numpy(seed: int, dtype_name: str, num_tokens: int) -> np.ndarray:
    import ml_dtypes

    name_to_np = {
        "bf16": ml_dtypes.bfloat16,
        "fp16": np.float16,
    }
    fp8 = getattr(ml_dtypes, "float8_e4m3fn", None)
    if fp8 is not None:
        name_to_np["fp8_e4m3fn"] = fp8

    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 200, size=_payload_shape(num_tokens), dtype=np.int32)
    return raw.astype(name_to_np[dtype_name])


def _make_payload(
    seed: int,
    dtype_name: str,
    dtype: jnp.dtype,
    num_tokens: int,
    sharding: NamedSharding,
) -> jax.Array:
    np_ref = _payload_numpy(seed, dtype_name, num_tokens)
    return jax.device_put(jnp.asarray(np_ref).astype(dtype), sharding)


def _arr_host_bytes(arr: jax.Array) -> bytes:
    """Concatenate per-shard host buffers, slicing each to its actual
    size to dodge the jax 0.8.1 transfer metadata bug. See
    ``_probe_transfer_readback.py`` for the diagnostic.
    """

    n_shards = len(arr.addressable_shards)
    shard_size = arr.shape[0] // n_shards
    parts: list[bytes] = []
    for i in range(n_shards):
        sub = arr.addressable_data(i)[:shard_size]
        parts.append(np.asarray(jax.device_get(sub)).tobytes())
    return b"".join(parts)


def _dtype_itemsize(dtype: jnp.dtype) -> int:
    return np.dtype(dtype).itemsize


def _print_cell_result(
    *,
    level: str,
    path_name: str,
    dtype_name: str,
    page_count: int,
    num_iters: int,
    itemsize: int,
    elapsed_s: float,
) -> None:
    bytes_per_iter = (
        np.prod(_payload_shape(page_count * PAGE_SIZE_TOKENS), dtype=np.int64)
        * itemsize
    )
    total_bytes = bytes_per_iter * num_iters
    throughput_mib_s = total_bytes / max(elapsed_s, 1e-9) / (1024**2)
    print(
        "[RESULT] "
        f"level={level} "
        f"path={path_name} "
        f"dtype={dtype_name} "
        f"pages={page_count} "
        f"tokens={page_count * PAGE_SIZE_TOKENS} "
        f"iters={num_iters} "
        f"bytes_per_iter={bytes_per_iter} "
        f"total_bytes={total_bytes} "
        f"elapsed_ms={elapsed_s * 1e3:.1f} "
        f"throughput_mib_s={throughput_mib_s:.2f}",
        flush=True,
    )


# --- control channel --------------------------------------------------------


def _accept_one(host: str, port: int) -> socket.socket:
    listen = socket.socket()
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((host, port))
    listen.listen(1)
    print(f"[ctl] waiting for D on {host}:{port}", flush=True)
    conn, peer = listen.accept()
    print(f"[ctl] D connected from {peer}", flush=True)
    listen.close()
    return conn


def _connect(host: str, port: int, timeout_s: float = 120.0) -> socket.socket:
    deadline = time.perf_counter() + timeout_s
    last_err: Exception = RuntimeError("no attempt yet")
    while time.perf_counter() < deadline:
        try:
            s = socket.socket()
            s.settimeout(10.0)
            s.connect((host, port))
            s.settimeout(None)
            return s
        except (TimeoutError, ConnectionRefusedError) as e:
            last_err = e
            time.sleep(1.0)
    raise TimeoutError(
        f"could not connect to {host}:{port} within {timeout_s}s: {last_err}"
    )


def _read_line(sock: socket.socket, buf: bytearray) -> str:
    sock.settimeout(180.0)
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("control channel closed mid-stream")
        buf.extend(chunk)
    nl = buf.index(b"\n")
    line = bytes(buf[:nl]).decode("utf-8")
    del buf[: nl + 1]
    return line


# --- prefill / decode entries ----------------------------------------------


def _prefill(args: argparse.Namespace) -> int:
    wrapper = get_or_create_wrapper(args.my_host, args.transfer_port)
    wrapper.start()
    p_notifier = ZmqPullNotifier("prefill", args.my_host, args.side_channel_port)
    p_notifier.start()
    sharding = _device_sharding()

    print(
        f"[P] wrapper {args.my_host}:{args.transfer_port} "
        f"side_channel {args.side_channel_port} "
        f"use_d2h={args.use_d2h_staging} "
        f"pool_size={args.pool_size if args.use_d2h_staging else 'n/a'}",
        flush=True,
    )

    conn = _accept_one(args.my_host, args.ctl_port)
    handshake = (
        f"{args.my_host} {args.transfer_port} "
        f"{args.side_channel_port} {int(args.use_d2h_staging)}\n"
    )
    conn.sendall(handshake.encode("utf-8"))

    failed_cells: list[tuple[str, str]] = []
    rx_buf = bytearray()
    cells = _all_cells()
    # Path A uses replicated sharding (matches the pool's
    # partition_spec=P()) so D2H staging does not trigger a cross-chip
    # gather collective. Path B keeps the existing sharded layout.
    payload_sharding = _replicated_sharding() if args.use_d2h_staging else sharding
    leak_total = 0

    mesh = Mesh(
        np.asarray(jax.local_devices()).reshape(len(jax.local_devices())),
        axis_names=("x",),
    )

    for cell in cells:
        num_tokens = cell.page_count * PAGE_SIZE_TOKENS
        cell_t0 = time.perf_counter()
        # Per-cell pool + mgr for path A: pool dtype must match cell
        # dtype, otherwise the ``.at[:n].set(staged)`` scatter
        # implicitly down-casts and breaks byte equality. Pool buffers
        # are sized to ``nelem`` exactly so D's spec matches with no
        # zero padding.
        host_pool: QueueHostKVPool | None = None
        if args.use_d2h_staging:
            host_pool = QueueHostKVPool(
                pool_size=args.pool_size,
                max_tokens_per_buffer=num_tokens,
                layer_num=KV_LAYER_NUM,
                kv_head_per_rank=KV_HEADS_PER_RANK,
                head_dim=KV_HEAD_DIM * KV_PACKING,
                dtype=cell.dtype,
                mesh=mesh,
                partition_spec=P(),
            )
        mgr = JaxTransferKVManager(wrapper, p_notifier, host_pool=host_pool)
        initial_avail = host_pool.available_size() if host_pool else 0

        senders: list[tuple[str, JaxTransferKVSender]] = []
        # Phase 1: dispatch all ITERATIONS at once (pipelined).
        for i in range(ITERATIONS):
            req_id = f"{cell.dtype_name}-{cell.page_count}-{i}"
            seed = hash((cell.dtype_name, cell.page_count, i)) & 0xFFFFFFFF
            sender = mgr.create_sender(req_id)
            sender.init(kv_indices=None)
            if args.use_d2h_staging:
                payload_flat = _make_payload(
                    seed,
                    cell.dtype_name,
                    cell.dtype,
                    num_tokens,
                    payload_sharding,
                )
                payload_flat.block_until_ready()
                payload = payload_flat
            else:
                payload = _make_payload(
                    seed,
                    cell.dtype_name,
                    cell.dtype,
                    num_tokens,
                    payload_sharding,
                )
                payload.block_until_ready()
            sender.attach_payload({"kv": payload}, use_d2h_staging=args.use_d2h_staging)
            sender.send()
            line = f"{req_id} {num_tokens} {cell.dtype_name} {seed}\n".encode()
            conn.sendall(line)
            senders.append((req_id, sender))

        # Phase 2: collect D acks (one per iter), assert state SUCCESS.
        ok_count = 0
        cell_aborted = False
        for req_id, sender in senders:
            ack = _read_line(conn, rx_buf).strip()
            if ack != "OK":
                print(
                    f"[P] D reported {ack!r} for {req_id} in cell "
                    f"{cell.dtype_name}/{cell.page_count}",
                    flush=True,
                )
                failed_cells.append((cell.dtype_name, ack))
                cell_aborted = True
                break
            deadline = time.perf_counter() + 5.0
            while sender.poll() != KVPoll.SUCCESS:
                if time.perf_counter() > deadline:
                    raise RuntimeError(
                        f"sender {req_id} stuck at {sender.poll().value} after D acked"
                    )
                time.sleep(0.001)
            ok_count += 1
        if cell_aborted:
            break
        if host_pool is not None:
            leaked = initial_avail - host_pool.available_size()
            leak_total += leaked
            print(
                f"[P] cell {cell.dtype_name}/{cell.page_count}: "
                f"{ok_count}/{ITERATIONS} leak={leaked}",
                flush=True,
            )
        else:
            print(
                f"[P] cell {cell.dtype_name}/{cell.page_count}: "
                f"{ok_count}/{ITERATIONS}",
                flush=True,
            )
        _print_cell_result(
            level="manager",
            path_name="path-a" if args.use_d2h_staging else "path-b",
            dtype_name=cell.dtype_name,
            page_count=cell.page_count,
            num_iters=ok_count,
            itemsize=_dtype_itemsize(cell.dtype),
            elapsed_s=time.perf_counter() - cell_t0,
        )

    conn.close()
    p_notifier.stop()
    failed = bool(failed_cells) or leak_total != 0
    total = len(cells) * ITERATIONS
    print(
        f"[P] done: failed_cells={failed_cells} leaked_total={leak_total} "
        f"total_target={total}",
        flush=True,
    )
    return 0 if not failed else 1


def _decode(args: argparse.Namespace) -> int:
    print(
        f"[D] connecting to P ctl at {args.remote}:{args.ctl_port}",
        flush=True,
    )
    ctl = _connect(args.remote, args.ctl_port)
    rx_buf = bytearray()
    handshake = _read_line(ctl, rx_buf)
    p_host, p_transfer_port, p_side_channel_port, p_use_d2h = handshake.split()
    p_transfer_addr = f"{p_host}:{p_transfer_port}"
    p_side_channel_port_int = int(p_side_channel_port)
    use_d2h = bool(int(p_use_d2h))
    print(
        f"[D] handshake: p_transfer={p_transfer_addr} "
        f"p_side_channel={p_host}:{p_side_channel_port_int} "
        f"use_d2h={use_d2h}",
        flush=True,
    )

    bind_host = args.my_host or "0.0.0.0"
    wrapper = get_or_create_wrapper(bind_host, args.transfer_port)
    wrapper.start()
    d_notifier = ZmqPullNotifier("decode", bind_host, args.side_channel_port)
    d_notifier.start()
    mgr = JaxTransferKVManager(wrapper, d_notifier)
    sharding = _device_sharding()
    repl_sharding = _replicated_sharding()
    name_to_dtype = {n: dt for n, dt in _dtypes()}

    cells = _all_cells()
    expected_total = len(cells) * ITERATIONS
    successes = 0
    failed_cells: list[str] = []

    for cell in cells:
        if failed_cells:
            break
        cell_t0 = time.perf_counter()
        # Phase 1: read ITERATIONS metadata lines.
        metas: list[tuple[str, PMetadata, int, str, int]] = []
        for _ in range(ITERATIONS):
            line = _read_line(ctl, rx_buf)
            toks = line.split()
            if len(toks) != 4:
                raise RuntimeError(f"bad metadata line: {line!r}")
            req_id, nelem_s, dtype_name, seed_s = toks
            num_tokens = int(nelem_s)
            seed = int(seed_s)
            dtype = name_to_dtype[dtype_name]
            if use_d2h:
                spec = jax.ShapeDtypeStruct(
                    _payload_shape(num_tokens), dtype, sharding=repl_sharding
                )
            else:
                spec = jax.ShapeDtypeStruct(
                    _payload_shape(num_tokens), dtype, sharding=sharding
                )
            metas.append(
                (
                    req_id,
                    PMetadata(
                        remote_addr=p_transfer_addr,
                        uuid=req_id,
                        specs={"kv": spec},
                        p_side_channel_host=p_host,
                        p_side_channel_port=p_side_channel_port_int,
                    ),
                    seed,
                    dtype_name,
                    num_tokens,
                )
            )

        # Phase 2: dispatch all receivers, drain, byte-check, ack.
        receivers: list[tuple[str, JaxTransferKVReceiver, int, str, int]] = []
        for req_id, meta, seed, dtype_name, num_tokens in metas:
            receiver = mgr.create_receiver(req_id)
            receiver.init(meta)
            receivers.append((req_id, receiver, seed, dtype_name, num_tokens))

        cell_done = 0
        for req_id, receiver, seed, dtype_name, num_tokens in receivers:
            deadline = time.perf_counter() + 180.0
            while True:
                state = receiver.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    break
                if time.perf_counter() > deadline:
                    state = KVPoll.FAILED
                    break
                time.sleep(0.001)

            if state != KVPoll.SUCCESS:
                ctl.sendall(b"FAIL\n")
                failed_cells.append(req_id)
                print(f"[D] FAIL {req_id} state={state.value}", flush=True)
                break

            arr = receiver.result["kv"]
            assert arr is not None
            got_bytes = _arr_host_bytes(arr)
            ref_bytes = _payload_numpy(seed, dtype_name, num_tokens).tobytes()
            if got_bytes != ref_bytes:
                ctl.sendall(b"MISMATCH\n")
                failed_cells.append(req_id)
                path_label = "path A" if use_d2h else "path B"
                print(f"[D] MISMATCH {req_id} ({path_label})", flush=True)
                break
            ctl.sendall(b"OK\n")
            successes += 1
            cell_done += 1
        print(
            f"[D] cell {cell.dtype_name}/{cell.page_count}: {cell_done}/{ITERATIONS}",
            flush=True,
        )
        _print_cell_result(
            level="manager",
            path_name="path-a" if use_d2h else "path-b",
            dtype_name=cell.dtype_name,
            page_count=cell.page_count,
            num_iters=cell_done,
            itemsize=_dtype_itemsize(cell.dtype),
            elapsed_s=time.perf_counter() - cell_t0,
        )

    ctl.close()
    d_notifier.stop()
    failed = bool(failed_cells)
    print(
        f"[D] done: {successes}/{expected_total} iters, failed={failed_cells}",
        flush=True,
    )
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["prefill", "decode"])
    ap.add_argument("--my-host", default="")
    ap.add_argument("--remote", default="")
    ap.add_argument("--ctl-port", type=int, default=31000)
    ap.add_argument("--transfer-port", type=int, default=31001)
    ap.add_argument("--side-channel-port", type=int, default=9600)
    ap.add_argument("--use-d2h-staging", action="store_true")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    args = ap.parse_args()

    print(f"[init] role={args.role} jax={jax.__version__}", flush=True)
    print(f"[init] local_devices={jax.local_devices()}", flush=True)

    if args.role == "prefill":
        if not args.my_host:
            print("prefill needs --my-host", file=sys.stderr)
            return 2
        return _prefill(args)
    if not args.remote:
        print("decode needs --remote", file=sys.stderr)
        return 2
    return _decode(args)


if __name__ == "__main__":
    sys.exit(main())
