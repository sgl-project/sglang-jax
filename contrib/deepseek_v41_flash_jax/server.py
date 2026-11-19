"""DeepSeek-V4.1-Flash server on a TPU slice (our JAX engine). Run the SAME command on every host:

    RANK=<worker id> NPROC=8 HEAD_IP=<host0 ip> python server.py <checkpoint dir> [port]

Host 0 serves HTTP (/health, /v1/chat/completions, /v1/completions); all hosts execute each request in lockstep: host 0 broadcasts
the request tokens (heartbeat every 0.5 s when idle), every host runs the identical generate() and gathers Engram rows from its own
memory-mapped copy of the tables.
"""

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_real as C
import engram_jax as E
import generate as G
import jax
import jax.numpy as jnp
import loader as L
import sharding as SH
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh

CKPT = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
rank, nproc = int(os.environ.get("RANK", "0")), int(os.environ.get("NPROC", "8"))
head = os.environ.get("HEAD_IP", "127.0.0.1")
DP, TP = int(os.environ.get("DP", "4")), int(os.environ.get("TP", "8"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "16384"))
MAXP = int(os.environ.get("MAX_PROMPT", str(int(os.environ.get("MAX_SEQ", "16384")) - 512)))
INLINE = 1024  # prompt tokens carried in the small heartbeat payload; longer prompts use a second broadcast
# SGLang-style knobs: MAX_RUNNING_REQUESTS rows are allocated at startup (multiple of DP; ROWS is an alias); decode steps are padded
# to the smallest of BS_PADDINGS (total batch sizes, default powers of two) >= live requests; MAX_BATCH is a runtime cap
# (POST /admin/config {"max_batch": N}).
ROWS = int(os.environ.get("MAX_RUNNING_REQUESTS", os.environ.get("ROWS", "16")))
ROWS = max(DP, ROWS // DP * DP)
BS_PADDINGS = (
    [int(x) for x in os.environ["BS_PADDINGS"].split(",")]
    if os.environ.get("BS_PADDINGS")
    else None
)
log = lambda *m: print(f"[{rank}] {time.strftime('%H:%M:%S')}", *m, flush=True)

if nproc > 1:
    jax.distributed.initialize(f"{head}:{os.environ.get('PORT_COORD', '5000')}", nproc, rank)
log("devices", jax.device_count())

cfg_path = os.path.join(CKPT, "config.json")
a = C.args_from_hf_config(cfg_path, max_seq_len=MAX_SEQ)
tcfg = json.load(open(cfg_path)).get("text_config", json.load(open(cfg_path)))
mesh = Mesh(mesh_utils.create_device_mesh((DP, TP)), ("dp", "tp"))
ck = L.SafetensorsDir(CKPT)

t0 = time.time()
log("loading weights ...")
params = L.build_params(ck, C.param_table(a), mesh, SH.param_spec, a.n_routed_experts, log=log)
jax.block_until_ready(params)
log(f"weights on device in {time.time()-t0:.0f}s")

# ---- Engram: host hasher + memory-mapped tables
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))


class _Tok:
    backend_tokenizer = tok

    def __len__(self):
        return tok.get_vocab_size(with_added_tokens=True)


tmap, vsize = E.build_compressed_token_map(_Tok())
assert vsize == tcfg["engram_compressed_vocab_size"], (vsize, tcfg["engram_compressed_vocab_size"])
layout = E.EngramLayout.make(
    a.engram_layer_ids,
    tcfg["engram_num_embeddings"],
    a.engram_max_ngram_size,
    a.engram_n_heads,
    a.engram_head_dim,
    tcfg["engram_vocab_size"],
)
engram = {
    "hasher": E.NgramHasher(layout, tmap, vsize, tcfg.get("engram_pad_token_id", 2), ROWS, MAX_SEQ),
    "tables": {l: L.EngramTable(ck, l) for l in a.engram_layer_ids},
}
log("engram tables mapped:", {l: engram["tables"][l].w.shape for l in a.engram_layer_ids})

if (
    os.environ.get("V41_MOE") == "kernel"
):  # SGL-JAX fused expert-parallel MoE kernel: convert expert layout once on device
    import model_ref as J

    params = SH.to_kernel_layout(params, mesh, a.n_layers)
    J.MOE_MESH = mesh
    log("experts converted to fused-MoE kernel layout")
elif os.environ.get("SPARSE_MOE", "1") == "1":
    import model_ref as J

    J.MOE_MESH = mesh  # decode-sized steps loop over only the routed local experts
gen = G.Generator(
    params,
    a,
    mesh,
    batch=ROWS,
    engram=engram,
    sizes=tuple(int(x) for x in os.environ.get("CHUNKS", "512,256,128,64,32,16,8,4,2").split(",")),
    log=log,
)
LADDER = (
    [int(x) for x in os.environ["CTX_LADDER"].split(",")] if os.environ.get("CTX_LADDER") else None
)
sched = G.BatchScheduler(gen, bs_paddings=BS_PADDINGS, ladder=LADDER)
log(
    f"max_running_requests={ROWS} decode bs buckets={[k * DP for k in sched.buckets]} context ladder={sched.ladder}"
)
sched.set_max_live(int(os.environ.get("MAX_BATCH", str(ROWS))))
log("compiling chunk shapes ...")
sched.warmup()
if os.environ.get("V41_WARM_FLOW", "1") == "1":
    sched.warmup_flow()
_ms = jax.local_devices()[0].memory_stats() or {}
log(
    "HBM after warm-up (device 0): in use %.2f GB, peak %.2f GB, limit %.2f GB"
    % (
        _ms.get("bytes_in_use", 0) / 1e9,
        _ms.get("peak_bytes_in_use", 0) / 1e9,
        _ms.get("bytes_limit", 0) / 1e9,
    )
)
log("READY")
BCAST_EVERY = int(
    os.environ.get("BCAST_EVERY", "4")
)  # while busy, poll for new requests every N scheduling rounds

# ---- V4 prompt encoding (DeepSeek's own encoder shipped with the checkpoint)
sys.path.insert(0, os.path.join(CKPT, "encoding"))
try:
    from encoding import (  # noqa: E402  (V4.1)
        encode_messages,
        parse_message_from_completion_text,
    )
except ModuleNotFoundError:
    from encoding_dsv4 import (  # noqa: E402  (V4)
        encode_messages,
        parse_message_from_completion_text,
    )

work = queue.Queue()

# Broadcast from the host whose RANK==0 (jax.process_index() is NOT guaranteed to equal RANK on TPU, so multihost_utils.broadcast_one_to_all,
# which sends from process 0, can pick the wrong host). One row per device; only the RANK-0 host contributes non-zero data; sum over devices.
_bm = Mesh(np.array(jax.devices()), ("d",))
_bin = jax.sharding.NamedSharding(_bm, jax.sharding.PartitionSpec("d", None))
_bsum = jax.jit(
    lambda x: jnp.sum(x, axis=0),
    out_shardings=jax.sharding.NamedSharding(_bm, jax.sharding.PartitionSpec()),
)


def bcast(payload):
    local = np.zeros((jax.local_device_count(), payload.shape[0]), np.int32)
    if rank == 0:
        local[0] = payload
    # one shard per local device (data only on the first local device of the RANK-0 host); avoids make_array_from_process_local_data's
    # cross-host shape consistency check (an allgather) on every call
    parts = [jax.device_put(local[j : j + 1], d) for j, d in enumerate(jax.local_devices())]
    g = jax.make_array_from_single_device_arrays(
        (jax.device_count(), payload.shape[0]), _bin, parts
    )
    return np.asarray(_bsum(g).addressable_data(0))


import http_api as H


def _admin(path, method, req):
    """/admin/config (GET: state; POST {"max_batch": N}: runtime concurrency cap, applied on every host at the next round) and
    /admin/profile (POST {"rounds": N}: trace the next N busy scheduling rounds on every host -> /tmp/prof_live).
    """
    if path.startswith("/admin/profile") and method == "POST":
        ev = threading.Event()
        work.put({"profile": int(req.get("rounds", 20)), "done": ev})
        ev.wait()
        return 200, {
            "profiling_rounds": int(req.get("rounds", 20)),
            "trace_dir": "/tmp/prof_live (each host)",
        }
    if path.startswith("/admin/config"):
        if method == "GET":
            return 200, {
                "max_batch": sched.max_live,
                "max_running_requests": sched.B,
                "bs_buckets": [k * DP for k in sched.buckets],
                "running": sched.live_count(),
                "queued": len(sched.pending),
                "pf_min_rows": sched.pf_min_rows,
                "pf_max_wait": sched.pf_max_wait,
            }
        if "pf_min_rows" in req or "pf_max_wait" in req:
            ev = threading.Event()
            work.put(
                {
                    "admin_pf": (
                        int(req.get("pf_min_rows", sched.pf_min_rows)),
                        int(req.get("pf_max_wait", sched.pf_max_wait)),
                    ),
                    "done": ev,
                }
            )
            ev.wait()
            return 200, {"pf_min_rows": sched.pf_min_rows, "pf_max_wait": sched.pf_max_wait}
        n = int(req["max_batch"])
        if not 1 <= n <= sched.B:
            return 400, {
                "error": f"max_batch must be in 1..{sched.B} (rows allocated at startup; set MAX_RUNNING_REQUESTS to allocate more)"
            }
        ev = threading.Event()
        work.put({"admin": n, "done": ev})
        ev.wait()
        return 200, {"max_batch": sched.max_live, "max_running_requests": sched.B}
    return None


def _health():
    return {
        "status": "ok",
        "model": "DeepSeek-V4.1-Flash",
        "devices": jax.device_count(),
        "max_batch": sched.max_live,
        "max_running_requests": sched.B,
        "running": sched.live_count(),
        "queued": len(sched.pending),
    }


deps = H.Deps(
    tok,
    encode_messages,
    parse_message_from_completion_text,
    gen.eos,
    MAX_SEQ,
    MAXP,
    work,
    _health,
    _admin,
)
items_by_rid = (
    {}
)  # rid -> request dict (rank 0); filled by the lockstep loop, read by the per-token streaming hook
if rank == 0:
    sched.on_token = lambda rid, t: (
        items_by_rid[rid]["stream"].put(t)
        if rid in items_by_rid and items_by_rid[rid].get("stream") is not None
        else None
    )

if rank == 0:
    srv = H.Server(("0.0.0.0", PORT), H.make_handler(deps))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"HTTP on :{PORT}")

HDR = 8


class _Admin(Exception):
    pass


items, next_rid, rnd, got, prof_left = items_by_rid, 1, 0, False, 0
while (
    True
):  # lockstep loop on every host; the scheduler is deterministic, only new arrivals are broadcast
    busy = sched.busy()
    if (
        not busy or got or rnd % BCAST_EVERY == 0
    ):  # after an arrival poll again next round (drains bursts quickly)
        payload = np.zeros(HDR + INLINE, np.int32)
        big_ids = None
        if rank == 0:
            try:
                item = work.get(timeout=0.5) if not busy else work.get_nowait()
                if "admin" in item:
                    payload[:2] = [2, item["admin"]]
                    raise _Admin(item)
                if "profile" in item:
                    payload[:2] = [3, item["profile"]]
                    raise _Admin(item)
                if "admin_pf" in item:
                    payload[:3] = [4, *item["admin_pf"]]
                    raise _Admin(item)
                n = len(item["ids"])
                items[next_rid] = item
                payload[:HDR] = [
                    1,
                    n,
                    item["max_new"],
                    int(item["temp"] * 1000),
                    item["seed"],
                    next_rid,
                    int(round(item["top_p"] * 1000)),
                    item["top_k"],
                ]
                payload[HDR : HDR + min(n, INLINE)] = item["ids"][:INLINE]
                big_ids = item["ids"][INLINE:]
                next_rid += 1
            except queue.Empty:
                pass
            except _Admin as ad:
                admin_item = ad.args[0]
        payload = bcast(payload)
        if payload[0] == 2:
            sched.set_max_live(int(payload[1]))
            log(f"max_batch set to {sched.max_live}")
            if rank == 0:
                admin_item["done"].set()
        if payload[0] == 4:
            sched.set_pf_policy(int(payload[1]), int(payload[2]))
            log(f"prefill policy: min_rows={sched.pf_min_rows} max_wait_rounds={sched.pf_max_wait}")
            if rank == 0:
                admin_item["done"].set()
        if payload[0] == 3:
            import shutil

            shutil.rmtree("/tmp/prof_live", ignore_errors=True)
            jax.profiler.start_trace("/tmp/prof_live")
            prof_left = int(payload[1])
            log(f"profiling the next {prof_left} busy rounds -> /tmp/prof_live")
            if rank == 0:
                admin_item["done"].set()
        got = payload[0] != 0
        if payload[0] == 1:
            n = int(payload[1])
            ids_all = payload[HDR : HDR + min(n, INLINE)].tolist()
            if (
                n > INLINE
            ):  # long prompt: second broadcast, padded to a multiple of 8192 tokens (few distinct shapes)
                rest = np.zeros(-(-(n - INLINE) // 8192) * 8192, np.int32)
                if rank == 0:
                    rest[: n - INLINE] = big_ids
                ids_all += bcast(rest)[: n - INLINE].tolist()
            sched.add(
                int(payload[5]),
                ids_all,
                int(payload[2]),
                payload[3] / 1000.0,
                int(payload[4]),
                payload[6] / 1000.0,
                int(payload[7]),
            )
    rnd += 1
    if not sched.busy():
        continue
    fin_now = sched.step()
    if prof_left > 0:
        prof_left -= 1
        if prof_left == 0:
            jax.profiler.stop_trace()
            log("profile saved to /tmp/prof_live")
    for rid, out, st in fin_now:
        if rank == 0:
            it = items.pop(rid)
            it["out"], it["stats"] = out, st
            log(f"request {rid}: {st}")
            it["done"].set()
            if it.get("stream") is not None:
                it["stream"].put(None)  # end-of-stream sentinel for the SSE handler
