"""Real BatchScheduler (tiny synthetic model, CPU) behind the HTTP server: concurrent streamed requests must equal non-streamed greedy answers.
Usage: python test_stream_engine.py <ref_cpu_dir> <synth_ckpt> <tokenizer.json> <encoding_dir>"""

import json
import os
import queue
import sys
import threading
import time
import urllib.request

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_NOQUANT"] = "1"
R, CK, TJ, ENC = sys.argv[1:5]
sys.path.insert(0, R)
sys.path.insert(0, ".")
sys.path.insert(0, ENC)
import config_real as C
import generate as G
import http_api as H
import jax
import jax.numpy as jnp
import loader as L
import model as M
import model_ref as J
import numpy as np
import sharding as SH
from encoding import encode_messages, parse_message_from_completion_text
from tokenizers import Tokenizer

base = dict(
    dtype="bf16",
    expert_dtype=None,
    n_mtp_layers=0,
    compress_ratios=(0, 2, 2, 1, 1, 1),
    kv_source_layers=(1, 3),
    index_source_layers=(1, 3, 5),
    n_layers=6,
    max_seq_len=256,
    max_batch_size=2,
    window_size=32,
    candidate_source_layer=3,
    candidate_topk_blocks=4,
    candidate_block_size=4,
    index_topk=8,
)
a = J.Args.from_torch(M.ModelArgs(**base))
mesh = SH.make_mesh(jax.devices(), 2, 4)
params = L.build_params(
    L.SafetensorsDir(CK),
    C.param_table(a, act=jnp.float32),
    mesh,
    SH.param_spec,
    a.n_routed_experts,
    threads=4,
    log=lambda *x: None,
)
gen = G.Generator(params, a, mesh, batch=4, sizes=(32, 8, 2), eos_id=-5)
sched = G.BatchScheduler(gen)
tok = Tokenizer.from_file(TJ)
work = queue.Queue()
items = {}
sched.on_token = lambda rid, t: (
    items[rid]["stream"].put(t) if items[rid].get("stream") is not None else None
)


def engine():
    nid = 1
    while True:
        while True:
            try:
                it = work.get(timeout=0.01 if sched.busy() else 0.2)
                items[nid] = it
                sched.add(nid, it["ids"], it["max_new"], it["temp"], it["seed"])
                nid += 1
            except queue.Empty:
                break
        if sched.busy():
            for rid, out, st in sched.step():
                it = items.pop(rid)
                it["out"], it["stats"] = out, st
                it["done"].set()
                if it.get("stream") is not None:
                    it["stream"].put(None)


threading.Thread(target=engine, daemon=True).start()
deps = H.Deps(
    tok,
    encode_messages,
    parse_message_from_completion_text,
    -5,
    200,
    150,
    work,
    lambda: {"status": "ok"},
    lambda p, m, r: None,
)
srv = H.Server(("127.0.0.1", 0), H.make_handler(deps))
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def post(body, stream):
    r = urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        ),
        timeout=300,
    )
    if not stream:
        return json.load(r)["choices"][0]["text"]
    out = ""
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            out += json.loads(line[6:])["choices"][0]["text"]
    return out


prompts = [
    "hello world this is a test of the streaming path",
    "another prompt that is a little longer than the first one is",
    "short",
]
want = [post({"prompt": p, "max_tokens": 12, "temperature": 0}, False) for p in prompts]
got = [None] * 3


def run(i):
    got[i] = post({"prompt": prompts[i], "max_tokens": 12, "temperature": 0, "stream": True}, True)


ths = [threading.Thread(target=run, args=(i,)) for i in range(3)]
[t.start() for t in ths]
[t.join(timeout=300) for t in ths]
for i in range(3):
    print(f"req {i}: streamed == non-streamed: {got[i] == want[i]}  ({len(want[i])} chars)")
print("ALL PASS" if all(got[i] == want[i] for i in range(3)) else "SOME FAILED")
