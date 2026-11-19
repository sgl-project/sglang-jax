"""HTTP API with a fake engine: streamed deltas must reproduce the non-streamed answer exactly (reasoning/content split, multi-byte
characters, </think> split across tokens). Usage: python test_http_api.py <tokenizer.json> <encoding_dir>"""

import json
import queue
import sys
import threading
import time
import urllib.request

sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[2])
import http_api as H
from encoding import encode_messages, parse_message_from_completion_text
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
EOS = 1
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
work = queue.Queue()
script = queue.Queue()


def engine():
    while True:
        item = work.get()
        out = script.get()
        for t in out:
            if item["stream"] is not None:
                item["stream"].put(t)
                time.sleep(0.002)
        item["out"] = out
        item["stats"] = {
            "prompt_tokens": len(item["ids"]),
            "new_tokens": len(out),
            "ttft_s": 0.01,
            "total_s": 0.05,
        }
        item["done"].set()
        if item["stream"] is not None:
            item["stream"].put(None)


threading.Thread(target=engine, daemon=True).start()
deps = H.Deps(
    tok,
    encode_messages,
    parse_message_from_completion_text,
    EOS,
    4096,
    3000,
    work,
    lambda: {"status": "ok"},
    lambda path, m, r: None,
)
srv = H.Server(("127.0.0.1", 0), H.make_handler(deps))
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def post(path, body, stream=False):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=30)
    if not stream:
        return json.load(r)
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: "):
            events.append(line[6:])
    return events


ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


think_ids = (
    enc("Let me think 思考中 🙂 step one.")
    + enc("</")
    + enc("think")
    + enc(">")
    + enc("Answer: 391 😀 终")
    + [EOS]
)
msgs = [{"role": "user", "content": "hi"}]
for label, thinking in (("thinking chat", True), ("plain chat", False)):
    out_ids = think_ids if thinking else enc("Answer: 391 😀 终 and 😀😀") + [EOS]
    script.put(out_ids)
    full = post("/v1/chat/completions", {"messages": msgs, "thinking": thinking, "max_tokens": 200})
    script.put(out_ids)
    ev = post(
        "/v1/chat/completions",
        {"messages": msgs, "thinking": thinking, "max_tokens": 200, "stream": True},
        stream=True,
    )
    chunks = [json.loads(e) for e in ev if e != "[DONE]"]
    r = "".join(
        c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"]
    )
    c_ = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    m = full["choices"][0]["message"]
    print(
        label,
        "| events",
        len(ev),
        "| finish",
        chunks[-1]["choices"][0]["finish_reason"],
        "| usage",
        chunks[-1].get("usage"),
    )
    check(f"{label}: content equals non-streamed", c_ == (m["content"] or ""))
    check(f"{label}: reasoning equals non-streamed", r == (m["reasoning_content"] or ""))
    check(f"{label}: no replacement characters", "�" not in r + c_)
    check(
        f"{label}: [DONE] last and finish_reason stop",
        ev[-1] == "[DONE]" and chunks[-1]["choices"][0]["finish_reason"] == "stop",
    )
    check(
        f"{label}: first chunk carries the assistant role",
        chunks[0]["choices"][0]["delta"].get("role") == "assistant",
    )
# raw completions stream, length-truncated (no EOS)
txt_ids = enc("alpha beta 你好 😀 gamma")
script.put(txt_ids)
full = post("/v1/completions", {"prompt": "x", "max_tokens": 50})
script.put(txt_ids)
ev = post("/v1/completions", {"prompt": "x", "max_tokens": 50, "stream": True}, stream=True)
chunks = [json.loads(e) for e in ev if e != "[DONE]"]
s = "".join(c["choices"][0]["text"] for c in chunks)
check("completions: streamed text equals non-streamed", s == full["choices"][0]["text"])
check(
    "completions: finish_reason length when no EOS",
    chunks[-1]["choices"][0]["finish_reason"] == "length",
)
# a tag-like prefix that is NOT </think> must still come out
script.put(enc("a </th not a tag and 5 < 6.") + enc("</think>") + enc("done") + [EOS])
ev = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 50, "stream": True}, stream=True)
chunks = [json.loads(e) for e in ev if e != "[DONE]"]
r = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"])
check("near-miss </th kept in reasoning", r == "a </th not a tag and 5 < 6.")
print("ALL PASS" if ok_all else "SOME FAILED")
