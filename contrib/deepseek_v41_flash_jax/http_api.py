"""OpenAI-style HTTP API for the V4.1 server (engine-independent, so it can be tested on CPU with a fake engine).

Endpoints: GET /health, GET|POST /admin/* (delegated to deps.admin), POST /v1/completions, POST /v1/chat/completions (both with
optional "stream": true -> server-sent events). Requests are handed to the engine through `deps.work` (a queue of dicts); the
engine thread fills item["out"], item["stats"], sets item["done"], and (for streaming requests) pushes every sampled token id into
item["stream"] followed by a None sentinel.
"""

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EOS_TEXT = "<｜end▁of▁sentence｜>"
THINK_END = "</think>"


class TextStreamer:
    """Incremental detokenizer: turns a growing token list into (reasoning_delta, content_delta) text pieces without ever emitting a
    partial UTF-8 character or half of the </think> tag. Concatenating all deltas equals the non-streamed split of the full text.
    """

    def __init__(self, tok, thinking):
        self.tok, self.thinking = tok, thinking
        self.ids, self.sent_r, self.sent_c = [], 0, 0

    def _split(self, text, final):
        text = text.replace(EOS_TEXT, "")
        if not final:
            text = (
                text.rstrip("�") if text.endswith("�") else text
            )  # incomplete multi-byte sequence: wait for more tokens
        if not self.thinking:
            return "", text
        k = text.find(THINK_END)
        if k >= 0:
            return text[:k], text[k + len(THINK_END) :]
        reasoning = text
        if not final:  # a suffix that could still become "</think>"
            for j in range(min(len(THINK_END) - 1, len(text)), 0, -1):
                if THINK_END.startswith(text[-j:]):
                    reasoning = text[:-j]
                    break
        return reasoning, ""

    def push(self, new_ids, final=False):
        self.ids.extend(new_ids)
        r, c = self._split(self.tok.decode(self.ids, skip_special_tokens=False), final)
        dr, dc = r[self.sent_r :], c[self.sent_c :]
        self.sent_r += len(dr)
        self.sent_c += len(dc)
        return dr, dc


class Deps:
    """What the handler needs from the server: tokenizer, prompt encoder/parser, limits, work queue, health/admin callbacks."""

    def __init__(
        self,
        tok,
        encode_messages,
        parse_message,
        eos_id,
        max_seq,
        max_prompt,
        work,
        health,
        admin,
        model_name="DeepSeek-V4.1-Flash",
    ):
        self.tok, self.encode_messages, self.parse_message, self.eos_id = (
            tok,
            encode_messages,
            parse_message,
            eos_id,
        )
        self.max_seq, self.max_prompt, self.work, self.health, self.admin, self.model_name = (
            max_seq,
            max_prompt,
            work,
            health,
            admin,
            model_name,
        )


def make_handler(deps: Deps):
    tok = deps.tok

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            r = deps.admin(self.path, "GET", {}) if self.path.startswith("/admin") else None
            self._send(*(r if r else (200, deps.health())))

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/admin"):
                r = deps.admin(self.path, "POST", req)
                return self._send(*(r if r else (404, {"error": "unknown admin path"})))
            chat = self.path.endswith("/chat/completions")
            if chat:
                mode = "thinking" if req.get("thinking", True) else "chat"
                text = deps.encode_messages(
                    req["messages"],
                    thinking_mode=mode,
                    reasoning_effort=req.get("reasoning_effort"),
                )
            else:
                mode, text = None, req["prompt"]
            ids = tok.encode(text, add_special_tokens=False).ids
            max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)
            if len(ids) + max_new > deps.max_seq or len(ids) > deps.max_prompt:
                return self._send(
                    400,
                    {
                        "error": f"prompt too long ({len(ids)} tokens; MAX_PROMPT={deps.max_prompt}, MAX_SEQ={deps.max_seq})"
                    },
                )
            stream = bool(req.get("stream"))
            item = {
                "ids": ids,
                "max_new": max_new,
                "temp": float(req.get("temperature") or 0.0),
                "seed": int(req.get("seed") or 0),
                "top_p": float(req.get("top_p") or 1.0),
                "top_k": int(req.get("top_k") or 0),
                "done": threading.Event(),
                "out": None,
                "stream": queue.Queue() if stream else None,
            }
            deps.work.put(item)
            if stream:
                return self._stream(item, chat, mode)
            item["done"].wait()
            out, st = item["out"], item["stats"]
            raw = tok.decode(out, skip_special_tokens=False)
            finish = "stop" if out and out[-1] == deps.eos_id else "length"
            text_out = raw.replace(EOS_TEXT, "")
            parse_text = raw if finish == "stop" else raw + EOS_TEXT
            if chat:
                try:
                    msg = deps.parse_message(parse_text, thinking_mode=mode)
                    message = {
                        "role": "assistant",
                        "content": msg.get("content"),
                        "reasoning_content": msg.get("reasoning_content"),
                    }
                except Exception:
                    message = {"role": "assistant", "content": text_out}
                choice = {"index": 0, "message": message, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text_out, "finish_reason": finish}
            self._send(
                200,
                {
                    "object": "chat.completion" if chat else "text_completion",
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": st["prompt_tokens"],
                        "completion_tokens": st["new_tokens"],
                        "total_tokens": st["prompt_tokens"] + st["new_tokens"],
                    },
                    "timings": {"ttft_s": st["ttft_s"], "total_s": st["total_s"]},
                },
            )

        # ---- server-sent events
        def _event(self, obj):
            data = (
                "data: "
                + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))
                + "\n\n"
            ).encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _stream(self, item, chat, mode):
            rid, created = f"cmpl-{int(time.time() * 1000)}", int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(delta=None, text=None, finish=None, extra=None):
                if chat:
                    c = {"index": 0, "delta": delta or {}, "finish_reason": finish}
                    o = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": deps.model_name,
                        "choices": [c],
                    }
                else:
                    o = {
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": deps.model_name,
                        "choices": [{"index": 0, "text": text or "", "finish_reason": finish}],
                    }
                if extra:
                    o.update(extra)
                return o

            streamer = TextStreamer(tok, thinking=(mode == "thinking"))
            try:
                if chat:
                    self._event(chunk({"role": "assistant"}))
                finished = False
                while not finished:
                    ids = [item["stream"].get()]
                    while True:  # coalesce whatever else is already available
                        try:
                            ids.append(item["stream"].get_nowait())
                        except queue.Empty:
                            break
                    if ids[-1] is None:
                        finished = True
                        ids = ids[:-1]
                    dr, dc = streamer.push(ids, final=finished)
                    if dr or dc:
                        if chat:
                            self._event(
                                chunk(
                                    {
                                        k: v
                                        for k, v in (("reasoning_content", dr), ("content", dc))
                                        if v
                                    }
                                )
                            )
                        else:
                            self._event(chunk(text=dc))
                item["done"].wait()
                st = item["stats"]
                finish = "stop" if streamer.ids and streamer.ids[-1] == deps.eos_id else "length"
                self._event(
                    chunk(
                        finish=finish,
                        extra={
                            "usage": {
                                "prompt_tokens": st["prompt_tokens"],
                                "completion_tokens": st["new_tokens"],
                                "total_tokens": st["prompt_tokens"] + st["new_tokens"],
                            },
                            "timings": {"ttft_s": st["ttft_s"], "total_s": st["total_s"]},
                        },
                    )
                )
                self._event("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                item["cancelled"] = (
                    True  # the client went away; generation still runs to completion (no cancel broadcast yet)
                )

        def log_message(self, *a):
            pass

    return Handler


class Server(ThreadingHTTPServer):
    request_queue_size = 2048  # default 5 overflows the accept backlog under hundreds of simultaneous clients (connection resets)
    daemon_threads = True
