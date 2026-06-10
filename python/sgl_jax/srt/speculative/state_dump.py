from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

try:
    import jax
except Exception:  # pragma: no cover
    jax = None

_lock = threading.Lock()
_counter = 0


def _enabled():
    return bool(os.environ.get("SGL_JAX_SPEC_STATE_DUMP_DIR"))


def _rank():
    for name in ("RANK", "JAX_PROCESS_INDEX", "NODE_RANK"):
        val = os.environ.get(name)
        if val is not None:
            return val
    if jax is not None:
        with suppress(Exception):
            return str(jax.process_index())
    return "unknown"


def _path():
    root = Path(os.environ["SGL_JAX_SPEC_STATE_DUMP_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    return root / f"spec_state_rank{_rank()}_{socket.gethostname()}.jsonl"


def _as_array(value):
    if value is None:
        return None, None
    try:
        arr = np.asarray(value)
        return arr, None
    except Exception as exc:
        if hasattr(value, "addressable_shards"):
            shards = []
            for shard in value.addressable_shards[:2]:
                with suppress(Exception):
                    shards.append(np.asarray(shard.data))
            if shards:
                return np.concatenate([s.reshape(-1)[:16] for s in shards]), str(exc)
        return None, str(exc)


def summarize(value, limit=16):
    if value is None:
        return None
    arr, err = _as_array(value)
    meta = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        with suppress(Exception):
            meta["shape"] = list(value.shape)
    if hasattr(value, "dtype"):
        with suppress(Exception):
            meta["dtype"] = str(value.dtype)
    if err is not None:
        meta["array_error"] = err[:240]
    if arr is None:
        return meta
    flat = arr.reshape(-1) if arr.shape else arr.reshape(1)
    meta["np_shape"] = list(arr.shape)
    meta["np_dtype"] = str(arr.dtype)
    vals = flat[:limit]
    if np.issubdtype(vals.dtype, np.integer):
        meta["head"] = [int(x) for x in vals]
    elif np.issubdtype(vals.dtype, np.floating):
        meta["head"] = [float(x) for x in vals]
    elif np.issubdtype(vals.dtype, np.bool_):
        meta["head"] = [bool(x) for x in vals]
    else:
        meta["head"] = [str(x) for x in vals]
    if flat.size and (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        try:
            meta["sum"] = float(np.asarray(flat, dtype=np.float64).sum())
            meta["min"] = float(np.asarray(flat).min())
            meta["max"] = float(np.asarray(flat).max())
        except Exception:
            pass
    return meta


def safe_getattr(obj, field):
    try:
        return getattr(obj, field)
    except Exception as exc:
        return {"getattr_error": f"{type(exc).__name__}: {str(exc)[:220]}"}


def req_summary(reqs):
    out = []
    for req in reqs or []:
        out.append(
            {
                "rid": getattr(req, "rid", None),
                "id": id(req),
                "finished": bool(req.finished()) if hasattr(req, "finished") else None,
                "output_len": len(getattr(req, "output_ids", []) or []),
                "last_output": list((getattr(req, "output_ids", []) or [])[-8:]),
                "kv_allocated_len": getattr(req, "kv_allocated_len", None),
                "kv_committed_len": getattr(req, "kv_committed_len", None),
            }
        )
    return out


def spec_summary(spec):
    if spec is None:
        return None
    fields = (
        "verified_id",
        "topk_index",
        "topk_p",
        "hidden_states",
        "new_seq_lens",
        "allocate_lens",
        "accept_length",
        "accept_length_cpu",
        "future_indices",
        "draft_token",
        "retrive_index",
        "next_verified_id",
        "sel_pos",
        "positions",
    )
    data = {"type": type(spec).__name__}
    for field in fields:
        value = safe_getattr(spec, field)
        if not isinstance(value, dict) or "getattr_error" not in value:
            data[field] = summarize(value)
    p = safe_getattr(spec, "pending_draft_extend_result")
    if not isinstance(p, dict) or "getattr_error" not in p:
        data["pending_type"] = None if p is None else type(p).__name__
    return data


def batch_summary(batch):
    if batch is None:
        return None
    data = {
        "type": type(batch).__name__,
        "bid": getattr(batch, "bid", None),
        "dp_size": getattr(batch, "dp_size", None),
        "per_dp_bs_size": getattr(batch, "per_dp_bs_size", None),
        "real_bs": getattr(batch, "real_bs", None),
        "real_bs_per_dp": getattr(batch, "real_bs_per_dp", None),
    }
    for field in (
        "input_ids",
        "seq_lens",
        "req_pool_indices",
        "logits_indices_selector",
        "positions",
        "out_cache_loc",
        "extend_seq_lens",
        "logits_indices",
    ):
        value = safe_getattr(batch, field)
        if isinstance(value, dict) and "getattr_error" in value:
            data[field] = value
        else:
            data[field] = summarize(value)
    spec_info_padded = safe_getattr(batch, "spec_info_padded")
    if not isinstance(spec_info_padded, dict) or "getattr_error" not in spec_info_padded:
        data["spec_info_padded"] = spec_summary(spec_info_padded)
    reqs_info = safe_getattr(batch, "reqs_info")
    if not isinstance(reqs_info, dict) or "getattr_error" not in reqs_info:
        infos = []
        for dp_rank, info in enumerate(reqs_info):
            infos.append(
                {
                    "dp_rank": dp_rank,
                    "reqs": req_summary(getattr(info, "reqs", None)),
                    "req_pool_indices": summarize(getattr(info, "req_pool_indices", None)),
                    "seq_lens": summarize(getattr(info, "seq_lens", None)),
                    "spec_info": spec_summary(getattr(info, "spec_info", None)),
                }
            )
        data["reqs_info"] = infos
    return data


def result_summary(result):
    if result is None:
        return None
    data = {"type": type(result).__name__, "bid": getattr(result, "bid", None)}
    for field in ("next_token_ids", "accept_lens", "allocate_lens"):
        if hasattr(result, field):
            data[field] = summarize(getattr(result, field))
    if hasattr(result, "next_draft_input"):
        data["next_draft_input"] = spec_summary(result.next_draft_input)
    return data


def dump_state(event, **kwargs):
    if not _enabled():
        return
    global _counter
    with _lock:
        _counter += 1
        rec = {
            "event_id": _counter,
            "event": event,
            "time": time.time(),
            "rank": _rank(),
            "host": socket.gethostname(),
        }
        for key, value in kwargs.items():
            if key == "batch":
                rec[key] = batch_summary(value)
            elif key == "result":
                rec[key] = result_summary(value)
            elif key == "spec":
                rec[key] = spec_summary(value)
            elif key == "reqs":
                rec[key] = req_summary(value)
            else:
                rec[key] = (
                    summarize(value)
                    if not isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
                    else value
                )
        with _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
