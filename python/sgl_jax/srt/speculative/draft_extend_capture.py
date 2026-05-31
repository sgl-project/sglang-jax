"""Capture draft_extend_for_decode entry state for offline testing.

Enable: SGLANG_CAPTURE_DRAFT_EXTEND=1
Dir:    SGLANG_CAPTURE_DIR (default /models/capature_niu)
Max:    SGLANG_CAPTURE_MAX (default 3)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import jax
import numpy as np

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_CAPTURE_DRAFT_EXTEND") == "1"
CAPTURE_DIR = os.environ.get("SGLANG_CAPTURE_DIR", "/models/capature_niu")
MAX_CAPTURES = int(os.environ.get("SGLANG_CAPTURE_MAX", "3"))
_call_count = 0


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (int, float)):
        return np.array(x)
    return np.asarray(jax.device_get(x))


def _sharding_spec_str(x) -> str:
    if not isinstance(x, jax.Array):
        return ""
    sh = x.sharding
    if hasattr(sh, "spec"):
        return str(sh.spec)
    return str(sh)


def maybe_capture_decode_entry(mwb, batch_output, accept_host, sel_pos, workers):
    """Call after prepare_for_extend_after_verify, before the N-layer loop."""
    global _call_count
    if not ENABLED:
        return
    if _call_count >= MAX_CAPTURES:
        return
    call_id = _call_count
    _call_count += 1

    out_dir = Path(CAPTURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"draft_extend_{call_id}.npz"
    logger.info("[CAPTURE] Saving draft_extend_for_decode entry #%d to %s", call_id, path)

    data = {}

    # mwb array fields (already numpy after device_get in draft_extend_for_decode)
    for name in [
        "input_ids",
        "positions",
        "seq_lens",
        "extend_seq_lens",
        "req_pool_indices",
        "cache_loc",
        "out_cache_loc",
        "logits_indices_selector",
        "logits_indices",
    ]:
        v = getattr(mwb, name, None)
        if v is not None:
            data[f"mwb_{name}"] = _to_np(v)

    # scalars
    data["mwb_dp_size"] = np.array(mwb.dp_size, dtype=np.int32)
    data["mwb_per_dp_bs_size"] = np.array(mwb.per_dp_bs_size, dtype=np.int32)
    data["mwb_real_bs"] = np.array(mwb.real_bs, dtype=np.int32)
    data["mwb_bid"] = np.array(getattr(mwb, "bid", 0), dtype=np.int32)
    data["mwb_forward_mode"] = np.array(int(mwb.forward_mode), dtype=np.int32)

    # spec_info fields
    si = mwb.spec_info_padded
    if si is not None:
        for name in ["hidden_states", "verified_id", "accept_length", "allocate_lens"]:
            v = getattr(si, name, None)
            if v is not None:
                data[f"si_{name}"] = _to_np(v)
                data[f"si_{name}_sharding"] = np.array(
                    _sharding_spec_str(getattr(si, name, None)), dtype=object
                )

    # accept_host, sel_pos (already numpy)
    data["accept_host"] = _to_np(accept_host)
    data["sel_pos"] = _to_np(sel_pos)

    # batch_output fields for final select_index
    ndi = batch_output.next_draft_input
    if ndi is not None:
        data["bo_verified_id"] = _to_np(ndi.verified_id)
        data["bo_verified_id_sharding"] = np.array(
            _sharding_spec_str(ndi.verified_id), dtype=object
        )
    data["bo_allocate_lens"] = _to_np(batch_output.allocate_lens)

    _savez_via_tmp(path, data)
    logger.info(
        "[CAPTURE] Saved %d arrays to %s (%.1f MB)", len(data), path, path.stat().st_size / 1e6
    )


def _savez_via_tmp(path: Path, data: dict):
    """Save .npz via /tmp to avoid GCS Fuse stale handle errors."""
    import shutil
    import tempfile

    tmp_path = Path(tempfile.mktemp(suffix=".npz", dir="/tmp"))
    np.savez(str(tmp_path), **data)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), str(path))


def maybe_capture_kv_after(call_id, workers, cache_loc_np, page_size):
    """Call after the N-layer loop to snapshot KV state after forward."""
    if not ENABLED:
        return
    if call_id >= MAX_CAPTURES:
        return

    out_dir = Path(CAPTURE_DIR)
    path = out_dir / f"draft_extend_{call_id}_kv_after.npz"

    page_indices_device = cache_loc_np[::page_size] // page_size
    data = {}
    for i, w in enumerate(workers):
        mr = w.model_runner
        kv_buf = mr.memory_pools.token_to_kv_pool.kv_buffer[0]
        data[f"kv_layer{i}_after"] = _to_np(kv_buf[page_indices_device])

    _savez_via_tmp(path, data)
    logger.info("[CAPTURE] Saved KV-after to %s", path)


def maybe_capture_golden_output(call_id, batch_output, allocate_lens, accept_host):
    """Call after final outputs are assembled to save golden reference."""
    if not ENABLED:
        return
    if call_id >= MAX_CAPTURES:
        return

    out_dir = Path(CAPTURE_DIR)
    path = out_dir / f"draft_extend_{call_id}_golden.npz"

    ndi = batch_output.next_draft_input
    data = {
        "golden_hidden_states": _to_np(ndi.hidden_states),
        "golden_topk_p": _to_np(ndi.topk_p),
        "golden_topk_index": _to_np(ndi.topk_index),
        "golden_verified_id": _to_np(ndi.verified_id),
        "golden_allocate_lens": _to_np(allocate_lens),
        "golden_accept_lens": _to_np(accept_host),
    }
    _savez_via_tmp(path, data)
    logger.info("[CAPTURE] Saved golden outputs to %s", path)


def load_capture(path: str) -> dict:
    """Load a captured .npz and return as a dict of numpy arrays."""
    return dict(np.load(path, allow_pickle=True))
