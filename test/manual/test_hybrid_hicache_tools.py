"""CPU validation of manual acceptance entrypoints; no model/server is started."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import hybrid_hicache_loaded_probe as loaded
import hybrid_hicache_scenarios as scenarios
import pytest


def config():
    return dict(
        model_path="google/gemma-4-31B-it",
        revision="a" * 40,
        device="tpu",
        tp_size=8,
        dp_size=1,
        ep_size=1,
        page_size=128,
        enable_unified_radix_tree=True,
        hicache_storage="none",
        hicache_transfer_backend="jax",
        hicache_write_policy="write_through",
    )


def test_loaded_driver_check_config_does_not_import_runtime(tmp_path):
    path = tmp_path / "args.json"
    path.write_text(json.dumps(config()))
    script = "import sys; import hybrid_hicache_loaded_probe as p; p.main(sys.argv[1:]); assert 'jax' not in sys.modules"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--server-args",
            str(path),
            "--output",
            str(tmp_path / "probe.json"),
            "--check-config",
        ],
        cwd=Path(__file__).parent,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "probe.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"device": "cpu"},
        {"dp_size": 2},
        {"nnodes": 2},
        {"hicache_storage": "disable"},
    ],
)
def test_loaded_driver_rejects_unfrozen_or_wrong_stack(changes):
    with pytest.raises(ValueError):
        loaded.validate_config(config() | changes)


def test_loaded_driver_requires_optin_before_runtime(monkeypatch, tmp_path):
    path = tmp_path / "args.json"
    path.write_text(json.dumps(config()))
    monkeypatch.setattr(loaded, "run_loaded_probe", lambda *args: pytest.fail("runtime reached"))
    with pytest.raises(SystemExit):
        loaded.main(["--server-args", str(path), "--output", str(tmp_path / "probe.json")])


@pytest.mark.parametrize("sample_pages", [0, 2])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_observer_records_real_eviction_restore_and_physical_samples(
    tmp_path, sample_pages, dp_size
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "srt"))
    import hybrid_hicache_observer as observer
    from hybrid_hicache_transfer_probe import _write_marker

    from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams
    from sgl_jax.test.mem_cache import conftest  # noqa: F401
    from sgl_jax.test.mem_cache.test_hybrid_hicache import (
        insert,
        make_cache,
        settle,
        shutdown,
    )

    rank = dp_size - 1
    cache, alloc, pool = make_cache(window=8, dp_size=dp_size)
    undo = observer.install(tmp_path, sample_pages=sample_pages)
    try:
        full, node = insert(cache, alloc, range(8), rank=rank)
        swa = alloc.translate_full_to_swa(full, dp_rank=rank)
        settle(cache)
        state = observer.snapshot(cache, physical=True)
        assert len(state["ranks"][rank]["full_to_swa_nonzero"]) == 8
        assert state["pool_storage"]["FULL"][0]["bytes_per_page"] > 0
        cache.evict(EvictParams(num_tokens=8, dp_rank=rank))
        for subpool, indices in ((pool.full_kv_pool, full), (pool.swa_kv_pool, swa)):
            _write_marker(subpool, indices, rank, -1)
        restored, _, _ = cache.init_load_back(node, 8)
        assert len(restored) == 8
        rows = [
            json.loads(line)
            for path in tmp_path.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert any(
            row["phase"] == "after" and row.get("restored_prefix_tokens") == 8 for row in rows
        )
        changed = [r for r in rows if r["phase"] == "old_slot_content_before_restore"]
        verified = [r for r in rows if r["phase"] == "restored_slot_content"]
        if sample_pages:
            assert {r["component"] for r in changed if r["changed"]} == {"FULL", "SWA"}
            assert {r["component"] for r in verified if r["matches"]} == {"FULL", "SWA"}
            assert all(row["dp_rank"] == rank for row in changed + verified)
        else:
            assert not changed and not verified
        assert all(r["source"] and r["sequence"] for r in rows)
    finally:
        undo()
        shutdown(cache)


@pytest.mark.parametrize("pending_rank", [None, 1])
def test_observer_preserves_pending_inputs_and_allocated_request_resources(
    monkeypatch, tmp_path, pending_rank
):
    import hybrid_hicache_observer as observer

    from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
    from sgl_jax.srt.sampling.sampling_params import SamplingParams
    from sgl_jax.test.mem_cache import conftest  # noqa: F401

    running = Req("running", "", [1, 2, 3], SamplingParams(max_new_tokens=1), dp_rank=0)
    waiting = Req("waiting", "", [4], SamplingParams(max_new_tokens=1), dp_rank=1)
    pending = TokenizedGenerateReqInput(rid="pending", dp_rank=pending_rank, input_ids=[5, 6])
    pool = ReqToTokenPool(size=2, max_context_len=8)
    pool.alloc([running])
    running.kv_allocated_len = 3
    running.cache_protected_len = 2
    running.swa_evicted_seqlen = 1
    pool.write((running.req_pool_idx, slice(0, 3)), [11, 12, 13])
    scheduler = SimpleNamespace(
        tree_cache=None,
        cur_batch=None,
        last_batch=None,
        running_batch=SimpleNamespace(reqs_info=[SimpleNamespace(reqs=[running])]),
        waiting_queue=[waiting],
        pending_dp_reqs=[pending],
        chunked_reqs=[],
        req_to_token_pool=pool,
    )
    monkeypatch.setattr(
        Scheduler,
        "get_internal_state",
        lambda self, request: SimpleNamespace(internal_state={"original_state": True}),
    )
    undo = observer.install(tmp_path, sample_pages=0)
    try:
        state = Scheduler.get_internal_state(scheduler, None).internal_state
        assert json.loads(json.dumps(state)) == state
        assert state["original_state"] is True
        requests = {row["rid"]: row for row in state["hybrid_hicache_requests"]}
        assert len(state["hybrid_hicache_requests"]) == len(requests) == 3
        # Pending routing inputs have no resource ownership yet, even when a
        # target rank is already known. Do not fabricate empty/zero KV fields.
        assert requests["pending"] == {
            "rid": "pending",
            "dp_rank": pending_rank,
            "stage": "pending_dp",
        }
        assert requests["running"] == {
            "rid": "running",
            "dp_rank": 0,
            "req_pool_idx": running.req_pool_idx,
            "kv_allocated_len": 3,
            "cache_protected_len": 2,
            "swa_evicted_seqlen": 1,
            "full_token_indices": [11, 12, 13],
        }
        assert requests["waiting"]["req_pool_idx"] is None
        assert requests["waiting"]["full_token_indices"] == []
        rows = [
            json.loads(line)
            for path in tmp_path.glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert rows[-1]["state"] == state
    finally:
        undo()


def manifest():
    return {
        "return_logprob": True,
        "sampling_params": {"no_stop_trim": True, "skip_special_tokens": False},
        "cases": [
            {
                "rid": "normal",
                "kind": "normal",
                "input_ids": [1, 2],
                "max_new_tokens": 2,
                "dp_rank": 0,
            }
        ],
        "stages": [{"rids": ["normal"]}],
    }


def test_scenarios_keep_raw_response_and_freeze_output_flags(monkeypatch):
    sent = []
    response = {"output_ids": [3, 4], "meta_info": {"id": "normal", "dp_rank": 0}}

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

    monkeypatch.setattr(scenarios.requests, "post", post)
    monkeypatch.setattr(
        scenarios.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"internal_states": []}
        ),
    )
    raw, observations = {"results": []}, []
    scenarios.run(manifest(), "http://unused", raw, observations, lambda: None)
    assert raw["results"] == [response]
    assert sent[0]["return_logprob"] is True
    assert sent[0]["sampling_params"]["no_stop_trim"] is True
    assert sent[0]["sampling_params"]["skip_special_tokens"] is False
    assert [item["label"] for item in observations] == ["stage-0-before", "stage-0-after"]


def test_scenarios_reject_duplicate_or_missing_staged_cases():
    data = manifest()
    data["stages"] = [{"rids": ["normal", "normal"]}]
    with pytest.raises(ValueError, match="exactly once"):
        scenarios.validate_manifest(data)


@pytest.mark.parametrize("running", [True, False])
def test_abort_waits_for_running_state_and_keeps_terminal_response(monkeypatch, running):
    data = manifest()
    data["cases"][0]["kind"] = "abort"
    data["stages"][0]["abort_when_running"] = "normal"
    posts = []

    def post(url, **kwargs):
        posts.append(url)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            status_code=200,
            text="",
            json=lambda: {
                "output_ids": [3],
                "meta_info": {"id": "normal", "finish_reason": {"type": "abort"}},
            },
        )

    monkeypatch.setattr(scenarios.requests, "post", post)
    monkeypatch.setattr(
        scenarios.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "internal_states": [{"running_batch_rids": ["normal"] if running else []}]
            },
        ),
    )
    raw, observations = {"results": []}, []
    if running:
        scenarios.run(data, "http://unused", raw, observations, lambda: None)
        assert "http://unused/abort_request" in posts
    else:
        with pytest.raises(RuntimeError, match="inconclusive"):
            scenarios.run(data, "http://unused", raw, observations, lambda: None)
        assert "http://unused/abort_request" not in posts
    assert len(raw["results"]) == 1


def test_spawn_entry_reinstalls_observer_without_starting_another_server(monkeypatch, tmp_path):
    import hybrid_hicache_observer as observer

    calls = []
    monkeypatch.setenv("SGLANG_HYBRID_ACCEPTANCE_TRACE", str(tmp_path))
    monkeypatch.setenv("SGLANG_HYBRID_ACCEPTANCE_SAMPLE_PAGES", "0")
    monkeypatch.setattr(observer, "install", lambda directory, **kw: calls.append((directory, kw)))
    monkeypatch.setattr(sys, "argv", ["manual-server", "--hicache-transfer-backend", "jax"])
    runpy.run_path(
        str(Path(__file__).with_name("hybrid_hicache_server.py")), run_name="__mp_main__"
    )
    assert calls == [(str(tmp_path), {"sample_pages": 0})]
