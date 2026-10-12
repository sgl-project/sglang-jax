"""Startup warning for the DSA sparse-prefill score-buffer OOM cliff.

Exercises ModelRunner._maybe_warn_dsa_sparse_prefill_temporaries with a mocked
runner: the method is pure glue over prefill_score_temp_bytes (tested in
test/srt/kernels/dsa/test_ref.py) plus a free-HBM probe.
"""

import logging
import types
from unittest.mock import patch

import pytest

from sgl_jax.srt.model_executor.model_runner import ModelRunner

# The observed 135k-context OOM: 22 full layers, chunk 8192 -> ~97.4 GB of
# score buffers against ~94.75 GB free.
_FREE_BYTES_135K = int(94.75e9)


def _fake_runner(
    *,
    attention_backend="dsa_sparse",
    use_mla_backend=True,
    chunked_prefill_size=8192,
    max_prefill_tokens=16384,
    context_len=135168,
    num_hidden_layers=78,
):
    sa = types.SimpleNamespace(
        attention_backend=attention_backend,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        device_indexes=None,
    )
    # GLM-5.2 IndexShare pattern: 22 "full" indexer layers out of 78.
    types_list = ["full"] * 3
    while len(types_list) < num_hidden_layers:
        types_list += ["shared"] * 3 + ["full"]
    types_list = types_list[:num_hidden_layers]
    cfg = types.SimpleNamespace(
        indexer_types=types_list,
        index_skip_topk_offset=3,
        num_hidden_layers=num_hidden_layers,
    )
    return types.SimpleNamespace(
        server_args=sa,
        use_mla_backend=use_mla_backend,
        device="tpu",
        model_config=types.SimpleNamespace(hf_text_config=cfg, context_len=context_len),
    )


def _run(runner, free_bytes, monkeypatch, env=None):
    env = {"DSA_PREFILL_SPARSE": "1", **(env or {})}
    for k in ("DSA_PREFILL_SPARSE", "DSA_INDEXER_KERNEL_PREFILL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with patch(
        "sgl_jax.srt.model_executor.model_runner.get_available_device_memory",
        return_value=free_bytes,
    ):
        ModelRunner._maybe_warn_dsa_sparse_prefill_temporaries(runner)


def test_warns_on_observed_135k_oom_config(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _run(_fake_runner(), _FREE_BYTES_135K, monkeypatch)
    assert any("RESOURCE_EXHAUSTED" in r.message for r in caplog.records)
    # The suggested chunk must be the one that actually survived in practice.
    assert any("4096" in r.message for r in caplog.records)


def test_silent_when_chunk_fits(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _run(_fake_runner(chunked_prefill_size=4096), _FREE_BYTES_135K, monkeypatch)
    assert not caplog.records


@pytest.mark.parametrize(
    "env",
    [
        {"DSA_PREFILL_SPARSE": "0"},
        {"DSA_INDEXER_KERNEL_PREFILL": "1"},
    ],
)
def test_silent_when_path_not_active(env, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _run(_fake_runner(), _FREE_BYTES_135K, monkeypatch, env=env)
    assert not caplog.records


def test_silent_on_other_backends(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _run(_fake_runner(attention_backend="fa"), _FREE_BYTES_135K, monkeypatch)
    assert not caplog.records


def test_uses_max_prefill_tokens_when_chunking_disabled(monkeypatch, caplog):
    runner = _fake_runner(chunked_prefill_size=-1, max_prefill_tokens=16384)
    with caplog.at_level(logging.WARNING):
        _run(runner, _FREE_BYTES_135K, monkeypatch)
    assert any("chunk 16384" in r.message for r in caplog.records)


def test_probe_failure_never_raises(monkeypatch):
    runner = _fake_runner()
    for k in ("DSA_INDEXER_KERNEL_PREFILL",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DSA_PREFILL_SPARSE", "1")
    with patch(
        "sgl_jax.srt.model_executor.model_runner.get_available_device_memory",
        side_effect=RuntimeError("no backend"),
    ):
        ModelRunner._maybe_warn_dsa_sparse_prefill_temporaries(runner)  # no raise
