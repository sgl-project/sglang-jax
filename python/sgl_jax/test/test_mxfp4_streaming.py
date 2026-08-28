"""Streamed, EP-filtered weight reads must be byte-identical to reading the shard locally.

This is the path that makes the full 93-layer model loadable at all: 1.42 TiB across 96 shards
against ~919 GB of host tmpfs. A streaming reader that is subtly wrong would produce a model that
loads and computes garbage, so the central test compares streamed tensors against the SAME tensors
read from a locally-staged shard with safetensors -- ground truth, not a re-implementation of the
same parsing.

Needs GCS access and a staged shard to compare against; skips otherwise rather than pretending.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from sgl_jax.srt.layers.quantization.mxfp4_streaming import (
    EXPERT_SUFFIXES,
    ShardReader,
    list_shards,
    local_expert_ids,
    parse_expert_id,
    parse_gs_uri,
    plan_fetch,
    should_skip,
    stream_shard,
)

MODEL_DIR = os.environ.get("KIMI_K3_MODEL_DIR", "/dev/shm/k3_4l")
GS_URI = os.environ.get("KIMI_K3_GS_URI", "gs://torchtitan-assets/moonshootai/kimi/3")


def _gcs_client():
    try:
        from google.cloud import storage
    except ImportError:
        pytest.skip("google-cloud-storage not installed")
    try:
        return storage.Client()
    except Exception as exc:  # noqa: BLE001 - no credentials in this environment
        pytest.skip(f"no GCS client: {exc}")


def _local_shard(*, with_experts: bool = False) -> str:
    """A staged shard, optionally one that actually carries per-expert tensors.

    Shard 1 holds embeddings and attention -- K3 quantizes ``Linear`` targets only -- so tests
    about expert filtering must not land on it, or they skip for the wrong reason.
    """
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    if not files:
        pytest.skip(f"no staged shard under {MODEL_DIR}")
    if not with_experts:
        return files[0]

    from safetensors import safe_open

    for path in files:
        with safe_open(path, "numpy") as h:
            if any(parse_expert_id(k) is not None for k in h.keys()):
                return path
    pytest.skip("no staged shard carries expert tensors")


# ----------------------------------------------------------------------------------------------
# expert filtering -- pure, no I/O
# ----------------------------------------------------------------------------------------------
def test_expert_ids_partition_exactly():
    """Every expert is owned by exactly one rank. A gap loads as zeros; an overlap wastes bytes."""
    num_experts, ep_size = 896, 32
    seen: set[int] = set()
    for rank in range(ep_size):
        ids = local_expert_ids(num_experts, ep_size, rank)
        assert len(ids) == num_experts // ep_size == 28
        assert not (ids & seen), f"rank {rank} overlaps an earlier rank"
        seen |= ids
    assert seen == set(range(num_experts))


def test_expert_ids_reject_indivisible_split():
    with pytest.raises(ValueError, match="divisible"):
        local_expert_ids(896, 30, 0)


def test_only_per_expert_tensors_are_skippable():
    """Dense layers, shared experts, the router and norms are needed by EVERY rank.

    Widening the skip rule to these is the failure that silently zeroes shared state, so it is
    pinned by name here.
    """
    local = {0, 1}
    stem = "language_model.model.layers.1.block_sparse_moe"

    assert should_skip(f"{stem}.experts.7.w1.weight_packed", local)
    assert should_skip(f"{stem}.experts.7.w1.weight_scale", local)
    assert not should_skip(f"{stem}.experts.0.w1.weight_packed", local)

    for shared in (
        f"{stem}.gate.weight",
        f"{stem}.shared_experts.gate_proj.weight",
        f"{stem}.routed_expert_down_proj.weight",
        "language_model.model.layers.1.input_layernorm.weight",
        "language_model.model.layers.1.self_attn.q_proj.weight",
    ):
        assert not should_skip(shared, local), shared

    # filtering off -> nothing is skipped
    assert not should_skip(f"{stem}.experts.7.w1.weight_packed", None)


def test_parse_expert_id():
    stem = "model.layers.3.block_sparse_moe.experts"
    assert parse_expert_id(f"{stem}.0.w1.weight_packed") == 0
    assert parse_expert_id(f"{stem}.895.w2.weight_scale") == 895
    assert parse_expert_id("model.layers.3.self_attn.q_proj.weight") is None


def test_expert_suffix_list_is_exact():
    """The suffix list is exact on purpose: widening it silently drops shared state."""
    assert EXPERT_SUFFIXES == (".weight", ".weight_packed", ".weight_scale")


def test_parse_gs_uri():
    assert parse_gs_uri("gs://b/p/q") == ("b", "p/q")
    assert parse_gs_uri("gs://b/p/") == ("b", "p")
    with pytest.raises(ValueError):
        parse_gs_uri("/local/path")


# ----------------------------------------------------------------------------------------------
# streamed reads vs the local file -- the correctness test
# ----------------------------------------------------------------------------------------------
def test_header_matches_local_safetensors():
    """Parsed spans must agree with safetensors' own view of the same shard."""
    from safetensors import safe_open

    client = _gcs_client()
    bucket, prefix = parse_gs_uri(GS_URI)
    local = _local_shard()
    shard_name = f"{prefix}/{os.path.basename(local)}"

    reader = ShardReader(bucket, shard_name, client=client)
    spans = reader.spans

    with safe_open(local, "numpy") as h:
        keys = list(h.keys())
        assert set(spans) == set(keys), "streamed header lists different tensors"
        for key in keys[:20]:
            assert spans[key].shape == tuple(h.get_slice(key).get_shape()), key


def test_streamed_tensor_bytes_equal_local_read():
    """A ranged GET must reproduce the local tensor EXACTLY -- not approximately.

    Compared on the uint8 tensors K3's MXFP4 weights and scales actually are. bf16 tensors are
    excluded from the *numpy* comparison because numpy cannot represent bfloat16 at all
    (``TypeError: data type 'bfloat16' not understood``) -- that is a numpy limit, not a reader
    one, and the reader keeps bf16 as raw uint16 pairs for the caller to reinterpret.
    """
    from safetensors import safe_open

    client = _gcs_client()
    bucket, prefix = parse_gs_uri(GS_URI)
    local = _local_shard(with_experts=True)
    reader = ShardReader(bucket, prefix + "/" + os.path.basename(local), client=client)

    names = [n for n in sorted(reader.spans) if n.endswith("weight_packed")][:2]
    names += [n for n in sorted(reader.spans) if n.endswith("weight_scale")][:2]
    assert names, "expected packed MXFP4 tensors in an expert-bearing shard"

    with safe_open(local, "numpy") as h:
        for name in names:
            got = reader.read(reader.spans[name])
            want = h.get_tensor(name)
            assert got.dtype == want.dtype == np.uint8, (name, got.dtype, want.dtype)
            assert got.shape == want.shape, name
            np.testing.assert_array_equal(got, want, err_msg=name)


def test_filter_actually_reduces_bytes_fetched():
    """The saving must be in bytes REQUESTED, not merely bytes retained.

    An implementation that downloads the shard and then discards experts would pass a
    correctness test while doing nothing for the constraint that motivates this code.
    """
    client = _gcs_client()
    bucket, prefix = parse_gs_uri(GS_URI)
    local = _local_shard(with_experts=True)
    reader = ShardReader(bucket, prefix + "/" + os.path.basename(local), client=client)
    assert any(parse_expert_id(n) is not None for n in reader.spans)

    full = plan_fetch(reader, local_ids=None)
    ep32 = plan_fetch(reader, local_ids=local_expert_ids(896, 32, 0))

    assert ep32.kept_bytes < full.kept_bytes
    assert ep32.skipped_bytes > 0
    # 28 of 896 experts: expert bytes should collapse by ~32x, so the shard's total must fall a
    # long way even with the non-expert tensors every rank keeps.
    assert ep32.kept_bytes < 0.5 * full.kept_bytes, ep32.summary()


def test_stream_shard_yields_only_kept_tensors():
    client = _gcs_client()
    bucket, prefix = parse_gs_uri(GS_URI)
    local = _local_shard(with_experts=True)
    reader = ShardReader(bucket, prefix + "/" + os.path.basename(local), client=client)

    local_ids = local_expert_ids(896, 32, 0)
    # cap the work: only the first few tensors of a `want` subset
    wanted = [n for n in sorted(reader.spans) if n.endswith("weight_scale")][:3]
    assert wanted, "expert-bearing shard must carry scales"

    seen = dict(stream_shard(reader, local_ids, want=lambda n: n in wanted))
    for name, arr in seen.items():
        assert not should_skip(name, local_ids), name
        assert arr.shape == reader.spans[name].shape


def test_list_shards_finds_the_release():
    client = _gcs_client()
    bucket, prefix = parse_gs_uri(GS_URI)
    shards = list_shards(bucket, prefix, client=client)
    assert len(shards) >= 90, f"expected the 96-shard release, got {len(shards)}"
    assert all(s.endswith(".safetensors") for s in shards)
