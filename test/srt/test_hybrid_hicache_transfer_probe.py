"""CPU checks for probe guards and per-layer/page evidence, not TPU transfers."""

from types import SimpleNamespace

import hybrid_hicache_transfer_probe as probe
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hybrid_hicache_transfer_probe import _pages, _verify_component, run_transfer_probe
from jax.sharding import AxisType, Mesh


def test_page_evidence_checks_every_layer_and_page():
    pages = [
        jnp.arange(48, dtype=jnp.float32).reshape(3, 2, 2, 2, 2),
        jnp.arange(48, dtype=jnp.float32).reshape(3, 2, 2, 2, 2) + 100,
    ]
    pool = SimpleNamespace(
        page_size=2,
        dp_size=1,
        kv_buffer=pages,
        kv_partition_axis="tensor",
        mesh=Mesh(
            np.asarray(jax.devices()[:1]).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        ),
    )
    indices = np.array([2, 3, 4, 5], dtype=np.int32)
    expected = [np.asarray(buf)[1:3].copy() for buf in pages]
    rows = _verify_component(pool, indices, 0, expected, [7, 13])
    assert [(r["model_layer"], r["device_page"]) for r in rows] == [
        (7, 1),
        (7, 2),
        (13, 1),
        (13, 2),
    ]
    expected[1][1, 0, 0, 0, 0] = -1
    with pytest.raises(AssertionError):
        _verify_component(pool, indices, 0, expected, [7, 13])


def test_page_contract_rejects_partial_and_noncontiguous_page():
    with pytest.raises(AssertionError):
        _pages([2, 3, 4], 2)
    with pytest.raises(AssertionError):
        _pages([2, 4], 2)


def test_probe_requires_explicit_opt_in_before_accessing_scheduler():
    with pytest.raises(ValueError, match="destructive_opt_in"):
        run_transfer_probe(None)


def test_marker_distinguishes_every_page_token_head_kv_and_dimension(monkeypatch):
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    pool = SimpleNamespace(
        page_size=2,
        dp_size=1,
        kv_buffer=[jnp.zeros((3, 2, 2, 2, 2), dtype=jnp.bfloat16)],
        kv_partition_axis="tensor",
        attention_data_partition_axis="data",
        mesh=mesh,
    )
    written = []

    def record(values, loc, buf, *args):
        written.append(np.asarray(values).reshape(2, 2, 2, 2, 2))
        return buf

    monkeypatch.setattr(probe, "write_kv_layer", record)
    probe._write_marker(pool, np.array([2, 3, 4, 5]), 0, 1)
    for axis in range(5):
        assert not np.array_equal(written[0], np.roll(written[0], 1, axis=axis)), axis


def test_verify_component_rejects_missing_expected_pages():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    pool = SimpleNamespace(
        page_size=2,
        dp_size=1,
        kv_buffer=[jnp.zeros((3, 2, 2, 2, 2))],
        kv_partition_axis="tensor",
        mesh=mesh,
    )
    with pytest.raises(AssertionError):
        _verify_component(pool, np.array([2, 3]), 0, [np.zeros((2, 2, 2, 2, 2))], [0])


def test_bf16_multidigit_signatures_identify_every_coordinate():
    # Include token offsets beyond BF16's consecutive-integer precision and
    # beyond one marker byte. One scalar/arange cast cannot satisfy this.
    dimensions = (2, 512, 2, 2, 4)
    shape = dimensions[1:]
    signatures = []
    for marker in range(4):  # two ranks x FULL/SWA
        for layer in range(2):
            digits = [
                probe._marker_values(shape, marker, layer, digit, dimensions).astype(jnp.bfloat16)
                for digit in range(3)
            ]
            code = sum(
                values.astype(np.uint64) << (8 * digit) for digit, values in enumerate(digits)
            )
            expected = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
            expected += np.uint64((marker * dimensions[0] + layer) * np.prod(shape))
            np.testing.assert_array_equal(code, expected)
            signatures.append(code.reshape(-1))
    combined = np.concatenate(signatures)
    assert len(np.unique(combined)) == len(combined)


@pytest.mark.parametrize("axis", range(5), ids=["page", "token", "head", "kv", "dimension"])
def test_position_markers_detect_axis_permutations(axis):
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    pool = SimpleNamespace(
        page_size=256, dp_size=1, kv_buffer=[], kv_partition_axis="tensor", mesh=mesh
    )
    dimensions = (1, 512, 2, 2, 2)
    rejected = []
    for digit in range(2):
        expected = probe._marker_values(dimensions[1:], 0, 0, digit, dimensions)
        expected = expected.reshape(2, 256, 2, 2, 2).astype(jnp.bfloat16)
        wrong = np.roll(expected, 1, axis=axis)
        pool.kv_buffer = [jax.device_put(np.concatenate([np.zeros_like(wrong[:1]), wrong]))]
        try:
            _verify_component(pool, np.arange(256, 768), 0, [expected], [0])
        except AssertionError:
            rejected.append(digit)
    assert rejected, f"axis {axis} permutation survived every marker digit"


@pytest.mark.parametrize("policy", ["write_through", "write_back"])
def test_cpu_probe_exercises_multidigit_dp_split_paths(monkeypatch, tmp_path, policy):
    # Only the TPU guard is bypassed. The actual cache/allocator/JAX host-copy
    # control flow runs through the existing CPU scatter shim, not real DMA.
    from sgl_jax.test.mem_cache import conftest  # noqa: F401
    from sgl_jax.test.mem_cache.test_hybrid_hicache import make_cache, shutdown

    assert len(jax.devices()) >= 2, "run this suite with two simulated CPU devices"
    cache, allocator, pool = make_cache(page=256, window=128, dp_size=2, policy=policy)
    flushes = []

    def flush():
        cache.reset()
        allocator.clear()
        flushes.append(True)
        return True, "", 0

    scheduler = SimpleNamespace(
        is_fully_idle=lambda: True,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        tp_worker=SimpleNamespace(get_model_runner=lambda: SimpleNamespace(token_to_kv_pool=pool)),
        server_args=SimpleNamespace(model_path="cpu-fixture", revision="test"),
        flush_cache=flush,
    )
    monkeypatch.setattr(probe.jax, "default_backend", lambda: "tpu")
    try:
        report = run_transfer_probe(
            scheduler, destructive_opt_in=True, report_path=tmp_path / "probe.json"
        )
        assert report["status"] == "passed"
        assert report["marker_digits"] == 2
        assert len(report["cases"]) == 8
        assert len(flushes) == 9
        assert report["cleanup"] == "flushed_pristine"
        for case in report["cases"]:
            assert len(case["pages"]["FULL"]) == 4  # two layers x two pages
            assert len(case["pages"]["SWA"]) == (1 if case["mode"] == "dual" else 2)
            assert case["restored_prefix_tokens"] == (512 if case["mode"] == "dual" else 256)
            assert case["restored_component_tokens"] == {
                "FULL": 512 if case["mode"] == "dual" else 0,
                "SWA": 256,
            }
        for rank in range(2):
            for mode in ("dual", "swa_only"):
                cases = [c for c in report["cases"] if c["rank"] == rank and c["mode"] == mode]
                assert (
                    cases[0]["pages"]["FULL"][0]["sha256"] != cases[1]["pages"]["FULL"][0]["sha256"]
                )
    finally:
        shutdown(cache)
