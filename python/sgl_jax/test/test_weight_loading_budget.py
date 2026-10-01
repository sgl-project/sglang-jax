"""Tiny budgets throttle real loads without rejecting indivisible weights."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.weights import LocalSource, WeightLoader, WeightSpec


@pytest.fixture
def mesh(monkeypatch):
    monkeypatch.delenv("SGLANG_PD_WEIGHT_CACHE", raising=False)
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:2]),
        ("tensor",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    with jax.set_mesh(mesh):
        yield mesh


def test_large_device_recipe_waits_before_and_after_loading(tmp_path, mesh, monkeypatch):
    monkeypatch.setenv("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", "1024")
    weights = {
        name: np.arange(size, dtype=np.float32) for name, size in (("a", 4), ("b", 64), ("c", 4))
    }
    save_file(weights, tmp_path / "model.safetensors")
    config = SimpleNamespace(model_path=str(tmp_path))
    model = nnx.Module()
    events, output_names, mappings = [], {}, {}
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    for name, weight in weights.items():
        setattr(model, name, nnx.Param(jax.ShapeDtypeStruct(weight.shape, weight.dtype)))

        def recipe(inputs, name=name):
            output = jax.device_put(inputs[0], sharding)
            output_names[id(output)] = name
            return (output,)

        mappings[name] = WeightSpec(name, sources=(name,), recipe=recipe)

    wait = jax.block_until_ready

    def record_wait(arrays):
        for array in jax.tree.leaves(arrays):
            if id(array) in output_names:
                events.append(("wait", output_names[id(array)]))
        return wait(arrays)

    monkeypatch.setattr(jax, "block_until_ready", record_wait)
    with LocalSource(config) as source:
        read = source.read_tensor

        def record_read(filename, name, index):
            events.append(("read", name))
            return read(filename, name, index)

        monkeypatch.setattr(source, "read_tensor", record_read)
        WeightLoader(model, config, mesh, source=source).load(mappings)

    for name, expected in weights.items():
        np.testing.assert_array_equal(getattr(model, name)[...], expected)
    assert events.index(("read", "a")) < events.index(("wait", "a")) < events.index(("read", "b"))
    assert events.index(("read", "b")) < events.index(("wait", "b")) < events.index(("read", "c"))


def test_large_host_recipe_loads_local_intervals(tmp_path, mesh, monkeypatch):
    monkeypatch.setenv("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", "1")
    weight = np.arange(16, dtype=np.float32).reshape(4, 4)
    save_file({"fused": weight}, tmp_path / "model.safetensors")
    config = SimpleNamespace(model_path=str(tmp_path))
    model = nnx.Module()
    model.left = nnx.Param(jax.ShapeDtypeStruct((4, 2), np.float32))
    model.right = nnx.Param(jax.ShapeDtypeStruct((4, 2), np.float32))
    input_shapes = []

    def split(inputs):
        input_shapes.append(inputs[0].shape)
        return np.split(inputs[0], 2, axis=1)

    WeightLoader(model, config, mesh).load(
        {
            "fused": WeightSpec(
                ["left", "right"], sources=("fused",), host_recipe=split, sharding=("tensor", None)
            )
        }
    )

    np.testing.assert_array_equal(model.left[...], weight[:, :2])
    np.testing.assert_array_equal(model.right[...], weight[:, 2:])
    assert input_shapes == [(4 // mesh.size, 4)] * mesh.size


@pytest.mark.parametrize("mode", ["ordinary", "split", "bulk"])
def test_large_expert_shards_use_one_worker_or_device_batch(tmp_path, mesh, monkeypatch, mode):
    from sgl_jax.srt.model_loader.weights import reader as reader_module

    monkeypatch.setenv("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", "1")
    monkeypatch.setenv("SGLANG_MOE_LOAD_WORKERS", "16")
    weights = {f"e{i}": np.arange(6, dtype=np.float32).reshape(2, 3) + 10 * i for i in range(4)}
    if mode == "split":
        save_file(
            {name: weight[:1] for name, weight in weights.items()}, tmp_path / "a.safetensors"
        )
        save_file(
            {name: weight[1:] for name, weight in weights.items()}, tmp_path / "b.safetensors"
        )
    else:
        save_file(weights, tmp_path / "model.safetensors")
    config = SimpleNamespace(model_path=str(tmp_path))
    model = nnx.Module()
    model.weight = nnx.Param(jax.ShapeDtypeStruct((4, 2, 3), np.float32))
    worker_counts, batch_sizes = [], []

    def executor(*, max_workers):
        worker_counts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(reader_module, "ThreadPoolExecutor", executor)
    with LocalSource(config) as source:
        source.prefers_bulk = mode == "bulk"
        read_ranges = source.read_ranges

        def record_ranges(ranges):
            batch_sizes.append(len(ranges))
            return read_ranges(ranges)

        monkeypatch.setattr(source, "read_ranges", record_ranges)
        WeightLoader(model, config, mesh, source=source).load(
            {
                "experts": WeightSpec(
                    "weight",
                    sources=tuple(weights),
                    sharding=("tensor", None, None),
                    concat_axis=0 if mode == "split" else None,
                )
            },
            safetensors_partition=2 if mode == "split" else 1,
        )

    np.testing.assert_array_equal(model.weight[...], np.stack(list(weights.values())))
    assert worker_counts and set(worker_counts) == {1}
    if mode == "bulk":
        assert batch_sizes == [4 // mesh.size] * mesh.size


@pytest.mark.parametrize("budget,expected_workers", [("80", 1), ("4096", 2)])
def test_expert_read_concurrency_follows_available_budget(
    tmp_path, mesh, monkeypatch, budget, expected_workers
):
    from sgl_jax.srt.model_loader.weights import reader as reader_module

    if mesh.size != 2:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=2 or more")
    monkeypatch.setenv("SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES", budget)
    weights = {f"e{i}": np.arange(6, dtype=np.float32).reshape(2, 3) + 10 * i for i in range(4)}
    save_file(weights, tmp_path / "model.safetensors")
    config = SimpleNamespace(model_path=str(tmp_path))
    model = nnx.Module()
    model.weight = nnx.Param(jax.ShapeDtypeStruct((4, 2, 3), np.float32))
    worker_counts = []

    def executor(*, max_workers):
        worker_counts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(reader_module, "ThreadPoolExecutor", executor)
    WeightLoader(model, config, mesh).load(
        {"experts": WeightSpec("weight", sources=tuple(weights), sharding=("tensor", None, None))}
    )

    np.testing.assert_array_equal(model.weight[...], np.stack(list(weights.values())))
    assert worker_counts and set(worker_counts) == {expected_workers}
