"""Check real safetensors/JAX loading; only SDK I/O uses a local file fake.

These tests cover numerical results, sharding and borrowed-buffer ownership,
not the real SDK, GCS access or loading performance.
"""

import dataclasses
import json
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.model_loader.weights.source import RunaiWeightSource
from sgl_jax.srt.utils import runai_utils


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    batches = []

    @dataclasses.dataclass
    class FileChunks:
        id: int
        path: str
        offset: int
        chunks: list[int]

    class FileStreamer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def stream_files(self, files, device):
            assert device == "cpu"
            batches.append(files)
            self.files = files

        def get_chunks(self):
            # Responses can arrive out of order and reuse the same allocation.
            buffer = np.empty(max(size for f in self.files for size in f.chunks), np.uint8)
            for f in reversed(self.files):
                path = tmp_path / f.path.rsplit("/", 1)[-1]
                offset = f.offset
                with path.open("rb") as handle:
                    for chunk_id, size in enumerate(f.chunks):
                        handle.seek(offset)
                        value = np.frombuffer(handle.read(size), np.uint8)
                        buffer[: len(value)] = value
                        yield f.id, chunk_id, SimpleNamespace(numpy=lambda n=len(value): buffer[:n])
                        buffer.fill(255)
                        offset += size

    fake = SimpleNamespace(
        FileChunks=FileChunks,
        FileStreamer=FileStreamer,
        list_safetensors=lambda root: [f"{root}/{p.name}" for p in tmp_path.glob("*.safetensors")],
        batches=batches,
    )
    monkeypatch.setattr(runai_utils, "_sdk", lambda: fake)
    monkeypatch.setattr(runai_utils, "_list_safetensors", fake.list_safetensors)
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "31")
    return fake


def test_shared_weight_loader_matches_local_sharding(sdk, tmp_path, monkeypatch):
    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.model_loader.loader import RunaiModelLoader

    if len(jax.devices()) < 4:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=4 to verify TP and expert sharding")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    weights = {
        "dense": np.arange(24, dtype=np.float32).reshape(6, 4),
        "qkv": np.arange(32, dtype=np.float32).reshape(8, 4),
        "expert.0": np.arange(24, dtype=np.float32).reshape(6, 4),
        "expert.1": np.arange(24, 48, dtype=np.float32).reshape(6, 4),
    }
    # Expert stacking must work across checkpoint files.
    save_file({k: v for k, v in weights.items() if k != "expert.1"}, tmp_path / "a.safetensors")
    save_file({"expert.1": weights["expert.1"]}, tmp_path / "b.safetensors")

    class Model(nnx.Module):
        def __init__(self, config=None, *, mesh=mesh, dtype=jnp.float32):
            self.mesh = mesh
            for name, shape in {
                "dense": (4, 6),
                "q": (4, 4),
                "k": (2, 4),
                "v": (2, 4),
                "experts": (2, 4, 6),
            }.items():
                setattr(self, name, nnx.Param(jnp.zeros(shape, dtype=dtype)))

        def load_weights(self, config):
            WeightLoader(self, config, self.mesh, jnp.float32).load(mappings)

    config = SimpleNamespace(
        model_path=str(tmp_path),
        num_attention_heads=2,
        get_total_num_kv_heads=lambda: 1,
        hidden_size=4,
        hf_config=SimpleNamespace(),
        quantization_config=None,
        ep_size=1,
        dtype=jnp.float32,
        revision=None,
    )
    mappings = {
        "dense": WeightSpec("dense", sharding=(None, "tensor"), transpose=True),
        "qkv": WeightSpec(["q", "k", "v"], sharding=("tensor", None)),
        "": WeightSpec(
            "experts",
            sources=("expert.0", "expert.1"),
            sharding=("data", None, "tensor"),
            transpose=True,
        ),
    }
    with jax.set_mesh(mesh):
        baseline = Model()
        WeightLoader(baseline, config, mesh, jnp.float32).load(mappings)
        config.model_weights = "gs://bucket/model"
        config.model_path = str(tmp_path / "metadata-only")
        monkeypatch.setattr(runai_utils, "download_metadata", lambda *_: config.model_path)
        load_config = LoadConfig(load_format="runai_streamer", model_class=Model)
        streamed = RunaiModelLoader(load_config, mesh).load_model(config)
        assert not hasattr(config, "_weight_source")
    for name in ("dense", "q", "k", "v", "experts"):
        expected, actual = getattr(baseline, name).value, getattr(streamed, name).value
        np.testing.assert_array_equal(actual, expected)
        assert actual.sharding == expected.sharding
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(streamed.dense.value, weights["dense"].T)
    np.testing.assert_array_equal(streamed.q.value, weights["qkv"][:4])
    np.testing.assert_array_equal(
        streamed.experts.value, np.stack([weights["expert.0"].T, weights["expert.1"].T])
    )


@pytest.mark.parametrize(
    "dtype_name,storage_dtype",
    [("bfloat16", "BF16"), ("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")],
)
def test_low_precision_bytes_are_reinterpreted_before_jax_transfer(
    sdk, tmp_path, dtype_name, storage_dtype
):
    import ml_dtypes

    value = np.linspace(-2, 2, 16).astype(getattr(ml_dtypes, dtype_name)).reshape(4, 4)
    # Construct the portable safetensors layout: older NumPy writers cannot
    # serialize FP8, even though the existing JAX loader supports these dtypes.
    header = json.dumps(
        {"weight": {"dtype": storage_dtype, "shape": [4, 4], "data_offsets": [0, value.nbytes]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + value.tobytes()
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",), axis_types=(AxisType.Explicit,))

    class Model(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.zeros((4, 4), dtype=jnp.bfloat16))

    with RunaiWeightSource("gs://bucket/model", str(tmp_path)) as source, jax.set_mesh(mesh):
        model = Model()
        config = SimpleNamespace(
            model_path=str(tmp_path), _weight_source=source, quantization_config=None
        )
        WeightLoader(model, config, mesh).load(
            {"weight": WeightSpec("weight", sharding=(None, "tensor"), transpose=True)}
        )
    np.testing.assert_array_equal(np.asarray(model.weight[...]), value.T.astype(ml_dtypes.bfloat16))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("storage_dtype", ["F32", "F8_E4M3"])
def test_runai_batches_cross_file_experts_and_scales(sdk, tmp_path, transpose, storage_dtype):
    """Real JAX assembly, duplicate expert placement, and one SDK submission."""
    if len(jax.devices()) < 4:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=4")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("expert",), axis_types=(AxisType.Explicit,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("expert", None, None))
    import ml_dtypes

    dtype = np.float32 if storage_dtype == "F32" else ml_dtypes.float8_e4m3fn
    weights = [(np.arange(12).reshape(3, 4) + i * 20).astype(dtype) for i in range(3)]
    for i, weight in enumerate(weights):
        header = json.dumps(
            {
                f"expert.{i}": {
                    "dtype": storage_dtype,
                    "shape": [3, 4],
                    "data_offsets": [0, weight.nbytes],
                }
            }
        ).encode()
        header += b" " * (-len(header) % 8)
        (tmp_path / f"{i}.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + weight.tobytes()
        )
    with RunaiWeightSource("gs://bucket/model", str(tmp_path)) as source, jax.set_mesh(mesh):
        config = SimpleNamespace(quantization_config=None)
        loader = WeightLoader(nnx.Module(), config, mesh, jnp.float32, source=source)
        sdk.batches.clear()
        actual = loader.reader.read(
            source,
            "experts",
            WeightSpec(
                "w",
                sources=tuple(f"expert.{i}" for i in range(3)),
                transpose=transpose,
                physical_to_logical_map=np.array([2, 0, 2, 1]),
            ),
            sharding,
        )
        actual.block_until_ready()
    expected = np.stack([weights[i] for i in [2, 0, 2, 1]])
    if transpose:
        expected = expected.transpose(0, 2, 1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.sharding == sharding
    assert len(sdk.batches) == 1
    assert len(sdk.batches[0]) == 3  # Duplicate logical expert is read once.
    assert sum(sum(f.chunks) for f in sdk.batches[0]) == sum(w.nbytes for w in weights)
