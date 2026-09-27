"""CPU coverage of RunAI I/O with real safetensors and a borrowed-buffer SDK fake."""

import argparse
import dataclasses
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh
from safetensors.numpy import save_file

from sgl_jax.srt.utils import runai_utils
from sgl_jax.srt.utils.runai_utils import RunaiWeightSource, configure_runai
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    requests = []
    instances = []

    @dataclasses.dataclass
    class FileChunks:
        id: int
        path: str
        offset: int
        chunks: list[int]

    class FileStreamer:
        def __enter__(self):
            self.closed = False
            instances.append(self)
            return self

        def __exit__(self, *_):
            self.closed = True

        def stream_files(self, files, device):
            assert device == "cpu"
            requests.extend(files)
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

    batches = []
    fake = SimpleNamespace(
        FileChunks=FileChunks,
        FileStreamer=FileStreamer,
        list_safetensors=lambda root: [f"{root}/{p.name}" for p in tmp_path.glob("*.safetensors")],
        requests=requests,
        batches=batches,
        instances=instances,
    )
    monkeypatch.setattr(runai_utils, "_sdk", lambda: fake)
    monkeypatch.setattr(runai_utils, "_list_safetensors", fake.list_safetensors)
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "31")
    return fake


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int64, np.bool_])
def test_slices_read_only_requested_bytes_and_own_buffers(sdk, tmp_path, dtype):
    value = np.arange(3 * 4 * 6).reshape(3, 4, 6).astype(dtype)
    save_file({"weight": value}, tmp_path / "model.safetensors")
    with RunaiWeightSource("gs://bucket/model", str(tmp_path)) as source:
        tensor = source.get_handle("gs://bucket/model/model.safetensors").get_slice("weight")
        for index in [
            (...,),
            (slice(1, 3), slice(None), slice(2, 5)),
            (-1, Ellipsis, 2),
            (slice(None), 1),
            (slice(2, 1),),
            (1, 2, 3),
        ]:
            sdk.requests.clear()
            actual = tensor[index]
            expected = value[index]
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
            assert sum(sum(f.chunks) for f in sdk.requests) == expected.nbytes
        with pytest.raises(ValueError, match="unit stride"):
            tensor[::2]
    assert sdk.instances[0].closed


def test_scalar_empty_and_bf16_bytes(sdk, tmp_path):
    import ml_dtypes

    tensors = {
        "scalar": np.array(3.5, np.float32),
        "empty": np.empty((0, 3), np.float32),
        "bf16": np.arange(12).astype(ml_dtypes.bfloat16).reshape(3, 4),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    with RunaiWeightSource(str(tmp_path), str(tmp_path)) as source:
        handle = source.get_handle(str(tmp_path / "model.safetensors"))
        for key, expected in tensors.items():
            actual = handle.get_slice(key)[...]
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()


def test_index_filters_other_checkpoints_and_missing_files_fail(sdk, tmp_path):
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    save_file({"weight": np.zeros(4, np.float32)}, tmp_path / "other.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    with RunaiWeightSource("gs://bucket/model", str(tmp_path)) as source:
        assert len(source.weight_info["weight"]) == 1
        assert all(f.path.endswith("/model.safetensors") for f in sdk.requests)
    index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    with (
        pytest.raises(ValueError, match="Missing checkpoint files"),
        RunaiWeightSource("gs://bucket/model", str(tmp_path)),
    ):
        pass


def test_invalid_header_closes_stream(sdk, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\xff" * 8)
    with (
        pytest.raises(ValueError, match="header length"),
        RunaiWeightSource(str(tmp_path), str(tmp_path)),
    ):
        pass
    assert sdk.instances[0].closed


@pytest.mark.parametrize(
    "options",
    [{"concurrency": 0}, {"memory_limit": True}, {"distributed": True}, {"typo": 1}, []],
)
def test_invalid_options_do_not_partially_mutate_environment(monkeypatch, options):
    monkeypatch.setenv("RUNAI_STREAMER_CONCURRENCY", "17")
    with pytest.raises(ValueError):
        configure_runai(options)
    import os

    assert os.environ["RUNAI_STREAMER_CONCURRENCY"] == "17"


def test_missing_sdk_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "runai_model_streamer", None)
    with pytest.raises(RuntimeError, match=r"sglang-jax\[runai\]"):
        runai_utils._sdk()


def test_server_args_keep_separate_remote_draft_source(monkeypatch, tmp_path):
    from sgl_jax.srt.server_args import ServerArgs

    calls = []

    def download(uri, download_dir):
        calls.append(uri)
        return str(tmp_path / uri.rsplit("/", 1)[-1])

    monkeypatch.setattr(runai_utils, "download_metadata", download)
    args = ServerArgs(
        model_path="gs://bucket/main",
        speculative_draft_model_path="gs://bucket/draft",
    )
    assert calls == ["gs://bucket/main", "gs://bucket/draft"]
    assert args.load_format == "runai_streamer"
    assert args.served_model_name == "gs://bucket/main"
    assert args.tokenizer_path == args.model_path == str(tmp_path / "main")
    assert args.runai_model_paths[args.model_path] == "gs://bucket/main"
    assert args.runai_model_paths[args.speculative_draft_model_path] == "gs://bucket/draft"
    reconstructed = ServerArgs(**dataclasses.asdict(args))
    assert reconstructed.runai_model_paths == args.runai_model_paths
    assert len(calls) == 2


def test_cli_auto_selects_runai_without_changing_local_defaults(monkeypatch, tmp_path):
    from sgl_jax.srt.server_args import ServerArgs

    monkeypatch.setattr(runai_utils, "download_metadata", lambda *_: str(tmp_path))
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    remote = ServerArgs.from_cli_args(parser.parse_args(["--model-path", "gs://bucket/model"]))
    assert remote.load_format == "runai_streamer"
    assert remote.runai_model_paths == {str(tmp_path): "gs://bucket/model"}
    local = ServerArgs.from_cli_args(parser.parse_args(["--model-path", str(tmp_path)]))
    assert local.load_format == "auto"
    assert local.runai_model_paths == {}


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
            WeightLoader(self, config, self.mesh, jnp.float32).load_weights_from_safetensors(
                mappings
            )

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
        "dense": WeightMapping("dense", sharding=(None, "tensor"), transpose=True),
        "qkv": WeightMapping(["q", "k", "v"], sharding=("tensor", None)),
        "__MOE_EXPERTS__": WeightMapping(
            ["experts", "expert.0", "expert.1"], sharding=("data", None, "tensor"), transpose=True
        ),
    }
    with jax.set_mesh(mesh):
        baseline = Model()
        WeightLoader(baseline, config, mesh, jnp.float32).load_weights_from_safetensors(mappings)
        config.model_weights = "gs://bucket/model"
        config.model_path = str(tmp_path / "metadata-only")
        monkeypatch.setattr(runai_utils, "download_metadata", lambda *_: config.model_path)
        load_config = LoadConfig(load_format="runai_streamer", model_class=Model)
        streamed = RunaiModelLoader(load_config, mesh).load_model(config)
        assert not hasattr(config, "_runai_weight_source")
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


def test_metadata_cache_is_source_specific_and_excludes_weights(monkeypatch, tmp_path):
    calls = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def download_to_filename(self, target):
            calls.append(self.name)
            Path(target).write_text("{}")

    monkeypatch.setattr(runai_utils, "_sdk", lambda: None)
    monkeypatch.setattr(
        runai_utils,
        "_gcs_blobs",
        lambda uri: [
            Blob(uri.split("/", 3)[3].rstrip("/") + "/" + name)
            for name in ("config.json", "nested/tokenizer.json", "model.safetensors", "model.bin")
        ],
    )
    main = runai_utils.download_metadata("gs://bucket/model", str(tmp_path))
    draft = runai_utils.download_metadata("gs://bucket/draft", str(tmp_path))
    assert main != draft
    assert runai_utils.download_metadata("gs://bucket/model/", str(tmp_path)) == main
    assert calls == [
        "model/config.json",
        "model/nested/tokenizer.json",
        "draft/config.json",
        "draft/nested/tokenizer.json",
    ]


def test_gcs_listing_never_fetches_bucket_metadata(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def bucket(self, name):
            return name

        def get_bucket(self, name):
            raise AssertionError("storage.buckets.get is not required")

        def list_blobs(self, bucket, **kwargs):
            calls.append((bucket, kwargs))
            return [
                SimpleNamespace(name="model/weights.safetensors"),
                SimpleNamespace(name="model/config.json"),
            ]

    monkeypatch.setitem(
        sys.modules, "google.cloud", SimpleNamespace(storage=SimpleNamespace(Client=Client))
    )
    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer_gcs.credentials.credentials",
        SimpleNamespace(get_credentials=lambda: SimpleNamespace(gcp_credentials=lambda: None)),
    )
    assert runai_utils._list_safetensors("gs://bucket/model") == [
        "gs://bucket/model/weights.safetensors"
    ]
    assert calls == [("bucket", {"prefix": "model/", "delimiter": "/"})]


def test_failed_metadata_download_is_retried_and_path_escape_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(runai_utils, "_sdk", lambda: None)
    state = {"fail": True, "calls": 0}

    def download(target):
        state["calls"] += 1
        Path(target).write_text("partial" if state["fail"] else "complete")
        if state["fail"]:
            raise OSError("network failure")

    blob = SimpleNamespace(name="model/config.json", download_to_filename=download)
    monkeypatch.setattr(runai_utils, "_gcs_blobs", lambda uri: [blob])
    with pytest.raises(OSError, match="network failure"):
        runai_utils.download_metadata("gs://bucket/model", str(tmp_path))
    state["fail"] = False
    result = runai_utils.download_metadata("gs://bucket/model", str(tmp_path))
    assert (Path(result) / "config.json").read_text() == "complete"
    assert state["calls"] == 2
    blob.name = "escape/../../outside.json"
    with pytest.raises(ValueError, match="escapes metadata"):
        runai_utils.download_metadata("gs://bucket/escape", str(tmp_path))


@pytest.mark.parametrize(
    "model_type,multimodal", [("gemma4", False), ("qwen3_5", False), ("qwen3_vl", True)]
)
def test_custom_local_file_loaders_fail_before_download(monkeypatch, model_type, multimodal):
    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.model_loader.loader import RunaiModelLoader

    loader = RunaiModelLoader(LoadConfig(load_format="runai_streamer"), mesh=None)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type), is_multimodal=multimodal
    )
    with pytest.raises(ValueError, match="additional local-file loaders"):
        loader.load_model(config)


def test_model_config_preserves_primary_and_draft_sources(monkeypatch, tmp_path):
    import sgl_jax.srt.configs.model_config as module
    from sgl_jax.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path=str(tmp_path / "main"), speculative_draft_model_path=str(tmp_path / "draft")
    )
    args.runai_model_paths = {
        args.model_path: "gs://bucket/main",
        args.speculative_draft_model_path: "gs://bucket/draft",
    }
    build = module.ModelConfig.from_server_args
    monkeypatch.setattr(module, "ModelConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    assert build(args).model_weights == "gs://bucket/main"
    assert (
        build(args, model_path=args.speculative_draft_model_path, is_draft_model=True).model_weights
        == "gs://bucket/draft"
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
            model_path=str(tmp_path), _runai_weight_source=source, quantization_config=None
        )
        WeightLoader(model, config, mesh).load_weights_from_safetensors(
            {"weight": WeightMapping("weight", sharding=(None, "tensor"), transpose=True)}
        )
    np.testing.assert_array_equal(np.asarray(model.weight[...]), value.T.astype(ml_dtypes.bfloat16))


def test_truncated_tensor_does_not_return_partial_weights(sdk, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": np.arange(16, dtype=np.float32)}, path)
    path.write_bytes(path.read_bytes()[:-4])
    with (
        RunaiWeightSource(str(tmp_path), str(tmp_path)) as source,
        pytest.raises(ValueError, match="truncated chunk"),
    ):
        source.get_handle(str(path)).get_slice("weight")[:]


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("storage_dtype", ["F32", "F8_E4M3"])
def test_runai_batches_cross_file_experts_and_scales(sdk, tmp_path, transpose, storage_dtype):
    """Real JAX assembly, duplicate expert placement, and one native I/O batch."""
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
        config = SimpleNamespace(_runai_weight_source=source, quantization_config=None)
        loader = WeightLoader(nnx.Module(), config, mesh, jnp.float32)
        sdk.requests.clear()
        sdk.batches.clear()
        actual = loader._create_stacked_moe_lazy_tensor(
            [f"expert.{i}" for i in range(3)],
            source.weight_info,
            source,
            do_transpose=transpose,
            target_sharding=sharding,
            physical_to_logical_map=np.array([2, 0, 2, 1]),
        )
        actual.block_until_ready()
    expected = np.stack([weights[i] for i in [2, 0, 2, 1]])
    if transpose:
        expected = expected.transpose(0, 2, 1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.sharding == sharding
    assert len(sdk.batches) == 1
    assert len(sdk.batches[0]) == 3  # Duplicate logical expert is read once.
    assert sum(sum(f.chunks) for f in sdk.requests) == sum(w.nbytes for w in weights)
