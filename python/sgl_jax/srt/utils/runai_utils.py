"""RunAI byte-range I/O for the shared JAX safetensors weight loader.

Unlike the PyTorch loaders' sequential tensor iterator, JAX requests local
shards in mapping order, including expert groups spanning multiple files.
FileStreamer preserves that contract without staging the checkpoint on disk.
"""

import hashlib
import itertools
import json
import math
import os
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

_DTYPES: dict[str, np.dtype] = {
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "BF16": np.dtype("V2"),
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "BOOL": np.dtype("bool"),
    "F8_E4M3": np.dtype("uint8"),
    "F8_E5M2": np.dtype("uint8"),
}
_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 100_000_000


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def _sdk():
    try:
        import runai_model_streamer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "RunAI loading requires the optional Linux dependency: "
            "pip install 'sglang-jax[runai]'"
        ) from exc
    return runai_model_streamer


def configure_runai(extra_config: dict) -> None:
    """Validate all options before applying the SDK's process-wide settings."""
    if not isinstance(extra_config, dict):
        raise ValueError("RunAI model-loader-extra-config must be a JSON object")
    unknown = set(extra_config) - {"concurrency", "memory_limit", "distributed"}
    if unknown:
        raise ValueError(f"Unexpected runai_streamer options: {sorted(unknown)}")
    if extra_config.get("distributed", False) is not False:
        raise ValueError(
            "RunAI distributed streaming requires torch.distributed and is not supported "
            "by the JAX loader; each JAX process reads its addressable shards."
        )
    updates = {}
    for key in ("concurrency", "memory_limit"):
        if key in extra_config:
            value = extra_config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"RunAI {key} must be a positive integer, got {value!r}")
            updates[f"RUNAI_STREAMER_{key.upper()}"] = str(value)
    os.environ.update(updates)


def download_metadata(uri: str, download_dir: str | None = None) -> str:
    """Stage config/tokenizer files once, leaving weight objects in GCS."""
    if not is_gcs_path(uri) or not uri[5:].strip("/"):
        raise ValueError(f"Expected a GCS model directory, got {uri!r}")
    root = Path(download_dir or os.getenv("HF_HOME") or Path.home() / ".cache/huggingface")
    root = root / "sglang-jax-runai"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / hashlib.sha256(uri.rstrip("/").encode()).hexdigest()
    _sdk()
    # The SDK's GCS metadata helpers call get_bucket(), which unnecessarily
    # requires storage.buckets.get. Object readers may only have list/get.
    # Keep the same source-specific, process-safe cache without that extra RPC.
    sentinel = destination / ".sglang_complete"
    with FileLock(str(destination) + ".lock"):
        if sentinel.exists():
            return str(destination)
        destination.mkdir(parents=True, exist_ok=True)
        _, prefix = _gcs_location(uri)
        count = 0
        for blob in _gcs_blobs(uri):
            if blob.name.endswith(("/", ".safetensors", ".bin", ".pt", ".pth", ".tensors")):
                continue
            if not blob.name.startswith(prefix):
                raise ValueError(f"Object is outside model prefix: {blob.name!r}")
            target = (destination / blob.name[len(prefix) :]).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Object escapes metadata directory: {blob.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            count += 1
        if not count:
            raise ValueError(f"No metadata files found at {uri!r}")
        sentinel.touch()
    return str(destination)


def _gcs_location(uri: str):
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/") + "/" if prefix.strip("/") else ""


def _gcs_blobs(uri: str, *, delimiter=None):
    from google.cloud import storage
    from runai_model_streamer_gcs.credentials.credentials import get_credentials

    bucket, prefix = _gcs_location(uri)
    # Reuse the SDK's credential selection; bucket() constructs a resource
    # without fetching bucket metadata, unlike get_bucket().
    client = storage.Client(credentials=get_credentials().gcp_credentials())
    return client.list_blobs(client.bucket(bucket), prefix=prefix, delimiter=delimiter)


def _list_safetensors(path: str):
    if not is_gcs_path(path):
        return _sdk().list_safetensors(path)
    bucket, _ = _gcs_location(path)
    return [
        f"gs://{bucket}/{blob.name}"
        for blob in _gcs_blobs(path, delimiter="/")
        if blob.name.endswith(".safetensors")
    ]


def _slice_ranges(shape: tuple[int, ...], index, itemsize: int):
    """Return output shape and contiguous byte intervals for a basic slice."""
    if not isinstance(index, tuple):
        index = (index,)
    if sum(part is Ellipsis for part in index) > 1:
        raise IndexError("Only one ellipsis is allowed")
    if any(part is Ellipsis for part in index):
        position = next(i for i, part in enumerate(index) if part is Ellipsis)
        index = (
            index[:position]
            + (slice(None),) * (len(shape) - len(index) + 1)
            + index[position + 1 :]
        )
    if len(index) > len(shape):
        raise IndexError("Too many indices for tensor")
    index += (slice(None),) * (len(shape) - len(index))
    bounds, output_shape = [], []
    for dim, part in zip(shape, index):
        if isinstance(part, (int, np.integer)):
            value = int(part) + (dim if part < 0 else 0)
            if not 0 <= value < dim:
                raise IndexError("Tensor index out of bounds")
            bounds.append((value, value + 1))
        elif isinstance(part, slice):
            start, stop, step = part.indices(dim)
            if step != 1:
                raise ValueError("RunAI weight slices require a unit stride")
            bounds.append((start, max(start, stop)))
            output_shape.append(max(0, stop - start))
        else:
            raise TypeError(f"Unsupported weight slice index: {part!r}")
    output_shape = tuple(output_shape)
    if any(start == stop for start, stop in bounds):
        return output_shape, []
    if not shape:
        return (), [(0, itemsize)]
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    pivot = len(shape) - 1
    while pivot > 0 and bounds[pivot] == (0, shape[pivot]):
        pivot -= 1
    width = (bounds[pivot][1] - bounds[pivot][0]) * strides[pivot] * itemsize
    ranges = []
    for prefix in itertools.product(*(range(a, b) for a, b in bounds[:pivot])):
        offset = sum(i * stride for i, stride in zip(prefix, strides))
        offset += bounds[pivot][0] * strides[pivot]
        ranges.append((offset * itemsize, width))
    return output_shape, ranges


class _TensorSlice:
    def __init__(self, source, path, metadata):
        self.source, self.path, self.metadata = source, path, metadata

    def __getitem__(self, index):
        meta = self.metadata
        dtype = _DTYPES[meta["dtype"]]
        shape, ranges = _slice_ranges(tuple(meta["shape"]), index, dtype.itemsize)
        if not ranges:
            return np.empty(shape, dtype=dtype)
        data = np.empty(math.prod(shape) * dtype.itemsize, dtype=np.uint8)
        self.source.read_ranges(
            [(self.path, meta["offset"] + offset, size) for offset, size in ranges],
            buffer=data,
        )
        return data.view(dtype).reshape(shape)


class _File:
    def __init__(self, source, path, metadata):
        self.source, self.path, self.metadata = source, path, metadata

    def get_slice(self, key):
        return _TensorSlice(self.source, self.path, self.metadata[key])


class RunaiWeightSource:
    """Safetensors-compatible reader with owned buffers and serialized SDK access."""

    def __init__(self, path: str, metadata_dir: str):
        self.path = path
        self.metadata_dir = metadata_dir
        self.handles: dict[str, _File] = {}
        self.weight_info: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._streamer = None

    def __enter__(self):
        sdk = _sdk()
        paths = sorted(_list_safetensors(self.path))
        index_path = Path(self.metadata_dir) / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as f:
                expected = set(json.load(f)["weight_map"].values())
            available = {path.rsplit("/", 1)[-1]: path for path in paths}
            if expected - available.keys():
                raise ValueError(f"Missing checkpoint files: {sorted(expected - available.keys())}")
            paths = [available[name] for name in sorted(expected)]
        if not paths:
            raise ValueError(f"No safetensors files found at {self.path!r}")
        self._chunks = sdk.FileChunks
        self._streamer = sdk.FileStreamer()
        self._streamer.__enter__()
        try:
            lengths = self.read_ranges([(path, 0, 8) for path in paths])
            sizes = [struct.unpack("<Q", value)[0] for value in lengths]
            if any(size <= 0 or size > _MAX_HEADER_BYTES for size in sizes):
                raise ValueError("Invalid safetensors header length")
            headers = self.read_ranges([(p, 8, size) for p, size in zip(paths, sizes)])
            for path, size, raw in zip(paths, sizes, headers):
                metadata = json.loads(raw.tobytes())
                metadata.pop("__metadata__", None)
                for key, meta in metadata.items():
                    dtype = _DTYPES.get(meta["dtype"])
                    if dtype is None:
                        raise ValueError(f"Unsupported safetensors dtype {meta['dtype']} for {key}")
                    shape = meta["shape"]
                    begin, end = meta["data_offsets"]
                    if (
                        any(not isinstance(d, int) or d < 0 for d in shape)
                        or begin < 0
                        or end - begin != math.prod(shape) * dtype.itemsize
                    ):
                        raise ValueError(f"Invalid safetensors metadata for {key}")
                    meta["offset"] = 8 + size + begin
                    self.weight_info.setdefault(key, []).append(
                        {"file": path, "shape": tuple(shape), "dtype": meta["dtype"]}
                    )
                self.handles[path] = _File(self, path, metadata)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._streamer is not None:
            self._streamer.__exit__(exc_type, exc_value, traceback)
            self._streamer = None

    def get_handle(self, filename):
        return self.handles[filename]

    def read_ranges(self, ranges, *, buffer=None):
        # A FileStreamer has one active request. MoE loading invokes readers
        # from multiple host threads, so serialize submissions, not SDK I/O.
        with self._lock:
            if self._streamer is None:
                raise RuntimeError("RunAI weight source is closed")
            if not ranges:
                return []
            # One owned allocation also lets strided tensor slices avoid a
            # second full-size copy when assembling their contiguous rows.
            if buffer is None:
                buffer = np.empty(sum(size for _, _, size in ranges), dtype=np.uint8)
            outputs = []
            cursor = 0
            for _, _, size in ranges:
                outputs.append(buffer[cursor : cursor + size])
                cursor += size
            chunk_bytes = min(
                _CHUNK_BYTES, int(os.environ.get("RUNAI_STREAMER_MEMORY_LIMIT", _CHUNK_BYTES))
            )
            if chunk_bytes <= 0:
                raise ValueError("RUNAI_STREAMER_MEMORY_LIMIT must be positive")
            requests, offsets = [], []
            for i, (path, offset, size) in enumerate(ranges):
                chunks = [min(chunk_bytes, size - p) for p in range(0, size, chunk_bytes)]
                requests.append(self._chunks(i, path, offset, chunks))
                offsets.append([0, *itertools.accumulate(chunks)])
            self._streamer.stream_files(requests, device="cpu")
            seen = set()
            for file_id, chunk_id, tensor in self._streamer.get_chunks():
                if (file_id, chunk_id) in seen:
                    raise ValueError("RunAI returned a duplicate chunk")
                begin, end = offsets[file_id][chunk_id : chunk_id + 2]
                data = tensor.numpy().reshape(-1)
                if data.nbytes != end - begin:
                    raise ValueError("RunAI returned a truncated chunk")
                # Copy before advancing: the next SDK response may reuse its
                # staging buffer while JAX still has an asynchronous transfer.
                outputs[file_id][begin:end] = data
                seen.add((file_id, chunk_id))
            if len(seen) != sum(len(offsets_) - 1 for offsets_ in offsets):
                raise ValueError("RunAI stream ended before all chunks were received")
            return outputs
