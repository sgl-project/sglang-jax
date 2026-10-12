import glob
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import struct
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import multihost_utils
from safetensors import safe_open

logger = logging.getLogger(__name__)

# safetensors 0.6.x resolves FP8 through NumPy attributes; NumPy delegates the
# actual dtype to ml_dtypes but does not expose these names itself.
if not hasattr(np, "float8_e4m3fn"):
    np.float8_e4m3fn = ml_dtypes.float8_e4m3fn
if not hasattr(np, "float8_e5m2"):
    np.float8_e5m2 = ml_dtypes.float8_e5m2

# safetensors header stores tensor dtype as a string. Map to jax dtype.
# Kept in one place because multiple callers used to inline the same dict.
_SAFETENSORS_DTYPE_TO_JAX: dict[str, jnp.dtype] = {
    "BF16": jnp.bfloat16,
    "F16": jnp.float16,
    "F32": jnp.float32,
    "I64": jnp.int64,
    "I32": jnp.int32,
    "BOOL": jnp.bool_,
    "F8_E4M3": jnp.float8_e4m3fn,
    "F8_E5M2": jnp.float8_e5m2,
}


def _reinterpret_dtype_if_needed(data: np.ndarray, target_dtype: jnp.dtype) -> np.ndarray:
    if data.dtype == np.uint8:
        if target_dtype == jnp.float8_e4m3fn:
            return data.view(ml_dtypes.float8_e4m3fn)
        elif target_dtype == jnp.float8_e5m2:
            return data.view(ml_dtypes.float8_e5m2)
    elif data.dtype == np.dtype("V2"):
        return data.view(ml_dtypes.bfloat16)
    return data


def _validate_tensor_metadata(key, meta):
    dtype = _SAFETENSORS_DTYPE_TO_JAX.get(meta["dtype"])
    if dtype is None:
        raise ValueError(f"Unsupported safetensors dtype {meta['dtype']} for {key}")
    begin, end = meta["data_offsets"]
    shape = meta["shape"]
    if (
        any(not isinstance(d, int) or d < 0 for d in shape)
        or begin < 0
        or end - begin != math.prod(shape) * np.dtype(dtype).itemsize
    ):
        raise ValueError(f"Invalid safetensors metadata for {key}")


def _coordinator_metadata(scan):
    payload = None
    if jax.process_index() == 0:
        try:
            payload = (scan(), None)
        except Exception as exc:
            payload = (None, f"{type(exc).__name__}: {exc}")
    if jax.process_count() > 1:
        # Broadcast failures too, so peers do not wait forever for a header
        # scan that failed only on the coordinator. Paths remain rank-local.
        data = pickle.dumps(payload) if payload is not None else b""
        size = multihost_utils.broadcast_one_to_all(np.array(len(data), np.int32))
        buffer = (
            np.frombuffer(data, np.uint8) if payload is not None else np.empty(int(size), np.uint8)
        )
        payload = pickle.loads(np.asarray(multihost_utils.broadcast_one_to_all(buffer)).tobytes())
    result, error = payload
    if error is not None:
        raise RuntimeError(error)
    return result


def coordinate_error(error, phase):
    """Fixed phase boundary; never called from an I/O callback or worker thread."""
    if jax.process_count() > 1:
        failures = multihost_utils.process_allgather(np.array(error is not None, np.int32))
        if np.any(failures):
            ranks = np.flatnonzero(failures).tolist()
            raise RuntimeError(f"Weight {phase} failed on ranks {ranks}: {error}") from error
    if error is not None:
        raise error


class WeightSource(ABC):
    """One checkpoint session, independent of models and device placement.

    All ranks access metadata in the same order; implementations coordinate
    header discovery. Tensor slices use checkpoint dtypes and may borrow file
    storage until ``release``/``close``. Byte ranges return owned uint8 arrays
    in request order, safe across subsequent reads. Reads can run on threads;
    they must not perform collectives. The caller retains returned arrays until
    H2D completes and closes only sessions it owns.
    """

    prefers_bulk = False
    retains_views = False  # Drain H2D before releasing completed file mappings.

    @property
    @abstractmethod
    def metadata(self) -> dict[str, list[dict[str, Any]]]:
        """Tensor fragments with file, shape, dtype and absolute byte ranges."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Checkpoint identity for process-local parameter reuse."""

    @abstractmethod
    def read_tensor(self, filename: str, name: str, index) -> np.ndarray:
        """Read a tensor slice from one checkpoint fragment."""

    @abstractmethod
    def read_ranges(self, ranges: list[tuple[str, int, int]]) -> list[np.ndarray]:
        """Read (file, byte offset, byte count) requests into owned buffers."""

    def prefetch(self):
        """Optionally warm storage once per session, after plan validation."""
        return None

    def release(self, filenames):
        """Optionally release files after their last group and H2D completion."""
        return None

    @abstractmethod
    def close(self):
        """Release session resources; safe to call more than once."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class LocalSource(WeightSource):
    """
    Manages open file handles during a weight loading session to prevent
    repeated opening/parsing of safetensors headers.
    """

    def __init__(self, model_config=None, *, warmup=False):
        self.handles = {}
        self.model_config = model_config
        self._warmup = warmup
        self._prefetched = False
        self._weight_info_cache = None
        self._lock = threading.Lock()

    def _get_handle(self, filename):
        with self._lock:
            if filename not in self.handles:
                self.handles[filename] = safe_open(filename, framework="np", device="cpu")
            return self.handles[filename]

    retains_views = True

    def read_tensor(self, filename, name, index):
        tensor = self._get_handle(filename).get_slice(name)
        return _reinterpret_dtype_if_needed(
            tensor[index], _SAFETENSORS_DTYPE_TO_JAX[tensor.get_dtype()]
        )

    @property
    def identity(self):
        # P/D reuse is process-local. mtime/size/inode avoid a stale hit when a
        # local checkpoint is replaced without changing model_path.
        files = sorted({info["file"] for infos in self.metadata.values() for info in infos})
        signature = [
            (name, (stat := os.stat(name)).st_size, stat.st_mtime_ns, stat.st_ino) for name in files
        ]
        return hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    prefers_bulk = False

    def read_ranges(self, ranges):
        """Return owned byte arrays in request order, coalescing nearby ranges."""
        groups = {}
        for i, (filename, offset, size) in enumerate(ranges):
            groups.setdefault(filename, []).append((i, offset, size))
        output = [None] * len(ranges)

        def read_file(item):
            filename, requests = item
            requests.sort(key=lambda request: request[1])
            start = requests[0][1]
            end = max(offset + size for _, offset, size in requests)
            useful = sum(size for _, _, size in requests)
            with open(filename, "rb") as stream:
                if len(requests) > 1 and end - start <= 2 * useful:
                    stream.seek(start)
                    data = stream.read(end - start)
                    if len(data) != end - start:
                        raise ValueError(f"Truncated checkpoint range: {filename}")
                    for i, offset, size in requests:
                        output[i] = np.frombuffer(
                            data, np.uint8, count=size, offset=offset - start
                        ).copy()
                else:
                    for i, offset, size in requests:
                        stream.seek(offset)
                        data = stream.read(size)
                        if len(data) != size:
                            raise ValueError(f"Truncated checkpoint range: {filename}")
                        output[i] = np.frombuffer(data, np.uint8)

        if groups:
            workers = int(os.environ.get("SGLANG_MOE_LOAD_WORKERS", "16"))
            with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
                list(pool.map(read_file, groups.items()))
        return output

    def prefetch(self):
        """Pre-read safetensors files to warm GCSFuse cache."""
        if not self._warmup or self._prefetched:
            return
        self._prefetched = True
        model_path = self.model_config.model_path
        try:
            with open("/proc/mounts") as fp:
                mounts = [line.split() for line in fp]
            mount = max(
                (
                    mount
                    for mount in mounts
                    if model_path == mount[1] or model_path.startswith(mount[1].rstrip("/") + "/")
                ),
                key=lambda mount: len(mount[1]),
            )
            if "fuse" not in mount[2]:
                logger.info("model_path on %s mount, skipping GCSFuse warm-up", mount[2])
                return
        except Exception:
            logger.warning("Failed to detect model_path mount type; skipping GCSFuse warm-up")
            return

        st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not st_files:
            return

        total_size = sum(os.path.getsize(path) for path in st_files)
        logger.info(
            "Warming up GCSFuse cache: %d files, %.1f GB",
            len(st_files),
            total_size / 1024**3,
        )

        def _read_file(path):
            buf = bytearray(4 * 1024 * 1024)
            with open(path, "rb") as fp:
                while fp.readinto(buf):
                    pass

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=min(8, len(st_files))) as executor:
            list(executor.map(_read_file, st_files))
        t1 = time.time()
        logger.info(
            "GCSFuse cache warm-up done: %.1fs (%.0f MB/s)",
            t1 - t0,
            total_size / 1024**2 / (t1 - t0) if t1 > t0 else 0,
        )

    def close(self):
        # safe_open objects don't strictly require close() as they rely on RAII/GC,
        # but clearing references ensures we don't hold descriptors.
        self.handles.clear()

    def release(self, filenames):
        """Drop completed file mappings once their last loading group is ready."""
        with self._lock:
            for filename in filenames:
                self.handles.pop(filename, None)

    def _scan(self):
        root = self.model_config.model_path
        files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
        if not files:
            raise RuntimeError(f"Cannot find any *.safetensors files in {root}")
        result = {}
        for filename in files:
            # Header-only reads avoid faulting whole checkpoint files on GCSFuse.
            with open(filename, "rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > 100_000_000:
                    raise ValueError(f"Invalid safetensors header length: {filename}")
                header = json.loads(stream.read(size))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                _validate_tensor_metadata(name, meta)
                start, end = meta["data_offsets"]
                result.setdefault(name, []).append(
                    {
                        "file": os.path.basename(filename),
                        "shape": tuple(meta["shape"]),
                        "dtype": meta["dtype"],
                        "byte_offset": 8 + size + start,
                        "byte_size": end - start,
                    }
                )
        return result

    @property
    def metadata(self):
        if self._weight_info_cache is not None:
            return self._weight_info_cache
        result = _coordinator_metadata(self._scan)
        for infos in result.values():
            for info in infos:
                info["file"] = os.path.join(self.model_config.model_path, info["file"])
        self._weight_info_cache = result
        return result


_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 100_000_000


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
        dtype = np.dtype(_SAFETENSORS_DTYPE_TO_JAX[meta["dtype"]])
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


class RunaiWeightSource(WeightSource):
    """Safetensors-compatible reader with owned buffers and serialized SDK access."""

    prefers_bulk = True

    def __init__(self, path: str, metadata_dir: str):
        self.path = path
        self.metadata_dir = metadata_dir
        self.handles: dict[str, _File] = {}
        self._metadata: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._streamer = None

    @property
    def metadata(self):
        return self._metadata

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        from sgl_jax.srt.utils import runai_utils

        try:
            error = None
            try:
                sdk = runai_utils._sdk()
                self._chunks = sdk.FileChunks
                self._streamer = sdk.FileStreamer()
                self._streamer.__enter__()
            except Exception as exc:
                error = exc
            coordinate_error(error, "source initialization")
            # Only the coordinator lists/scans checkpoint headers. Broadcast
            # basenames, then bind files to each process's local or remote root.
            headers = _coordinator_metadata(self._scan)
            for basename, size, metadata in headers:
                path = self.path.rstrip("/") + "/" + basename
                for key, meta in metadata.items():
                    begin, end = meta["data_offsets"]
                    meta["offset"] = 8 + size + begin
                    self.metadata.setdefault(key, []).append(
                        {
                            "file": path,
                            "shape": tuple(meta["shape"]),
                            "dtype": meta["dtype"],
                            "byte_offset": meta["offset"],
                            "byte_size": end - begin,
                        }
                    )
                self.handles[path] = _File(self, path, metadata)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _scan(self):
        from sgl_jax.srt.utils import runai_utils

        paths = sorted(runai_utils._list_safetensors(self.path))
        index_path = Path(self.metadata_dir) / "model.safetensors.index.json"
        if index_path.exists():
            expected = set(json.loads(index_path.read_text())["weight_map"].values())
            available = {path.rsplit("/", 1)[-1]: path for path in paths}
            if expected - available.keys():
                raise ValueError(f"Missing checkpoint files: {sorted(expected - available.keys())}")
            paths = [available[name] for name in sorted(expected)]
        if not paths:
            raise ValueError(f"No safetensors files found at {self.path!r}")
        lengths = self.read_ranges([(path, 0, 8) for path in paths])
        sizes = [struct.unpack("<Q", value)[0] for value in lengths]
        if any(size <= 0 or size > _MAX_HEADER_BYTES for size in sizes):
            raise ValueError("Invalid safetensors header length")
        raw_headers = self.read_ranges([(p, 8, size) for p, size in zip(paths, sizes)])
        result = []
        for path, size, raw in zip(paths, sizes, raw_headers):
            metadata = json.loads(raw.tobytes())
            metadata.pop("__metadata__", None)
            for key, meta in metadata.items():
                _validate_tensor_metadata(key, meta)
            result.append((path.rsplit("/", 1)[-1], size, metadata))
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        if self._streamer is not None:
            self._streamer.__exit__(exc_type, exc_value, traceback)
            self._streamer = None

    @property
    def identity(self):
        from sgl_jax.srt.utils import runai_utils

        if not runai_utils.is_gcs_path(self.path):
            signature = [
                (name, (stat := os.stat(name)).st_size, stat.st_mtime_ns, stat.st_ino)
                for name in sorted(self.handles)
            ]
        else:
            signature = sorted(
                (blob.name, blob.generation, blob.size)
                for blob in runai_utils._gcs_blobs(self.path, delimiter="/")
                if blob.name.endswith(".safetensors")
            )
        return hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    def read_tensor(self, filename, name, index):
        return self.handles[filename].get_slice(name)[index]

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
                _CHUNK_BYTES,
                int(os.environ.get("RUNAI_STREAMER_MEMORY_LIMIT", _CHUNK_BYTES)),
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
