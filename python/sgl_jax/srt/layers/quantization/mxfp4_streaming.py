"""Stream K3's MXFP4 experts from GCS, fetching only the bytes this rank needs.

Kimi-K3 is 1.42 TiB across 96 shards. sglang-jax's loader wants every shard present on local
disk, and a ``tpu7x-standard-4t`` node has ~919 GB of tmpfs -- so the released checkpoint does not
fit on a host *before* any sharding. That is the blocker for the full 93-layer model; HBM is not
(experts kept in fp4 are 1,357 GiB, which fits 16 chips).

Two reductions, the same two the vllm-torchtpu lane uses to serve this model today
(``model_loader_patches.py::_sharded_runai_weights_iterator``):

**Stream.** safetensors is a header plus a flat data block, and every tensor's byte range is in
that header. So the shard never has to land on disk: read the header, then issue ranged GETs for
exactly the tensors wanted. Peak local footprint is one tensor, not one shard and not the model.

**Filter.** Under expert parallelism a rank owns ``num_experts / ep_size`` experts -- 28 of 896 at
``ep_size=32``. The other 868 experts' ``weight_packed``/``weight_scale`` are never requested, so
the saving is in bytes *moved*, not merely bytes retained. Everything that is not a per-expert
tensor (dense layers, shared experts, the router, norms) is read by every rank.

The vllm lane's filter and its FusedMoE expert map are derived from the same computation so they
cannot disagree; :func:`local_expert_ids` here plays that role, and the caller is expected to use
it for both the fetch plan and the device placement.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

# safetensors dtype string -> numpy. Only what K3 actually ships is listed; an unknown dtype
# should fail loudly rather than be guessed at, since a wrong width silently misreads every
# following tensor in the range.
_DTYPES = {
    "U8": np.uint8,
    "I8": np.int8,
    "BF16": None,  # handled specially: numpy has no bfloat16
    "F16": np.float16,
    "F32": np.float32,
    "I32": np.int32,
    "I64": np.int64,
    "BOOL": np.bool_,
}

_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")

# Suffixes of PER-EXPERT tensors that may be skipped for non-local experts. Anything else --
# dense layers, shared experts, the gate, norms -- is needed by every rank. Mirrors
# vllm_torchtpu's _EXPERT_WEIGHT_SUFFIXES; widening this list silently drops shared state.
EXPERT_SUFFIXES = (".weight", ".weight_packed", ".weight_scale")


class TensorSpan(NamedTuple):
    """Where one tensor lives inside a shard, and how to interpret it."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in the file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def parse_expert_id(name: str) -> int | None:
    """The expert index in a tensor name, or None if it is not per-expert."""
    m = _EXPERT_RE.search(name)
    return int(m.group(1)) if m else None


def local_expert_ids(num_experts: int, ep_size: int, ep_rank: int) -> set[int]:
    """Contiguous-block expert assignment: rank r owns ``[r*n/ep, (r+1)*n/ep)``.

    Must match whatever the model uses to place experts on devices. If the two disagree a rank
    fetches experts it will not use and lacks ones it will -- and the resulting model still loads,
    because the missing slots are simply zeros.
    """
    if num_experts % ep_size:
        raise ValueError(f"num_experts {num_experts} not divisible by ep_size {ep_size}")
    per = num_experts // ep_size
    return set(range(ep_rank * per, (ep_rank + 1) * per))


def should_skip(name: str, local_ids: set[int] | None) -> bool:
    """True when this tensor belongs to an expert this rank does not own."""
    if local_ids is None:
        return False
    expert_id = parse_expert_id(name)
    if expert_id is None:
        return False
    if not name.endswith(EXPERT_SUFFIXES):
        return False
    return expert_id not in local_ids


# ----------------------------------------------------------------------------------------------
# GCS-backed shard reading
# ----------------------------------------------------------------------------------------------
class ShardReader:
    """Ranged reads against one safetensors shard in GCS.

    The header carries every tensor's byte range, so after one small read the shard is fully
    addressable without downloading it.
    """

    def __init__(self, bucket: str, name: str, client=None):
        from google.cloud import storage

        self._blob = (client or storage.Client()).bucket(bucket).blob(name)
        self.name = name
        self._spans: dict[str, TensorSpan] | None = None

    @property
    def spans(self) -> dict[str, TensorSpan]:
        if self._spans is None:
            self._spans = self._read_header()
        return self._spans

    def _read_header(self) -> dict[str, TensorSpan]:
        # 8-byte little-endian header length, then that many bytes of JSON, then the data block.
        raw_len = self._blob.download_as_bytes(start=0, end=7)
        (header_len,) = struct.unpack("<Q", raw_len)
        header = json.loads(self._blob.download_as_bytes(start=8, end=8 + header_len - 1))
        data_start = 8 + header_len

        spans: dict[str, TensorSpan] = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            spans[key] = TensorSpan(
                name=key,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                start=data_start + lo,
                end=data_start + hi,
            )
        return spans

    def read_range(self, lo: int, hi: int) -> bytes:
        """One ranged GET covering ``[lo, hi)`` -- the unit a coalesced run fetches."""
        return self._blob.download_as_bytes(start=lo, end=hi - 1)

    def read(self, span: TensorSpan) -> np.ndarray:
        """Fetch exactly one tensor's bytes and view them as its declared dtype/shape."""
        if span.nbytes == 0:
            return np.zeros(span.shape, dtype=_np_dtype(span.dtype))
        raw = self._blob.download_as_bytes(start=span.start, end=span.end - 1)
        return _view(raw, span)


def _np_dtype(dtype: str):
    if dtype not in _DTYPES:
        raise ValueError(f"unhandled safetensors dtype {dtype!r}")
    if dtype == "BF16":
        # numpy has no bfloat16; keep the raw pair-of-bytes and let the caller reinterpret via
        # jnp/ml_dtypes. Returning float32 here would silently double every value's width.
        return np.uint16
    return _DTYPES[dtype]


def _view(raw: bytes, span: TensorSpan) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=_np_dtype(span.dtype))
    return arr.reshape(span.shape)


# ----------------------------------------------------------------------------------------------
# Fetch planning
# ----------------------------------------------------------------------------------------------
class FetchPlan(NamedTuple):
    """What a rank will read, and what it will not."""

    spans: list[TensorSpan]
    kept_bytes: int
    skipped_tensors: int
    skipped_bytes: int

    def summary(self) -> str:
        total = self.kept_bytes + self.skipped_bytes
        pct = (100.0 * self.kept_bytes / total) if total else 100.0
        return (
            f"{len(self.spans)} tensors / {self.kept_bytes / 1e9:.1f} GB kept; "
            f"{self.skipped_tensors} tensors / {self.skipped_bytes / 1e9:.1f} GB skipped "
            f"({pct:.1f}% of shard bytes fetched)"
        )


def plan_fetch(
    reader: ShardReader,
    local_ids: set[int] | None = None,
    want: "callable | None" = None,
) -> FetchPlan:
    """Decide which of a shard's tensors this rank reads.

    ``want`` is an optional extra predicate on the tensor name, so a caller that only needs, say,
    the MoE experts does not pay for the rest of the shard either.
    """
    spans, kept_bytes, skipped_n, skipped_bytes = [], 0, 0, 0
    for span in reader.spans.values():
        if should_skip(span.name, local_ids) or (want is not None and not want(span.name)):
            skipped_n += 1
            skipped_bytes += span.nbytes
            continue
        spans.append(span)
        kept_bytes += span.nbytes
    spans.sort(key=lambda s: s.start)  # sequential order: friendlier to object storage
    return FetchPlan(spans, kept_bytes, skipped_n, skipped_bytes)


def stream_shard(
    reader: ShardReader,
    local_ids: set[int] | None = None,
    want: "callable | None" = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, array)`` for the tensors this rank keeps, one at a time.

    One tensor is resident at a time, so peak local footprint is a single tensor (~16 MB for a K3
    expert projection) rather than a shard (~15 GB) or the model (1.42 TiB). The caller is
    expected to move each array onto a device and drop it before requesting the next.
    """
    plan = plan_fetch(reader, local_ids, want)
    for span in plan.spans:
        yield span.name, reader.read(span)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """``gs://bucket/prefix`` -> ``(bucket, prefix)``; prefix has no leading or trailing slash."""
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri!r}")
    bucket, _, prefix = uri[len("gs://"):].partition("/")
    return bucket, prefix.strip("/")


def list_shards(bucket: str, prefix: str, client=None) -> list[str]:
    """Every ``*.safetensors`` object under a prefix, in name order."""
    from google.cloud import storage

    client = client or storage.Client()
    names = [
        b.name
        for b in client.list_blobs(bucket, prefix=f"{prefix}/")
        if b.name.endswith(".safetensors")
    ]
    return sorted(names)


# ----------------------------------------------------------------------------------------------
# Tensor sources -- one interface over a local directory and a gs:// prefix
# ----------------------------------------------------------------------------------------------
class LocalSource:
    """safetensors files already on disk."""

    def __init__(self, files: list[str], keep: "callable | None" = None):
        from safetensors import safe_open

        self._handles = {f: safe_open(f, framework="np") for f in files}
        self._where: dict[str, str] = {}
        for path, handle in self._handles.items():
            for key in handle.keys():
                if keep is None or keep(key):
                    self._where[key] = path

    def has(self, key: str) -> bool:
        return key in self._where

    def prefetch(self, keys: "list[str]") -> None:
        """No-op: local reads are mmap'd, so there is no round trip to hide."""
        return None

    def get(self, key: str) -> np.ndarray:
        return self._handles[self._where[key]].get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass


def _size_http_pool(client, workers: int) -> None:
    """Widen the storage client's connection pool to match the fetch-thread count.

    Silent no-op if the client does not expose a mountable session -- a smaller pool is a
    performance problem, not a correctness one, and is not worth failing the load over.
    """
    try:
        import requests

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max(workers, 16),
            pool_maxsize=max(workers, 16),
            max_retries=3,
        )
        client._http.mount("https://", adapter)
        client._http.mount("http://", adapter)
    except Exception:  # noqa: BLE001
        pass


class GcsSource:
    """safetensors shards in GCS, read by byte range and never staged.

    Building this reads one header per shard (96 small GETs for the K3 release), after which every
    tensor is addressable. No shard is downloaded whole at any point.

    **Requests are the cost, not bytes.** K3 has 896 experts x 3 projections x 2 tensors per MoE
    layer, so one GET per tensor is ~5.4k requests for a 4-layer truncation and ~165k for the full
    93 -- at even 20 ms of round-trip each that is most of an hour of pure latency. Two fixes, both
    of which the vllm-torchtpu lane also applies:

    * **prefetch in parallel** -- a small thread pool hides round-trip latency; GCS is happy with
      concurrent ranged reads and the bottleneck becomes bandwidth rather than RTT;
    * **read ahead in shard order** -- callers ask for tensors grouped by (layer, projection),
      which is close to byte order within a shard, so prefetching the plan in offset order keeps
      the access pattern sequential.
    """

    def __init__(self, uri: str, keep: "callable | None" = None, client=None, workers: int = 0):
        import os
        from concurrent.futures import ThreadPoolExecutor

        from google.cloud import storage

        # Latency-bound, not CPU-bound: each worker spends its time waiting on a ranged GET, so
        # the useful count is far above core count. Measured at 16 workers the 4L expert load ran
        # 5,376 tensors in 245 s; the full model asks ~124k tensors per host at ep_size=32, so an
        # under-sized pool is an hour of avoidable waiting.
        workers = workers or int(os.environ.get("KIMI_K3_FETCH_WORKERS", "64"))

        client = client or storage.Client()
        # The client's default HTTP pool holds 10 connections. Running 64 fetch threads against
        # it makes urllib3 discard and re-establish constantly ("Connection pool is full,
        # discarding connection: storage.googleapis.com"), which is both slow and destabilising --
        # a 4-host run died silently right after starting the fetch. Size the pool to the fleet.
        _size_http_pool(client, workers)
        bucket, prefix = parse_gs_uri(uri)
        self._workers = workers
        self._readers: dict[str, ShardReader] = {}
        self._where: dict[str, ShardReader] = {}

        shards = list_shards(bucket, prefix, client=client)
        readers = [ShardReader(bucket, name, client=client) for name in shards]
        # header reads are independent; doing 96 of them serially is ~96 round trips of nothing
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda r: r.spans, readers))
        for reader in readers:
            self._readers[reader.name] = reader
            for key in reader.spans:
                if keep is None or keep(key):
                    self._where[key] = reader
        self._cache: dict[str, np.ndarray] = {}

    def has(self, key: str) -> bool:
        return key in self._where

    # Bridge gaps up to this size rather than splitting a run: re-fetching a few unwanted MB in
    # one sequential read is far cheaper than the round trip a split costs.
    COALESCE_GAP = 8 << 20

    def prefetch(self, keys: "list[str]") -> None:
        """Fetch these tensors into the cache, coalescing adjacent ranges into single GETs.

        Requests dominate: K3 asks 1,792 tensors per (layer, projection) group, and one GET each
        measured **46 s per group** on a 4-host run -- about 3.5 h for the model. But a shard lays
        consecutive experts out contiguously, so those thousands of small ranges collapse into a
        handful of large sequential reads. This is what the vllm-torchtpu lane does with the
        Run:AI streamer's ``FileChunks`` runs.

        Call with one group at a time. Fetching the whole model here would defeat the point --
        the cache would become the local copy this class exists to avoid.
        """
        from concurrent.futures import ThreadPoolExecutor

        pending = [k for k in keys if k in self._where and k not in self._cache]
        if not pending:
            return

        # group by shard, then build runs of near-adjacent spans in byte order
        by_shard: dict[str, list[str]] = {}
        for key in pending:
            by_shard.setdefault(self._where[key].name, []).append(key)

        runs: list[tuple[ShardReader, list[str]]] = []
        for shard, shard_keys in by_shard.items():
            reader = self._where[shard_keys[0]]
            shard_keys.sort(key=lambda k: reader.spans[k].start)
            current: list[str] = []
            end = None
            for key in shard_keys:
                span = reader.spans[key]
                if current and span.start - end <= self.COALESCE_GAP:
                    current.append(key)
                else:
                    if current:
                        runs.append((reader, current))
                    current = [key]
                end = span.end
                end = max(end, span.end)
            if current:
                runs.append((reader, current))

        def _read_run(item):
            reader, run_keys = item
            lo = min(reader.spans[k].start for k in run_keys)
            hi = max(reader.spans[k].end for k in run_keys)
            blob = reader.read_range(lo, hi)
            out = []
            for key in run_keys:
                span = reader.spans[key]
                out.append((key, _view(blob[span.start - lo : span.end - lo], span)))
            return out

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            for chunk in pool.map(_read_run, runs):
                for key, arr in chunk:
                    self._cache[key] = arr

    def get(self, key: str) -> np.ndarray:
        cached = self._cache.pop(key, None)  # pop: single-use, so the cache cannot grow into a copy
        if cached is not None:
            return cached
        reader = self._where[key]
        return reader.read(reader.spans[key])

    def close(self) -> None:
        self._cache.clear()


def open_source(path: str, keep: "callable | None" = None):
    """A LocalSource or GcsSource depending on what ``path`` is."""
    if str(path).startswith("gs://"):
        return GcsSource(path, keep=keep)
    import glob as _glob
    import os as _os

    files = sorted(_glob.glob(_os.path.join(path, "*.safetensors")))
    return LocalSource(files, keep=keep)


# ----------------------------------------------------------------------------------------------
# Per-device assembly -- what makes the fetch filter reduce HOST work too
# ----------------------------------------------------------------------------------------------
def build_sharded_expert_param(global_shape, sharding, fetch_expert):
    """Assemble a globally-sharded ``[E, ...]`` array without materializing it on the host.

    ``jax.device_put(global_array, sharding)`` requires the whole array on the host first, which
    for K3 is the entire expert tensor set -- the thing the EP filter exists to avoid. Building
    each device's slice and handing them to ``make_array_from_single_device_arrays`` bounds host
    residency to ONE device's experts, and lets ``fetch_expert`` skip the rest entirely.

    ``fetch_expert(e)`` returns expert ``e``'s array; it is called only for experts some local
    device owns, so it is where the fetch saving is realized.
    """
    import jax
    import numpy as _np

    index_map = sharding.addressable_devices_indices_map(tuple(global_shape))
    shards = []
    for device, index in index_map.items():
        expert_slice = index[0] if isinstance(index, tuple) else index
        expert_ids = range(*expert_slice.indices(global_shape[0])) if isinstance(
            expert_slice, slice
        ) else [int(expert_slice)]
        local = _np.stack([_np.asarray(fetch_expert(e)) for e in expert_ids], axis=0)
        # honour any non-expert slicing this device also carries (e.g. a tensor-parallel split)
        rest = index[1:] if isinstance(index, tuple) else ()
        if rest:
            local = local[(slice(None), *rest)]
        shards.append(jax.device_put(local, device))
    return jax.make_array_from_single_device_arrays(tuple(global_shape), sharding, shards)
