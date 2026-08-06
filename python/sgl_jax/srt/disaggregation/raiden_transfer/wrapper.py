"""Thin process-level wrapper over tpu-raiden's JAX KV cache manager."""

from __future__ import annotations

import logging
import threading
from typing import Any

import jax

logger = logging.getLogger(__name__)

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_WRAPPER: RaidenTransferWrapper | None = None


class RaidenTransferWrapper:
    def __init__(
        self,
        host_ip: str,
        control_port: int = 0,
        *,
        parallelism: int = 1,
    ) -> None:
        self._host_ip = host_ip
        self._control_port = control_port
        self._parallelism = max(1, int(parallelism))
        self._init_lock = threading.Lock()
        self._engines: dict[int, Any] = {}
        self._endpoints_by_dp_rank: dict[int, list[Any]] = {}

    @property
    def host_ip(self) -> str:
        return self._host_ip

    @property
    def control_port(self) -> int:
        return self._control_port

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @property
    def is_started(self) -> bool:
        return bool(self._engines)

    @property
    def engine(self) -> Any:
        return self._engines.get(0)

    @property
    def endpoints(self) -> list[Any] | None:
        return self._endpoints_by_dp_rank.get(0)

    @property
    def endpoints_by_dp_rank(self) -> dict[int, list[Any]]:
        return {rank: list(endpoints) for rank, endpoints in self._endpoints_by_dp_rank.items()}

    @property
    def dp_size(self) -> int:
        return len(self._engines)

    def start(
        self,
        kv_caches: list[Any],
        *,
        max_blocks: int,
        num_slots: int,
        dp_size: int = 1,
        timeout_s: float = 120.0,
    ) -> Any:
        if self._engines:
            return self.engine
        with self._init_lock:
            if self._engines:
                return self.engine
            try:
                from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Raiden is enabled but tpu_raiden is not installed; install "
                    "a wheel matching the active JAX and libtpu versions"
                ) from exc

            if not kv_caches:
                raise ValueError("Raiden requires at least one KV cache tensor")
            if max_blocks <= 0 or num_slots <= 0:
                raise ValueError("Raiden max_blocks and num_slots must be positive")
            if dp_size <= 0:
                raise ValueError("Raiden dp_size must be positive")

            caches_by_rank = _split_kv_caches_by_dp_rank(kv_caches, int(dp_size))
            for dp_rank, rank_caches in caches_by_rank.items():
                control_port = 0 if self._control_port == 0 else self._control_port + dp_rank
                if control_port > 65535:
                    raise ValueError(
                        f"Raiden control port overflows for dp_rank={dp_rank}: {control_port}"
                    )
                engine = KVCacheManager(
                    kv_caches=rank_caches,
                    local_control_port=control_port,
                    max_blocks=int(max_blocks),
                    num_slots=int(num_slots),
                    timeout_s=float(timeout_s),
                    parallelism=self._parallelism,
                    # A manager-lifetime PJRT hold would block serving compute.
                    unsafe_skip_buffer_lock=True,
                )
                endpoints = list(engine.get_local_endpoints())
                self._engines[dp_rank] = engine
                self._endpoints_by_dp_rank[dp_rank] = endpoints
                logger.info(
                    "Raiden started for dp_rank=%d at requested_port=%s with endpoints=%s (jax=%s)",
                    dp_rank,
                    control_port,
                    endpoints,
                    jax.__version__,
                )
        return self.engine

    def _engine_for_rank(self, dp_rank: int) -> Any:
        try:
            return self._engines[int(dp_rank)]
        except KeyError as exc:
            raise ValueError(
                f"Raiden dp_rank={dp_rank} is unavailable; started ranks={sorted(self._engines)}"
            ) from exc

    def register_read(
        self, req_id: str, uuid: int, block_ids: list[int], *, dp_rank: int = 0
    ) -> bool:
        if not self._engines:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        return bool(self._engine_for_rank(dp_rank).register_read(req_id, uuid, list(block_ids)))

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: Any,
        remote_block_ids: list[int],
        local_block_ids: list[int],
        *,
        decode_dp_rank: int = 0,
    ) -> None:
        if not self._engines:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        self._engine_for_rank(decode_dp_rank).start_read(
            req_id,
            uuid,
            remote_endpoint,
            list(remote_block_ids),
            list(local_block_ids),
            self._parallelism,
        )

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        if not self._engines:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        sent: list[str] = []
        received: list[str] = []
        failed: list[str] = []
        for dp_rank, engine in self._engines.items():
            try:
                rank_sent, rank_received, rank_failed = engine.poll_stats()
            except Exception:
                # Preserve events already drained from healthy ranks.
                logger.exception("Raiden poll_stats failed for dp_rank=%d", dp_rank)
                continue
            sent.extend(rank_sent)
            received.extend(rank_received)
            failed.extend(rank_failed)
        return sent, received, failed


def _partition_without_data_axis(partition: object) -> object:
    if partition == "data":
        return None
    if isinstance(partition, tuple) and "data" in partition:
        remaining = tuple(axis for axis in partition if axis != "data")
        if not remaining:
            return None
        return remaining[0] if len(remaining) == 1 else remaining
    return partition


def _rank_local_array(array: Any, dp_rank: int, dp_size: int) -> Any:
    """Expose one data shard without changing physical buffer ownership."""

    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    sharding = getattr(array, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError(
            f"Raiden DP requires NamedSharding KV arrays; got {type(sharding).__name__}"
        )
    mesh = sharding.mesh
    axis_names = tuple(mesh.axis_names)
    if "data" not in axis_names:
        raise ValueError(f"Raiden DP requires a data mesh axis, got {axis_names}")
    mesh_dp_size = int(mesh.shape["data"])
    if mesh_dp_size != dp_size:
        raise ValueError(
            f"Raiden DP topology mismatch: configured dp_size={dp_size}, "
            f"KV mesh data size={mesh_dp_size}"
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} is outside [0, {dp_size})")

    original_spec = tuple(sharding.spec)
    data_sharded_dims = []
    for dim, partition in enumerate(original_spec):
        axes = partition if isinstance(partition, tuple) else (partition,)
        if "data" in axes:
            data_sharded_dims.append(dim)
    if not data_sharded_dims:
        raise ValueError(
            "Raiden DP requires the KV PartitionSpec to shard a tensor "
            f"dimension along the data mesh axis; got spec={sharding.spec}"
        )

    data_axis = axis_names.index("data")
    rank_devices = np.take(np.asarray(mesh.devices), dp_rank, axis=data_axis)
    rank_axis_names = tuple(axis for axis in axis_names if axis != "data")
    if not rank_axis_names:
        rank_devices = np.asarray(rank_devices).reshape(1)
        rank_axis_names = ("_raiden",)
    rank_mesh = Mesh(rank_devices, rank_axis_names)

    rank_spec = tuple(_partition_without_data_axis(partition) for partition in original_spec)
    rank_sharding = NamedSharding(rank_mesh, PartitionSpec(*rank_spec))

    rank_shape = list(array.shape)
    for dim, partition in enumerate(original_spec):
        axes = partition if isinstance(partition, tuple) else (partition,)
        if "data" in axes:
            if rank_shape[dim] % dp_size:
                raise ValueError(
                    f"KV dimension {dim} size {rank_shape[dim]} is not divisible "
                    f"by dp_size={dp_size}"
                )
            rank_shape[dim] //= dp_size
    rank_shape = tuple(rank_shape)

    shard_by_device = {shard.device: shard.data for shard in array.addressable_shards}
    device_indices = rank_sharding.addressable_devices_indices_map(rank_shape)
    missing = [device for device in device_indices if device not in shard_by_device]
    if missing:
        raise RuntimeError(
            f"Raiden dp_rank={dp_rank} is not fully addressable on process "
            f"{jax.process_index()}; missing devices={missing}"
        )
    return jax.make_array_from_single_device_arrays(
        rank_shape,
        rank_sharding,
        [shard_by_device[device] for device in device_indices],
    )


def _split_kv_caches_by_dp_rank(kv_caches: list[Any], dp_size: int) -> dict[int, list[Any]]:
    if dp_size == 1:
        return {0: list(kv_caches)}
    return {
        dp_rank: [_rank_local_array(cache, dp_rank, dp_size) for cache in kv_caches]
        for dp_rank in range(dp_size)
    }


def get_or_create_raiden_wrapper(
    host_ip: str,
    control_port: int = 0,
    *,
    parallelism: int = 1,
) -> RaidenTransferWrapper:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        if _GLOBAL_WRAPPER is None:
            _GLOBAL_WRAPPER = RaidenTransferWrapper(
                host_ip,
                control_port,
                parallelism=parallelism,
            )
            return _GLOBAL_WRAPPER
        existing = _GLOBAL_WRAPPER
        if (existing.host_ip, existing.control_port) != (host_ip, control_port):
            raise RuntimeError("RaidenTransferWrapper is already bound to another endpoint")
        if existing.parallelism != parallelism:
            raise RuntimeError("RaidenTransferWrapper parallelism cannot change after creation")
        return existing


def _reset_raiden_singleton_for_test() -> None:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        _GLOBAL_WRAPPER = None
