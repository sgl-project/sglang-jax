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
        self._engine: Any | None = None
        self._endpoints: list[Any] | None = None

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
        return self._engine is not None

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def endpoints(self) -> list[Any] | None:
        return self._endpoints

    def start(
        self,
        kv_caches: list[Any],
        *,
        max_blocks: int,
        num_slots: int,
        timeout_s: float = 120.0,
    ) -> Any:
        if self._engine is not None:
            return self._engine
        with self._init_lock:
            if self._engine is not None:
                return self._engine
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

            self._engine = KVCacheManager(
                kv_caches=list(kv_caches),
                local_control_port=self._control_port,
                max_blocks=int(max_blocks),
                num_slots=int(num_slots),
                timeout_s=float(timeout_s),
                parallelism=self._parallelism,
                unsafe_skip_buffer_lock=True,
            )
            self._endpoints = self._engine.get_local_endpoints()
            logger.info(
                "Raiden started at %s with endpoints=%s (jax=%s)",
                self._control_port,
                self._endpoints,
                jax.__version__,
            )
        return self._engine

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> bool:
        if self._engine is None:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        return bool(self._engine.register_read(req_id, uuid, list(block_ids)))

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: Any,
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        if self._engine is None:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        self._engine.start_read(
            req_id,
            uuid,
            remote_endpoint,
            list(remote_block_ids),
            list(local_block_ids),
            self._parallelism,
        )

    def poll_stats(self) -> tuple[list[str], list[str], list[str]]:
        if self._engine is None:
            raise RuntimeError("RaidenTransferWrapper.start() must be called first")
        return self._engine.poll_stats()


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
