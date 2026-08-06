"""Runtime wiring for scheduler PD disaggregation mode."""

from __future__ import annotations

import logging
import os
import signal
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def install_disaggregation_wiring(scheduler: Scheduler, server_args: ServerArgs) -> None:
    """Wire up PD runtime attributes when disaggregation mode is enabled."""

    mode = server_args.disaggregation_mode
    if mode == "null":
        return
    if server_args.dp_size > 1 and not server_args.disaggregation_use_raiden:
        raise RuntimeError(
            "PD disaggregation with dp_size>1 requires the Raiden transfer "
            f"engine (got dp_size={server_args.dp_size})."
        )
    if server_args.disaggregation_bootstrap_url is None:
        raise RuntimeError("disaggregation_mode != null requires bootstrap_url")

    import jax

    if jax.process_count() > 1 and server_args.disaggregation_enable_d2h:
        raise RuntimeError(
            "PD D2H host staging (--disaggregation-enable-d2h) is single-host "
            "only. The host KV pool is built on the global kv_pool mesh, but "
            "multi-host prefill extracts a local-mesh shard, so copy_from_device "
            "would reshard-fail. Run multi-host without d2h (path B: direct HBM "
            "transfer)."
        )

    from sgl_jax.srt.disaggregation.bootstrap import (
        BootstrapClient,
        BootstrapServer,
        HeartbeatDaemon,
        PrefillInfoCache,
        resolve_kv_dtype_name,
    )
    from sgl_jax.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeTransferQueue,
    )
    from sgl_jax.srt.disaggregation.decode_watchdog import EventLoopWatchdog
    from sgl_jax.srt.disaggregation.factory import create_transfer_backend
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
    from sgl_jax.srt.disaggregation.pd_auth import resolve_secret
    from sgl_jax.srt.disaggregation.prefill import PrefillBootstrapQueue

    local_host = resolve_host_ip(server_args.disaggregation_host_ip)
    transfer_port = server_args.disaggregation_transfer_port
    side_channel_port = server_args.disaggregation_side_channel_port
    role = "prefill" if mode == "prefill" else "decode"
    shared_secret = resolve_secret(server_args.disaggregation_shared_secret)

    if os.environ.get("DISAGG_LAUNCH_BOOTSTRAP", "") == "1":
        scheduler.disagg_bootstrap_server = BootstrapServer(
            host=local_host,
            port=server_args.disaggregation_bootstrap_port,
            shared_secret=shared_secret,
        )
        scheduler.disagg_bootstrap_server.start()
        logger.info(
            "embedded BootstrapServer started at %s:%d",
            local_host,
            server_args.disaggregation_bootstrap_port,
        )

    scheduler.disagg_bootstrap_client = BootstrapClient(
        server_args.disaggregation_bootstrap_url,
        timeout_s=server_args.disaggregation_bootstrap_timeout_seconds,
        shared_secret=shared_secret,
    )

    scheduler.disagg_kv_manager = create_transfer_backend(
        scheduler,
        server_args,
        local_host=local_host,
        role=role,
        shared_secret=shared_secret,
        bootstrap_client=scheduler.disagg_bootstrap_client,
    )
    scheduler.disagg_kv_manager.start_reaper()
    scheduler.disagg_use_d2h_staging = scheduler.disagg_kv_manager.requires_host_staging

    if mode == "prefill":
        import jax

        scheduler.disagg_prefill_queue = PrefillBootstrapQueue()
        prefill_kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        kv_dtype_name = resolve_kv_dtype_name(prefill_kv_pool.dtype)

        bootstrap_keys = []
        for dp_rank in range(server_args.dp_size):
            bootstrap_key = f"{local_host}:{transfer_port}:dp_{dp_rank}"
            scheduler.disagg_bootstrap_client.register_prefill(
                bootstrap_key=bootstrap_key,
                host=local_host,
                transfer_port=transfer_port,
                side_channel_port=side_channel_port,
                tp_rank=server_args.node_rank,
                tp_size=server_args.tp_size // server_args.dp_size,
                system_dp_rank=dp_rank,
                jax_process_index=jax.process_index(),
                jax_process_count=jax.process_count(),
                page_size=server_args.page_size,
                kv_dtype=kv_dtype_name,
                transport_metadata=(
                    scheduler.disagg_kv_manager.prefill_transport_metadata(dp_rank)
                ),
            )
            bootstrap_keys.append(bootstrap_key)
        scheduler.disagg_heartbeat = HeartbeatDaemon(
            scheduler.disagg_bootstrap_client, bootstrap_keys
        )
        scheduler.disagg_heartbeat.start()
        scheduler.disagg_bootstrap_keys = bootstrap_keys
        scheduler.disagg_bootstrap_key = bootstrap_keys[0]
    else:
        scheduler.disagg_prefill_info_cache = PrefillInfoCache(scheduler.disagg_bootstrap_client)
        scheduler.disagg_prealloc_queue = DecodePreallocQueue()
        scheduler.disagg_transfer_queue = DecodeTransferQueue()
        scheduler.disagg_decode_watchdog = EventLoopWatchdog(
            stall_threshold_s=server_args.disaggregation_decode_watchdog_seconds,
            snapshot_provider=scheduler._decode_backlog_snapshot,
        )

    scheduler.disagg_shutdown = _make_disagg_shutdown(scheduler, mode)
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def _handler(_signum, _frame, _prev=previous):
            try:
                scheduler.disagg_shutdown()
            finally:
                if callable(_prev) and _prev not in (
                    signal.SIG_DFL,
                    signal.SIG_IGN,
                ):
                    _prev(_signum, _frame)

        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, RuntimeError):
        logger.info(
            "PD graceful shutdown handler skipped; call "
            "scheduler.disagg_shutdown() from the main thread."
        )


def _make_disagg_shutdown(scheduler: Scheduler, mode: str):
    """Create an idempotent graceful-shutdown closure."""

    state = {"done": False}

    def _shutdown():
        if state["done"]:
            return
        state["done"] = True
        if mode == "prefill":
            keys = getattr(
                scheduler,
                "disagg_bootstrap_keys",
                [scheduler.disagg_bootstrap_key],
            )
            for key in keys:
                try:
                    scheduler.disagg_bootstrap_client.unregister_prefill(key)
                except Exception:
                    logger.warning(
                        "PD shutdown: unregister_prefill failed for %s",
                        key,
                        exc_info=True,
                    )
            with suppress(Exception):
                scheduler.disagg_heartbeat.stop()
        try:
            scheduler.disagg_kv_manager.graceful_shutdown(drain_timeout_seconds=30.0)
        except Exception:
            logger.warning(
                "PD shutdown: manager.graceful_shutdown failed",
                exc_info=True,
            )
        notifier = getattr(scheduler.disagg_kv_manager, "zmq_notifier", None)
        if notifier is not None:
            with suppress(Exception):
                notifier.stop()
        if scheduler.disagg_decode_watchdog is not None:
            with suppress(Exception):
                scheduler.disagg_decode_watchdog.stop()

    return _shutdown
