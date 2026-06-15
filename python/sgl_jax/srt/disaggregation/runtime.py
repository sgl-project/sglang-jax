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
    if server_args.dp_size > 1:
        raise RuntimeError(
            f"PD disaggregation does not yet support dp_size>1 "
            f"(got dp_size={server_args.dp_size}). This will be "
            f"supported in a future PR."
        )
    if server_args.disaggregation_bootstrap_url is None:
        raise RuntimeError("disaggregation_mode != null requires bootstrap_url")

    from sgl_jax.srt.disaggregation.bootstrap import (
        BootstrapClient,
        BootstrapServer,
        HeartbeatDaemon,
        PrefillInfoCache,
        resolve_kv_dtype_name,
    )
    from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
    from sgl_jax.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeTransferQueue,
    )
    from sgl_jax.srt.disaggregation.decode_watchdog import EventLoopWatchdog
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
    from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
    from sgl_jax.srt.disaggregation.jax_transfer.wrapper import get_or_create_wrapper
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

    wrapper = get_or_create_wrapper(
        local_host,
        transfer_port,
        channel_number=server_args.disaggregation_channel_number,
    )
    wrapper.start()
    notifier = ZmqPullNotifier(
        role,
        local_host,
        side_channel_port,
        shared_secret=shared_secret,
    )
    notifier.start()
    host_pool = None
    if server_args.disaggregation_enable_d2h and mode == "prefill":
        from sgl_jax.srt.disaggregation.prefill import (
            _KV_GATHER_PAGE_BUCKETS,
            _pad_to_page_bucket,
        )
        from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool

        kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        per_layer_shape = tuple(int(d) for d in kv_pool.kv_buffer[0].shape[1:])
        # Size each host buffer to the largest single-request KV that can be
        # staged. A buffer must hold one request's padded pages; the cap is the
        # D2H max-token setting when given, else the device KV budget (no single
        # request can exceed max_total_num_tokens). Round up via the same bucket
        # logic as gather so copy_from_device never rejects an admissible req.
        d2h_max_tokens = server_args.disaggregation_d2h_max_tokens
        if d2h_max_tokens is None:
            d2h_max_tokens = scheduler.max_total_num_tokens
        page_size = server_args.page_size
        max_request_pages = (d2h_max_tokens + page_size - 1) // page_size
        max_padded_pages = max(
            _pad_to_page_bucket(max_request_pages), _KV_GATHER_PAGE_BUCKETS[-1]
        )
        host_pool = QueueHostKVPool(
            pool_size=server_args.disaggregation_d2h_pool_size,
            max_padded_pages=max_padded_pages,
            layer_num=kv_pool.layer_num,
            per_layer_shape=per_layer_shape,
            dtype=kv_pool.dtype,
            mesh=kv_pool.mesh,
            partition_spec=kv_pool.kv_sharding.spec,
            pool_name="pd_prefill",
        )
        logger.info(
            "D2H host pool wired: pool_size=%d max_padded_pages=%d layer_num=%d "
            "per_layer_shape=%s",
            server_args.disaggregation_d2h_pool_size,
            max_padded_pages,
            kv_pool.layer_num,
            per_layer_shape,
        )

    scheduler.disagg_kv_manager = JaxTransferKVManager(
        wrapper,
        notifier,
        host_pool=host_pool,
        ack_timeout_seconds=server_args.disaggregation_ack_timeout_seconds,
        pull_timeout_seconds=server_args.disaggregation_pull_timeout_seconds,
        reaper_interval_seconds=(server_args.disaggregation_orphan_reaper_interval_seconds),
        pull_worker_count=server_args.disaggregation_channel_number,
    )
    scheduler.disagg_kv_manager.start_reaper()
    scheduler.disagg_use_d2h_staging = server_args.disaggregation_enable_d2h

    scheduler.disagg_bootstrap_client = BootstrapClient(
        server_args.disaggregation_bootstrap_url,
        timeout_s=server_args.disaggregation_bootstrap_timeout_seconds,
        shared_secret=shared_secret,
    )

    if mode == "prefill":
        import jax

        scheduler.disagg_prefill_queue = PrefillBootstrapQueue()
        bootstrap_key = f"{local_host}:{transfer_port}"
        prefill_kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        scheduler.disagg_bootstrap_client.register_prefill(
            bootstrap_key=bootstrap_key,
            host=local_host,
            transfer_port=transfer_port,
            side_channel_port=side_channel_port,
            tp_rank=server_args.node_rank,
            tp_size=server_args.tp_size,
            system_dp_rank=0,
            jax_process_index=jax.process_index(),
            jax_process_count=jax.process_count(),
            page_size=server_args.page_size,
            kv_dtype=resolve_kv_dtype_name(prefill_kv_pool.dtype),
        )
        scheduler.disagg_heartbeat = HeartbeatDaemon(
            scheduler.disagg_bootstrap_client, bootstrap_key
        )
        scheduler.disagg_heartbeat.start()
        scheduler.disagg_bootstrap_key = bootstrap_key
    else:
        scheduler.disagg_prefill_info_cache = PrefillInfoCache(
            scheduler.disagg_bootstrap_client
        )
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
            try:
                key = scheduler.disagg_bootstrap_key
                scheduler.disagg_bootstrap_client.unregister_prefill(key)
            except Exception:
                logger.warning(
                    "PD shutdown: unregister_prefill failed",
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
        with suppress(Exception):
            scheduler.disagg_kv_manager.zmq_notifier.stop()
        if scheduler.disagg_decode_watchdog is not None:
            with suppress(Exception):
                scheduler.disagg_decode_watchdog.stop()

    return _shutdown
