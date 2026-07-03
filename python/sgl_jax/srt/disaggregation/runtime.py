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
    from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
    from sgl_jax.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeTransferQueue,
    )
    from sgl_jax.srt.disaggregation.decode_watchdog import EventLoopWatchdog
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip
    from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
    from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
        get_or_create_raiden_wrapper,
        get_or_create_wrapper,
    )
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
    if (
        server_args.disaggregation_enable_d2h
        and mode == "prefill"
        and not server_args.disaggregation_use_raiden
    ):
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
        max_padded_pages = max(_pad_to_page_bucket(max_request_pages), _KV_GATHER_PAGE_BUCKETS[-1])
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

    scheduler.disagg_bootstrap_client = BootstrapClient(
        server_args.disaggregation_bootstrap_url,
        timeout_s=server_args.disaggregation_bootstrap_timeout_seconds,
        shared_secret=shared_secret,
    )

    # raiden data plane (Phase 0). When enabled, construct the process-level
    # RaidenTransferWrapper over the device KV pool's per-layer tensors and hand
    # it (plus the bootstrap client, used for the P->D per-request block-metadata
    # channel) to the manager. path-A stays fully wired so it can be selected by
    # leaving --disaggregation-use-raiden off for A/B baselining.
    raiden_wrapper = None
    if server_args.disaggregation_use_raiden:
        kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        raiden_wrapper = get_or_create_raiden_wrapper(
            local_host,
            server_args.disaggregation_raiden_control_port,
            parallelism=server_args.disaggregation_channel_number,
        )
        # raiden staging-slot sizing. ``max_blocks`` must fit the largest
        # per-request page count (a single slot holds one request's pages);
        # ``num_slots`` bounds concurrent in-flight receives on decode. Passing
        # ``host_blocks_to_allocate`` instead routes to raiden's legacy ctor
        # overload that leaves the slot pool unconfigured (max_blocks_=0) so
        # every start_read fails the size guard immediately. Add headroom over
        # max_seq_len/page_size: a max-length prompt spills into one extra page
        # once BOS / chat-template tokens push it past the page boundary
        # (observed 33 pages for a 4096-token input at page_size 128).
        page_size = max(1, int(server_args.page_size))
        max_blocks = (int(server_args.max_seq_len) + page_size - 1) // page_size + 8
        num_slots = max(16, int(server_args.disaggregation_max_inflight_transfers) * 2)
        # For hybrid SWA models the pool is a SWAKVPool with separate full and SWA
        # sub-pools. Pass both so raiden creates two KVCacheManagers (one per pool);
        # raiden broadcasts block_ids to all layers within an engine, so per-pool
        # engines are needed when full and SWA layers have different page numbering.
        kv_caches_swa = None
        if hasattr(kv_pool, "full_kv_pool"):
            # SWAKVPool: expose sub-pool buffers directly (kv_pool.kv_buffer is not
            # defined — get_kv_buffer dispatches per-layer).
            kv_caches_full = list(kv_pool.full_kv_pool.kv_buffer)
            kv_caches_swa = list(kv_pool.swa_kv_pool.kv_buffer)
        else:
            kv_caches_full = list(kv_pool.kv_buffer)
        raiden_wrapper.start(
            kv_caches=kv_caches_full,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=float(server_args.disaggregation_pull_timeout_seconds),
            kv_caches_swa=kv_caches_swa,
        )
        logger.info(
            "raiden data plane enabled: control_port=%s max_blocks=%d "
            "num_slots=%d layer_num=%d is_hybrid_swa=%s swa_layer_num=%d",
            server_args.disaggregation_raiden_control_port,
            max_blocks,
            num_slots,
            len(kv_caches_full),
            kv_caches_swa is not None,
            len(kv_caches_swa) if kv_caches_swa else 0,
        )

    scheduler.disagg_kv_manager = JaxTransferKVManager(
        wrapper,
        notifier,
        host_pool=host_pool,
        raiden_wrapper=raiden_wrapper,
        bootstrap_client=scheduler.disagg_bootstrap_client,
        ack_timeout_seconds=server_args.disaggregation_ack_timeout_seconds,
        pull_timeout_seconds=server_args.disaggregation_pull_timeout_seconds,
        reaper_interval_seconds=(server_args.disaggregation_orphan_reaper_interval_seconds),
        pull_worker_count=server_args.disaggregation_channel_number,
    )
    scheduler.disagg_kv_manager.start_reaper()
    scheduler.disagg_use_d2h_staging = server_args.disaggregation_enable_d2h

    if mode == "prefill":
        import jax

        scheduler.disagg_prefill_queue = PrefillBootstrapQueue()
        prefill_kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        kv_dtype_name = resolve_kv_dtype_name(prefill_kv_pool.dtype)

        # raiden: advertise the control endpoint so D can reach P's TransferEngine.
        raiden_control_port = 0
        raiden_endpoints_json = ""
        if raiden_wrapper is not None:
            import json as _json

            raiden_control_port = server_args.disaggregation_raiden_control_port
            endpoints = raiden_wrapper.endpoints
            raiden_endpoints_json = (
                _json.dumps(endpoints) if endpoints is not None else ""
            )

        bootstrap_keys = []
        for system_dp_rank in range(server_args.dp_size):
            bootstrap_key = f"{local_host}:{transfer_port}:dp_{system_dp_rank}"
            bootstrap_keys.append(bootstrap_key)
            scheduler.disagg_bootstrap_client.register_prefill(
                bootstrap_key=bootstrap_key,
                host=local_host,
                transfer_port=transfer_port,
                side_channel_port=side_channel_port,
                tp_rank=server_args.node_rank,
                tp_size=server_args.tp_size,
                system_dp_rank=system_dp_rank,
                jax_process_index=jax.process_index(),
                jax_process_count=jax.process_count(),
                page_size=server_args.page_size,
                kv_dtype=kv_dtype_name,
                local_control_port=raiden_control_port,
                raiden_endpoints_json=raiden_endpoints_json,
            )

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
            for key in (
                getattr(scheduler, "disagg_bootstrap_keys", None)
                or [scheduler.disagg_bootstrap_key]
            ):
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
        with suppress(Exception):
            scheduler.disagg_kv_manager.zmq_notifier.stop()
        if scheduler.disagg_decode_watchdog is not None:
            with suppress(Exception):
                scheduler.disagg_decode_watchdog.stop()

    return _shutdown
