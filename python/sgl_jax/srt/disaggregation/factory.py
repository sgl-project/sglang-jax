"""Construct the selected PD transfer backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sgl_jax.srt.disaggregation.common.capacity import (
    CHUNK_TRANSFER_WINDOW,
    per_rank_inflight_limit,
)

if TYPE_CHECKING:
    from sgl_jax.srt.disaggregation.base.transfer import TransferBackend
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.server_args import ServerArgs


def _raiden_transfer_pool_shape(
    *,
    max_req_input_len: int,
    page_size: int,
    parent_slots: int,
    chunk_prefill_size: int | None,
    chunk_transfer_enabled: bool,
) -> tuple[int, int]:
    """Return ``(max_blocks_per_slot, slots)`` for one Raiden DP rank."""

    max_tokens_per_transfer = int(max_req_input_len)
    num_slots = int(parent_slots)
    if chunk_transfer_enabled:
        if chunk_prefill_size is None or chunk_prefill_size <= 0:
            raise ValueError("chunk transfer requires a positive chunk_prefill_size")
        max_tokens_per_transfer = min(max_tokens_per_transfer, int(chunk_prefill_size))
        # One wave can finish while the following chunk is registered. Raiden
        # allocates max_blocks worth of host staging for every slot, so size
        # each slot for one chunk rather than for the whole long prompt.
        num_slots *= CHUNK_TRANSFER_WINDOW
    max_blocks = (max_tokens_per_transfer + page_size - 1) // page_size
    return max(1, max_blocks), max(1, num_slots)


logger = logging.getLogger(__name__)


def create_transfer_backend(
    scheduler: Scheduler,
    server_args: ServerArgs,
    *,
    local_host: str,
    role: str,
    shared_secret: str | None,
    bootstrap_client: object,
) -> TransferBackend:
    if server_args.disaggregation_use_raiden:
        return _create_raiden_backend(
            scheduler,
            server_args,
            local_host=local_host,
            bootstrap_client=bootstrap_client,
        )
    return _create_jax_backend(
        scheduler,
        server_args,
        local_host=local_host,
        role=role,
        shared_secret=shared_secret,
    )


def _create_jax_backend(
    scheduler: Scheduler,
    server_args: ServerArgs,
    *,
    local_host: str,
    role: str,
    shared_secret: str | None,
):
    from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
    from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
    from sgl_jax.srt.disaggregation.jax_transfer.wrapper import get_or_create_wrapper

    wrapper = get_or_create_wrapper(
        local_host,
        server_args.disaggregation_transfer_port,
        channel_number=server_args.disaggregation_channel_number,
    )
    wrapper.start()
    notifier = ZmqPullNotifier(
        role,
        local_host,
        server_args.disaggregation_side_channel_port,
        shared_secret=shared_secret,
    )
    notifier.start()

    host_pool = None
    if server_args.disaggregation_enable_d2h and role == "prefill":
        from sgl_jax.srt.disaggregation.prefill import (
            _KV_GATHER_PAGE_BUCKETS,
            _pad_to_page_bucket,
        )
        from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool

        kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        max_tokens = server_args.disaggregation_d2h_max_tokens
        if max_tokens is None:
            max_tokens = scheduler.max_total_num_tokens
        max_pages = (max_tokens + server_args.page_size - 1) // server_args.page_size
        max_padded_pages = max(
            _pad_to_page_bucket(max_pages),
            _KV_GATHER_PAGE_BUCKETS[-1],
        )
        host_pool = QueueHostKVPool(
            pool_size=server_args.disaggregation_d2h_pool_size,
            max_padded_pages=max_padded_pages,
            layer_num=kv_pool.layer_num,
            per_layer_shape=tuple(int(d) for d in kv_pool.kv_buffer[0].shape[1:]),
            dtype=kv_pool.dtype,
            mesh=kv_pool.mesh,
            partition_spec=kv_pool.kv_sharding.spec,
            pool_name="pd_prefill",
        )

    return JaxTransferKVManager(
        wrapper,
        notifier,
        host_pool=host_pool,
        use_d2h_staging=server_args.disaggregation_enable_d2h and role == "prefill",
        ack_timeout_seconds=server_args.disaggregation_ack_timeout_seconds,
        pull_timeout_seconds=server_args.disaggregation_pull_timeout_seconds,
        reaper_interval_seconds=server_args.disaggregation_orphan_reaper_interval_seconds,
        pull_worker_count=server_args.disaggregation_channel_number,
    )


def _tree_cache_supports_swa(tree_cache: object) -> bool:
    """Probe optional SWA support without requiring every cache to implement it."""
    supports_swa = getattr(tree_cache, "supports_swa", None)
    return bool(supports_swa()) if callable(supports_swa) else False


def _create_raiden_backend(
    scheduler: Scheduler,
    server_args: ServerArgs,
    *,
    local_host: str,
    bootstrap_client: object,
):
    if server_args.device != "tpu":
        raise ValueError("Raiden requires device=tpu")
    if server_args.disaggregation_enable_d2h:
        raise ValueError("Raiden and D2H staging select different transfer engines")
    if server_args.disaggregation_max_inflight_transfers <= 0:
        raise ValueError("Raiden requires max_inflight_transfers > 0")
    if not getattr(server_args, "disable_radix_cache", False):
        raise ValueError("Raiden requires --disable-radix-cache")
    chunk_transfer_enabled = bool(
        getattr(server_args, "disaggregation_enable_chunk_prefill_transfer", False)
    )
    if (
        chunk_transfer_enabled
        and getattr(scheduler, "tree_cache", None) is not None
        and _tree_cache_supports_swa(scheduler.tree_cache)
    ):
        raise ValueError("Raiden chunk prefill transfer does not support sliding-window KV")
    if bootstrap_client is None or not hasattr(bootstrap_client, "require_capability"):
        raise RuntimeError("Raiden requires a bootstrap client with capability probing")
    bootstrap_client.require_capability("transfer_metadata")
    if chunk_transfer_enabled:
        bootstrap_client.require_capability("chunk_transfer_metadata")

    from sgl_jax.raiden import require_raiden_preloaded

    require_raiden_preloaded()

    from sgl_jax.srt.disaggregation.raiden_transfer.conn import RaidenTransferKVManager
    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import (
        get_or_create_raiden_wrapper,
    )

    kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    wrapper = get_or_create_raiden_wrapper(
        local_host,
        0,
        parallelism=server_args.disaggregation_channel_number,
    )
    page_size = max(1, int(server_args.page_size))
    parent_slots = per_rank_inflight_limit(
        server_args.disaggregation_max_inflight_transfers,
        server_args.dp_size,
    )
    max_blocks, num_slots = _raiden_transfer_pool_shape(
        max_req_input_len=int(scheduler.max_req_input_len),
        page_size=page_size,
        parent_slots=parent_slots,
        chunk_prefill_size=getattr(server_args, "chunked_prefill_size", None),
        chunk_transfer_enabled=chunk_transfer_enabled,
    )
    wrapper.start(
        kv_caches=list(kv_pool.kv_buffer),
        max_blocks=max_blocks,
        num_slots=num_slots,
        dp_size=server_args.dp_size,
        timeout_s=float(server_args.disaggregation_pull_timeout_seconds),
    )
    logger.info(
        "Raiden backend ready: max_blocks=%d slots_per_rank=%d layers=%d "
        "dp_size=%d chunk_transfer=%s",
        max_blocks,
        num_slots,
        kv_pool.layer_num,
        server_args.dp_size,
        chunk_transfer_enabled,
    )
    return RaidenTransferKVManager(
        wrapper,
        bootstrap_client,
        enable_chunk_prefill_transfer=getattr(
            server_args,
            "disaggregation_enable_chunk_prefill_transfer",
            False,
        ),
        ack_timeout_seconds=server_args.disaggregation_ack_timeout_seconds,
        pull_timeout_seconds=server_args.disaggregation_pull_timeout_seconds,
        reaper_interval_seconds=server_args.disaggregation_orphan_reaper_interval_seconds,
    )
