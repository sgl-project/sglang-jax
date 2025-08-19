"""Data parallel attention utilities for sgl-jax."""

from typing import Optional, Tuple


def compute_dp_attention_world_info(
    enable_dp_attention: bool,
    node_rank: int,
    tp_size: int,
    dp_size: int,
    node_tp_size: Optional[int] = 4,
) -> Tuple[int, int]:
    # dp rank, attention tp size
    attn_tp_size = tp_size // dp_size
    if not enable_dp_attention:
        return attn_tp_size, 0
    dp_rank = (node_rank * node_tp_size) // attn_tp_size
    attn_dp_rank = node_rank - dp_rank * (attn_tp_size // node_tp_size)
    return attn_tp_size, attn_dp_rank, dp_rank


def should_run_publisher(server_args) -> bool:
    """
    Determine if current scheduler should run publisher or subscriber.

    In DP architecture:
    - Multiple DP groups, each group has multiple schedulers
    - Only the rank 0 scheduler within each DP group should run publisher
    - Other schedulers in the same group run subscriber

    Returns:
        True if should run publisher (rank 0 in DP group), False if should run subscriber
    """
    if not server_args.enable_dp_attention or server_args.dp_size == 1:
        # Use original logic: node_rank == 0 runs publisher
        return server_args.node_rank == 0

    # With DP attention enabled:
    # Use tp_size to represent total number of schedulers across all nodes/groups
    # Use node_rank as this scheduler's global rank
    tp_size = server_args.tp_size  # Total number of schedulers
    tp_rank = server_args.node_rank  # This scheduler's global rank

    attn_tp_size, attn_dp_rank, dp_rank = compute_dp_attention_world_info(
        server_args.enable_dp_attention,
        tp_rank,  # This scheduler's global rank
        tp_size,  # Total number of schedulers
        server_args.dp_size,  # Number of DP groups
    )

    # Only the scheduler with attn_dp_rank == 0 in each DP group runs publisher
    return attn_dp_rank == 0
