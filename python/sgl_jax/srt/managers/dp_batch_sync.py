"""DP Attention batch synchronization utilities for JAX."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import broadcast_one_to_all

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


@dataclass
class DPBatchSyncInfo:
    """Information needed for DP batch synchronization."""

    dp_group_id: int
    batch_size: int
    max_seq_len: int
    has_requests: bool
    need_idle_batch: bool


class DPBatchSynchronizer:
    """
    Handles batch synchronization across DP groups for attention computation.

    In DP attention, all groups need to have compatible batch shapes for efficient
    collective operations. This class ensures:
    1. All groups know about each other's batch dimensions
    2. Idle batches are generated when needed to maintain shape consistency
    3. Load balancing information is exchanged periodically
    """

    def __init__(self, server_args, dp_group_info: dict):
        """
        Initialize DP batch synchronizer.

        Args:
            server_args: Server configuration
            dp_group_info: DP group information for this scheduler
        """
        self.server_args = server_args
        self.dp_group_info = dp_group_info
        self.dp_group_id = dp_group_info["dp_group_id"]
        self.dp_size = server_args.dp_size
        self.enable_dp_attention = server_args.enable_dp_attention

        # Batch sync state
        self.forward_ct = 0
        self.last_sync_info = {}
        self._needs_idle_batch = False

        logger.info(f"DPBatchSynchronizer initialized for group {self.dp_group_id}")

    def prepare_dp_sync_batch(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """
        Prepare batch for DP attention computation.

        This is the JAX equivalent of sglang's prepare_mlp_sync_batch.
        It ensures all DP groups have compatible batch shapes.

        Args:
            batch: The current batch (can be None)

        Returns:
            Synchronized batch (may be idle batch if needed)
        """
        if not self.enable_dp_attention or self.dp_size <= 1:
            # No DP attention, return original batch
            return batch

        try:
            # Step 1: Collect local batch information
            local_sync_info = self._get_local_batch_info(batch)

            # Step 2: Exchange batch info across all DP groups
            all_sync_info = self._exchange_batch_sync_info(local_sync_info)

            # Step 3: Determine if we need idle batch for synchronization
            need_idle_batch = self._should_generate_idle_batch(
                local_sync_info, all_sync_info
            )

            # Step 4: Signal need for idle batch if required
            if need_idle_batch and batch is None:
                logger.debug(
                    f"DP group {self.dp_group_id}: Signaling need for idle batch"
                )
                # Return a special marker to indicate idle batch is needed
                # The scheduler will handle creating the actual idle batch
                self._needs_idle_batch = True
                return None
            else:
                self._needs_idle_batch = False

            # Step 5: Handle load balancing (periodic)
            if self.forward_ct % 40 == 0:
                self._handle_dp_load_balancing(all_sync_info)

            self.forward_ct += 1
            return batch

        except Exception as e:
            logger.error(f"DP batch sync failed for group {self.dp_group_id}: {e}")
            # Fallback: return original batch
            return batch

    def _get_local_batch_info(self, batch: Optional[ScheduleBatch]) -> DPBatchSyncInfo:
        """Get local batch information for synchronization."""
        if batch is None or batch.is_empty():
            return DPBatchSyncInfo(
                dp_group_id=self.dp_group_id,
                batch_size=0,
                max_seq_len=0,
                has_requests=False,
                need_idle_batch=False,
            )

        return DPBatchSyncInfo(
            dp_group_id=self.dp_group_id,
            batch_size=batch.batch_size,
            max_seq_len=(
                max(req.extend_input_len for req in batch.reqs) if batch.reqs else 0
            ),
            has_requests=True,
            need_idle_batch=False,
        )

    def _exchange_batch_sync_info(
        self, local_info: DPBatchSyncInfo
    ) -> Dict[int, DPBatchSyncInfo]:
        """
        Exchange batch sync information across all DP groups using jax.lax.all_gather.

        This provides true all-gather semantics where each process contributes its data
        and all processes receive the complete gathered data.
        """
        try:
            if jax.process_count() <= 1 or self.dp_size <= 1:
                # Single process or single DP group
                return {self.dp_group_id: local_info}

            # Create a JAX function that uses all_gather
            # This ensures we're in the right context for collective operations
            @jax.jit
            def gather_batch_info(local_data):
                """JAX function to perform all_gather on batch sync info."""
                try:
                    # Method 1: Try using 'data' axis (most appropriate for DP groups)
                    return jax.lax.all_gather(local_data, axis_name="data")
                except (ValueError, NameError, KeyError):
                    try:
                        # Method 2: Try using 'tensor' axis
                        return jax.lax.all_gather(local_data, axis_name="tensor")
                    except (ValueError, NameError, KeyError):
                        # If named axes fail, return just the local data reshaped
                        # This will trigger the fallback logic
                        return jnp.expand_dims(local_data, axis=0)

            with jax.default_device(jax.local_devices()[0]):
                # Encode local batch sync info as a fixed-size array
                # Format: [dp_group_id, batch_size, max_seq_len, has_requests]
                local_data = jnp.array(
                    [
                        local_info.dp_group_id,
                        local_info.batch_size,
                        local_info.max_seq_len,
                        int(local_info.has_requests),
                    ],
                    dtype=jnp.int32,
                )

                try:
                    # Use the JAX-compiled function for all_gather
                    all_data = gather_batch_info(local_data)

                    # Check if we got real all_gather results or fallback
                    if all_data.shape[0] == 1:
                        # Only got our own data back, use fallback
                        logger.debug(
                            "all_gather returned single process data, using fallback"
                        )
                        return self._fallback_exchange_batch_info(local_info)

                    logger.debug(f"all_gather successful, shape: {all_data.shape}")

                except Exception as e:
                    logger.warning(f"JAX all_gather failed: {e}, using fallback")
                    return self._fallback_exchange_batch_info(local_info)

                # Parse the gathered data
                # all_data shape: [num_processes, 4] where each row is one process's data
                result = {}
                num_processes = all_data.shape[0]

                # Initialize all groups with default values
                for group_id in range(self.dp_size):
                    result[group_id] = DPBatchSyncInfo(
                        dp_group_id=group_id,
                        batch_size=0,
                        max_seq_len=0,
                        has_requests=False,
                        need_idle_batch=False,
                    )

                # Update with actual data from each process
                for process_idx in range(num_processes):
                    process_data = all_data[process_idx]
                    group_id = int(process_data[0])

                    # Validate group_id
                    if 0 <= group_id < self.dp_size:
                        result[group_id] = DPBatchSyncInfo(
                            dp_group_id=group_id,
                            batch_size=int(process_data[1]),
                            max_seq_len=int(process_data[2]),
                            has_requests=bool(process_data[3]),
                            need_idle_batch=False,
                        )

                logger.debug(
                    f"DP batch sync: Gathered info for {len(result)} groups from {num_processes} processes"
                )
                return result

        except Exception as e:
            logger.error(f"all_gather batch sync failed: {e}")
            return self._fallback_exchange_batch_info(local_info)

    def _fallback_exchange_batch_info(
        self, local_info: DPBatchSyncInfo
    ) -> Dict[int, DPBatchSyncInfo]:
        """
        Fallback method when all_gather is not available.
        Uses conservative assumptions based on local information.
        """
        logger.debug("Using fallback batch sync (conservative assumptions)")

        result = {self.dp_group_id: local_info}

        # For other groups, make conservative assumptions
        for group_id in range(self.dp_size):
            if group_id != local_info.dp_group_id:
                # Conservative assumption: if we have requests, others might too
                # This ensures idle batches are generated when needed
                result[group_id] = DPBatchSyncInfo(
                    dp_group_id=group_id,
                    batch_size=local_info.batch_size if local_info.has_requests else 0,
                    max_seq_len=(
                        local_info.max_seq_len if local_info.has_requests else 0
                    ),
                    has_requests=local_info.has_requests,  # Conservative: assume same as us
                    need_idle_batch=False,
                )

        return result

    def _should_generate_idle_batch(
        self, local_info: DPBatchSyncInfo, all_info: Dict[int, DPBatchSyncInfo]
    ) -> bool:
        """
        Determine if we need to generate an idle batch for synchronization.

        Idle batch is needed when:
        1. This group has no requests but other groups do
        2. All groups need to participate in collective attention operations
        """
        if local_info.has_requests:
            # We have requests, no idle batch needed
            return False

        # Check if any other group has requests
        any_group_has_requests = any(info.has_requests for info in all_info.values())

        if any_group_has_requests:
            logger.debug(
                f"DP group {self.dp_group_id}: Need idle batch (others have requests)"
            )
            return True

        return False

    def _generate_idle_batch_for_dp_sync(
        self, all_info: Dict[int, DPBatchSyncInfo]
    ) -> Optional[ScheduleBatch]:
        """
        Generate an idle batch with appropriate dimensions for DP sync.

        The idle batch should have compatible shapes with other groups' batches
        for attention computation.

        Returns:
            None to indicate that scheduler should use its own get_idle_batch() method
        """
        # Find the maximum dimensions across all groups
        max_batch_size = max((info.batch_size for info in all_info.values()), default=0)
        max_seq_len = max((info.max_seq_len for info in all_info.values()), default=0)

        logger.debug(
            f"DP group {self.dp_group_id}: Need idle batch for sync "
            f"(other groups: batch_size={max_batch_size}, seq_len={max_seq_len})"
        )

        # Return None to signal that the scheduler should use its existing idle batch mechanism
        # This allows us to reuse the existing get_idle_batch() method which has access
        # to all the necessary components (req_to_token_pool, tree_cache, etc.)
        return None

    def _handle_dp_load_balancing(self, all_info: Dict[int, DPBatchSyncInfo]):
        """
        Handle periodic load balancing across DP groups.

        This is the JAX equivalent of sglang's handle_dp_balance_data.
        """
        if not all_info:
            return

        # Calculate load distribution
        total_requests = sum(info.batch_size for info in all_info.values())
        group_loads = {group_id: info.batch_size for group_id, info in all_info.items()}

        if total_requests > 0:
            # Calculate load balance metrics
            avg_load = total_requests / self.dp_size
            max_load = max(group_loads.values())
            min_load = min(group_loads.values())
            load_imbalance = (max_load - min_load) / max(avg_load, 1)

            logger.debug(
                f"DP load balance check: avg={avg_load:.1f}, "
                f"max={max_load}, min={min_load}, imbalance={load_imbalance:.3f}"
            )

            # Log warning if severely imbalanced
            if load_imbalance > 0.5:  # 50% imbalance threshold
                logger.warning(f"DP groups are imbalanced: {group_loads}")

    def is_dp_sync_enabled(self) -> bool:
        """Check if DP synchronization is enabled and needed."""
        return self.enable_dp_attention and self.dp_size > 1

    def needs_idle_batch(self) -> bool:

        return getattr(self, "_needs_idle_batch", False)


def create_dp_batch_synchronizer(
    server_args, dp_group_info: dict
) -> Optional[DPBatchSynchronizer]:
    """
    Create a DP batch synchronizer if DP attention is enabled.

    Args:
        server_args: Server configuration
        dp_group_info: DP group information

    Returns:
        DPBatchSynchronizer instance or None if DP is disabled
    """
    if not server_args.enable_dp_attention or server_args.dp_size <= 1:
        return None

    return DPBatchSynchronizer(server_args, dp_group_info)
