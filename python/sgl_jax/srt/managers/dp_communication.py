"""Data Parallel communication utilities using JAX distributed primitives."""

import logging
import pickle
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import broadcast_one_to_all

logger = logging.getLogger(__name__)

# Global storage for inter-DP-group communication
_dp_group_request_storage = {}


class DPGroupCommunicator:
    """
    Handle communication within a DP group using JAX distributed primitives.

    Unlike the previous ZMQ approach, this uses JAX's built-in collective operations
    which are more efficient and integrate better with JAX's distributed runtime.
    """

    def __init__(self, dp_group_info: dict):
        """
        Initialize DP group communicator.

        Args:
            dp_group_info: Dictionary containing:
                - dp_group_id: Which DP group this scheduler belongs to
                - rank_in_group: Rank within the DP group
                - group_size: Number of schedulers in this DP group
                - is_publisher: Whether this scheduler should broadcast requests
        """
        self.dp_group_info = dp_group_info
        self.dp_group_id = dp_group_info["dp_group_id"]
        self.rank_in_group = dp_group_info["rank_in_group"]
        self.group_size = dp_group_info["group_size"]
        self.is_publisher = dp_group_info["is_publisher"]

        logger.info(
            f"DPGroupCommunicator initialized: group={self.dp_group_id}, "
            f"rank={self.rank_in_group}/{self.group_size}, "
            f"is_publisher={self.is_publisher}"
        )

    def broadcast_requests(self, requests: List[Any]) -> List[Any]:
        """
        Broadcast requests within the DP group using JAX collective operations.

        This replaces the ZMQ pub/sub mechanism with JAX's broadcast_one_to_all,
        which is more efficient and doesn't require manual address management.

        Args:
            requests: List of requests to broadcast (None for subscribers)

        Returns:
            List of requests (broadcasted from publisher to all subscribers)
        """
        try:
            if self.group_size == 1:
                # Single scheduler in group, no need to broadcast
                return requests

            # Serialize requests for broadcasting
            if self.is_publisher:
                # Publisher: serialize the requests
                if requests is None:
                    requests = []
                serialized_data = pickle.dumps(requests)
                # Convert to JAX array for broadcasting
                data_bytes = jnp.array(list(serialized_data), dtype=jnp.uint8)
                data_size = jnp.array(len(serialized_data), dtype=jnp.int32)
            else:
                # Subscribers: prepare empty data
                data_bytes = jnp.array([0], dtype=jnp.uint8)
                data_size = jnp.array(0, dtype=jnp.int32)

            # Broadcast the data size first
            with jax.default_device(jax.local_devices()[0]):
                broadcasted_size = broadcast_one_to_all(data_size).item()

                if broadcasted_size > 0:
                    # Resize subscriber arrays to match the actual data size
                    if not self.is_publisher:
                        data_bytes = jnp.zeros(broadcasted_size, dtype=jnp.uint8)

                    # Broadcast the actual data
                    broadcasted_data = broadcast_one_to_all(data_bytes)

                    # Deserialize the broadcasted data
                    serialized_bytes = bytes(broadcasted_data.tolist())
                    requests = pickle.loads(serialized_bytes)
                else:
                    requests = []

            logger.debug(
                f"DP group {self.dp_group_id}: {'published' if self.is_publisher else 'received'} "
                f"{len(requests)} requests"
            )

            return requests

        except Exception as e:
            logger.error(f"DP group {self.dp_group_id} broadcast failed: {e}")
            # Fallback: return original requests or empty list
            return requests if requests is not None else []

    def is_communication_needed(self) -> bool:
        """Check if communication is needed in this DP group."""
        return self.group_size > 1

    def get_group_info(self) -> dict:
        """Get DP group information."""
        return self.dp_group_info.copy()

    def broadcast_inter_group_requests(
        self, dp_group_requests: Dict[int, List[Any]]
    ) -> Dict[int, List[Any]]:
        """
        Broadcast requests from node 0 to all DP group leaders.

        This handles the first tier of communication: Node 0 → DP group leaders

        Args:
            dp_group_requests: Dict mapping group_id -> requests for that group

        Returns:
            The same dict, but broadcasted to all nodes
        """
        try:
            # Serialize the entire request distribution map
            if dp_group_requests:
                serialized_data = pickle.dumps(dp_group_requests)
                data_bytes = jnp.array(list(serialized_data), dtype=jnp.uint8)
                data_size = jnp.array(len(serialized_data), dtype=jnp.int32)
            else:
                data_bytes = jnp.array([0], dtype=jnp.uint8)
                data_size = jnp.array(0, dtype=jnp.int32)

            # Global broadcast from node 0 to all nodes
            with jax.default_device(jax.local_devices()[0]):
                broadcasted_size = broadcast_one_to_all(data_size).item()

                if broadcasted_size > 0:
                    # Resize arrays for non-node-0 processes
                    if jax.process_index() != 0:
                        data_bytes = jnp.zeros(broadcasted_size, dtype=jnp.uint8)

                    # Broadcast the actual data
                    broadcasted_data = broadcast_one_to_all(data_bytes)

                    # Deserialize the broadcasted data
                    serialized_bytes = bytes(broadcasted_data.tolist())
                    result = pickle.loads(serialized_bytes)
                else:
                    result = {}

            logger.debug(f"Inter-group broadcast: {len(result)} groups distributed")
            return result

        except Exception as e:
            logger.error(f"Inter-group broadcast failed: {e}")
            return dp_group_requests if dp_group_requests else {}

    def get_requests_for_my_group(
        self, all_requests: Dict[int, List[Any]]
    ) -> List[Any]:
        """
        Extract requests assigned to this scheduler's DP group.

        Args:
            all_requests: Dict mapping group_id -> requests for that group

        Returns:
            List of requests for this scheduler's DP group
        """
        return all_requests.get(self.dp_group_id, [])


def create_dp_communicator(server_args) -> Optional[DPGroupCommunicator]:
    """
    Create a DP group communicator based on server arguments.

    Args:
        server_args: Server configuration

    Returns:
        DPGroupCommunicator instance or None if DP is disabled
    """
    if not server_args.enable_dp_attention or server_args.dp_size == 1:
        return None

    # Import here to avoid circular dependency
    from sgl_jax.srt.layers.dp_attention import compute_dp_attention_world_info

    attn_tp_size, attn_dp_rank, dp_rank = compute_dp_attention_world_info(
        server_args.enable_dp_attention,
        server_args.node_rank,  # tp_rank = scheduler's global rank
        server_args.tp_size,  # total schedulers
        server_args.dp_size,  # number of DP groups
    )

    dp_group_info = {
        "dp_group_id": attn_dp_rank,
        "rank_in_group": attn_dp_rank,
        "group_size": attn_tp_size,
        "is_publisher": (attn_dp_rank == 0),
    }

    return DPGroupCommunicator(dp_group_info)
