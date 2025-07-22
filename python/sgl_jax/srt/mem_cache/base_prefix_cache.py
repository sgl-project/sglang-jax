import abc
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.radix_cache import TreeNode
else:
    TreeNode = Any  # Placeholder for TreeNode type when not type checking


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Length of the KV cache hit on the host, if applicable.
                            0 if HiCache is not enabled.
    """

    device_indices: jnp.ndarray
    last_device_node: Optional[TreeNode]
    last_host_node: Optional[TreeNode]
    host_hit_length: int = 0


class BasePrefixCache(abc.ABC):
    """Cache can be indexed by either rid or key."""

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        pass

    @abc.abstractmethod
    def cache_finished_req(self, req: Any, **kwargs):
        pass

    @abc.abstractmethod
    def cache_unfinished_req(self, req: Any, **kwargs):
        pass

    @abc.abstractmethod
    def evict(self, num_tokens: int):
        pass

    @abc.abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abc.abstractmethod
    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        raise NotImplementedError()

    def take_events(self):
        return []
