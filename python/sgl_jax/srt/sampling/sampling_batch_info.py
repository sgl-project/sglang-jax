from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.sampling.sampling_params import TOP_K_ALL
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

import threading

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import mesh as mesh_lib

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclasses.dataclass
class SamplingBatchInfo:
    """
    keep the array on device same to sglang
    """

    # Basic batched sampling params
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array

    # Whether all requests use greedy sampling
    is_all_greedy: bool = False

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool = False

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool = False

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool = False

    # An event used for overlap schedule
    sampling_info_done: Optional[threading.Event] = None

    def tree_flatten(self):
        children = (
            self.temperatures,
            self.top_ps,
            self.top_ks,
            self.min_ps,
        )

        aux_data = {
            "is_all_greedy": self.is_all_greedy,
            "need_top_p_sampling": self.need_top_p_sampling,
            "need_top_k_sampling": self.need_top_k_sampling,
            "need_min_p_sampling": self.need_min_p_sampling,
            "sampling_info_done": self.sampling_info_done,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.temperatures = children[0]
        obj.top_ps = children[1]
        obj.top_ks = children[2]
        obj.min_ps = children[3]

        obj.is_all_greedy = aux_data["is_all_greedy"]
        obj.need_top_p_sampling = aux_data["need_top_p_sampling"]
        obj.need_top_k_sampling = aux_data["need_top_k_sampling"]
        obj.need_min_p_sampling = aux_data["need_min_p_sampling"]
        obj.sampling_info_done = aux_data["sampling_info_done"]

        return obj

    @classmethod
    def generate_for_precompile(cls, bs: int, vocab_size: int, mesh: mesh_lib.Mesh):
        # device = batch.device
        temperatures = jnp.array([1.0 for _ in range(bs)], dtype=jnp.float32).reshape(
            -1, 1
        )
        top_ps = jnp.array([1.0 for _ in range(bs)], dtype=jnp.float32)
        top_ks = jnp.array([-1 for _ in range(bs)], dtype=jnp.int32)
        min_ps = jnp.array([0.0 for _ in range(bs)], dtype=jnp.float32)

        temperatures_device = device_array(mesh, temperatures)
        top_ps_device = device_array(mesh, top_ps)
        top_ks_device = device_array(mesh, top_ks)
        min_ps_device = device_array(mesh, min_ps)

        ret = cls(
            temperatures=temperatures_device,
            top_ps=top_ps_device,
            top_ks=top_ks_device,
            min_ps=min_ps_device,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            sampling_info_done=None,
        )
        return ret

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        reqs = batch.reqs
        # device = batch.device
        temperatures = np.array(
            [r.sampling_params.temperature for r in reqs],
            dtype=jnp.float32,
        ).reshape(-1, 1)
        top_ps = np.array([r.sampling_params.top_p for r in reqs], dtype=jnp.float32)
        top_ks = np.array([r.sampling_params.top_k for r in reqs], dtype=jnp.int32)
        min_ps = np.array([r.sampling_params.min_p for r in reqs], dtype=jnp.float32)

        temperatures_device = device_array(batch.mesh, temperatures)
        top_ps_device = device_array(batch.mesh, top_ks)
        top_ks_device = device_array(batch.mesh, top_ks)
        min_ps_device = device_array(batch.mesh, min_ps)

        # temperatures_device, top_ps_device, top_ks_device, min_ps_device = device_array(
        #     batch.mesh, (temperatures, top_ps, top_ks, min_ps)
        # )

        # logit_bias = None
        # if any(r.sampling_params.logit_bias is not None for r in reqs):
        #     logit_bias = np.zeros(len(reqs), vocab_size)
        #     logit_bias = device_array(batch.mesh, logit_bias)
        #     for i, r in enumerate(reqs):
        #         if r.sampling_params.logit_bias is not None:
        #             for key, value in r.sampling_params.logit_bias.items():
        #                 logit_bias[i, int(key)] = value

        ret = cls(
            temperatures=temperatures_device,
            top_ps=top_ps_device,
            top_ks=top_ks_device,
            min_ps=min_ps_device,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def apply_logits_bias(self, logits: jax.Array):
        # if self.logit_bias is not None:
        #     return logits + self.logit_bias
        return logits

    def filter_batch(self, keep_indices: List[int], keep_indices_device: jax.Array):
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices_device])

    def merge_batch(self, other: "SamplingBatchInfo", mesh: mesh_lib.Mesh):
        # Note: because the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, jnp.concat([self_val, other_val]))

        self.is_all_greedy &= other.is_all_greedy
        self.need_top_p_sampling |= other.need_top_p_sampling
        self.need_top_k_sampling |= other.need_top_k_sampling
        self.need_min_p_sampling |= other.need_min_p_sampling


def merge_bias_tensor(
    lhs: Optional[jax.Array],
    rhs: Optional[jax.Array],
    bs1: int,
    bs2: int,
    default: float,
    mesh: mesh_lib.Mesh,
):
    """Merge two bias array for batch merging.

    Args:
        lhs: Left-hand side array
        rhs: Right-hand side array
        bs1: Batch size of left-hand side array
        bs2: Batch size of right-hand side array
        device: Device to place the merged array on
        default: Default value for missing array elements

    Returns:
        Merged array or None if both inputs are None
    """
    if lhs is None and rhs is None:
        return None

    if lhs is not None and rhs is not None:
        return jax.concat([lhs, rhs])
    else:
        if lhs is not None:
            shape, dtype = lhs.shape[1:], lhs.dtype
        else:
            shape, dtype = rhs.shape[1:], rhs.dtype

        if lhs is None:
            lhs = device_array(
                mesh,
                jnp.full((bs1, *shape), fill_value=default, dtype=dtype),
            )
        if rhs is None:
            rhs = device_array(
                mesh,
                jnp.full((bs2, *shape), fill_value=default, dtype=dtype),
            )
        return jnp.concat([lhs, rhs])
