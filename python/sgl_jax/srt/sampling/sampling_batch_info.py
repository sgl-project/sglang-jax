from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.sampling.sampling_params import DEFAULT_SAMPLING_SEED, TOP_K_ALL
from sgl_jax.srt.utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ModelWorkerBatch

import threading

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import mesh as mesh_lib

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclasses.dataclass
class SamplingMetadata:
    """
    SamplingMetadata is used as input parameter for jitted sample function.
    """

    # logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # sample
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array
    sampling_seeds: jax.Array
    is_all_greedy: bool = False
    need_min_p_sampling: bool = False

    def tree_flatten(self):
        children = (
            self.temperatures,
            self.top_ps,
            self.top_ks,
            self.min_ps,
            self.sampling_seeds,
            self.is_all_greedy,
            self.need_min_p_sampling,
        )

        aux_data = {
            "return_logprob": self.return_logprob,
            "top_logprobs_nums": self.top_logprobs_nums,
            "token_ids_logprobs": self.token_ids_logprobs,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.temperatures = children[0]
        obj.top_ps = children[1]
        obj.top_ks = children[2]
        obj.min_ps = children[3]
        obj.sampling_seeds = children[4]
        obj.is_all_greedy = children[5]
        obj.need_min_p_sampling = children[6]

        obj.return_logprob = aux_data["return_logprob"]
        obj.top_logprobs_nums = aux_data["top_logprobs_nums"]
        obj.token_ids_logprobs = aux_data["token_ids_logprobs"]

        return obj

    @classmethod
    def from_model_worker_batch(
        cls,
        batch: ModelWorkerBatch,
        pad_size: int = 0,
        mesh: Mesh = None,
    ) -> SamplingMetadata:
        sharding = (
            NamedSharding(mesh, PartitionSpec()) if jax.process_count() == 1 else None
        )
        padded_temperatures = np.concat(
            [
                batch.sampling_info.temperatures,
                np.array(
                    [1.0] * pad_size, dtype=batch.sampling_info.temperatures.dtype
                ),
            ]
        ).reshape(-1, 1)
        padded_top_ps = np.concat(
            [
                batch.sampling_info.top_ps,
                np.array([1.0] * pad_size, dtype=batch.sampling_info.top_ps.dtype),
            ]
        )
        padded_top_ks = np.concat(
            [
                batch.sampling_info.top_ks,
                np.array([1] * pad_size, dtype=batch.sampling_info.top_ks.dtype),
            ]
        )
        padded_min_ps = np.concat(
            [
                batch.sampling_info.min_ps,
                np.array([0.0] * pad_size, dtype=batch.sampling_info.min_ps.dtype),
            ]
        )
        if batch.sampling_info.sampling_seeds is not None:
            padded_sampling_seeds = np.concat(
                [
                    batch.sampling_info.sampling_seeds,
                    np.array(
                        [DEFAULT_SAMPLING_SEED] * pad_size,
                        dtype=batch.sampling_info.sampling_seeds.dtype,
                    ),
                ]
            )
            sampling_seeds_device = device_array(
                padded_sampling_seeds, sharding=sharding
            )
        else:
            sampling_seeds_device = None

        (temperatures_device, top_ps_device, top_ks_device, min_ps_device) = (
            device_array(
                (padded_temperatures, padded_top_ps, padded_top_ks, padded_min_ps),
                sharding=sharding,
            )
        )

        return cls(
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            temperatures=temperatures_device,
            top_ps=top_ps_device,
            top_ks=top_ks_device,
            min_ps=min_ps_device,
            sampling_seeds=sampling_seeds_device,
            is_all_greedy=batch.sampling_info.is_all_greedy,
            need_min_p_sampling=batch.sampling_info.need_min_p_sampling,
        )


@dataclasses.dataclass
class SamplingBatchInfo:
    """
    keep the array on device same to sglang
    """

    # Basic batched sampling params
    temperatures: np.ndarray
    top_ps: np.ndarray
    top_ks: np.ndarray
    min_ps: np.ndarray

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

    sampling_seeds: Optional[np.ndarray] = None

    @classmethod
    def _get_global_server_args_dict(cls):
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        return global_server_args_dict

    @classmethod
    def generate_for_precompile(cls, bs: int):
        temperatures = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([0.9 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([30 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING"):
            sampling_seeds = np.array(
                [DEFAULT_SAMPLING_SEED for _ in range(bs)], dtype=np.int32
            )
        else:
            sampling_seeds = None

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=True,
            sampling_info_done=None,
            sampling_seeds=sampling_seeds,
        )
        return ret

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        global_server_args_dict = cls._get_global_server_args_dict()
        enable_deterministic = global_server_args_dict["enable_deterministic_sampling"]
        reqs = batch.reqs
        temperatures = np.array(
            [r.sampling_params.temperature for r in reqs],
            dtype=np.float32,
        )
        top_ps = np.array([r.sampling_params.top_p for r in reqs], dtype=np.float32)
        top_ks = np.array([r.sampling_params.top_k for r in reqs], dtype=np.int32)
        min_ps = np.array([r.sampling_params.min_p for r in reqs], dtype=np.float32)

        sampling_seeds = (
            np.array(
                [r.sampling_params.sampling_seed for r in reqs],
                dtype=np.int32,
            )
            if enable_deterministic
            else None
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            sampling_seeds=sampling_seeds,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def apply_logits_bias(self, logits: jax.Array):
        return logits

    def filter_batch(self, keep_indices: np.ndarray):
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seeds",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices])

    def merge_batch(self, other: "SamplingBatchInfo", mesh: Mesh):
        # Note: because the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "sampling_seeds",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            if self_val is not None and other_val is not None:
                setattr(self, item, np.concat([self_val, other_val]))

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
                jnp.full((bs1, *shape), fill_value=default, dtype=dtype),
            )
        if rhs is None:
            rhs = device_array(
                jnp.full((bs2, *shape), fill_value=default, dtype=dtype),
            )
        return jnp.concat([lhs, rhs])
