from __future__ import annotations

import dataclasses
import logging
from functools import partial
from typing import TYPE_CHECKING

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.sampling import penaltylib
from sgl_jax.srt.sampling.sampling_params import DEFAULT_SAMPLING_SEED, TOP_K_ALL
from sgl_jax.srt.utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch

import threading

import jax
import numpy as np

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclasses.dataclass
class SamplingMetadata:
    """
    SamplingMetadata is used as input parameter for jitted sample function.
    """

    # logprob
    return_logprob: bool
    top_logprobs_nums: list[int] | None
    token_ids_logprobs: list[list[int]] | None

    # sample
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array
    sampling_seeds: jax.Array
    is_all_greedy: bool = False
    need_min_p_sampling: bool = False

    # penalty
    do_penalties: bool = False
    linear_penalty: jax.Array = None

    def tree_flatten(self):
        children = (
            self.temperatures,
            self.top_ps,
            self.top_ks,
            self.min_ps,
            self.sampling_seeds,
            self.is_all_greedy,
            self.need_min_p_sampling,
            self.linear_penalty,
            self.do_penalties,
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
        obj.linear_penalty = children[7]
        obj.do_penalties = children[8]

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
        vocab_size: int = 32000,
    ) -> SamplingMetadata:
        sharding = NamedSharding(mesh, PartitionSpec()) if jax.process_count() == 1 else None
        padded_temperatures = np.concat(
            [
                batch.sampling_info.temperatures,
                np.array([1.0] * pad_size, dtype=batch.sampling_info.temperatures.dtype),
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
            sampling_seeds_device = device_array(padded_sampling_seeds, sharding=sharding)
        else:
            sampling_seeds_device = None

        (temperatures_device, top_ps_device, top_ks_device, min_ps_device) = device_array(
            (padded_temperatures, padded_top_ps, padded_top_ks, padded_min_ps),
            sharding=sharding,
        )

        # Extract penalty information from penalizer orchestrator
        linear_penalty_device = None
        do_penalties = False
        linear_penalty_sharding = NamedSharding(mesh, PartitionSpec(None, "tensor"))

        # Handle linear penalty independently (created by update_penalties)
        if (
            batch.sampling_info.linear_penalty is not None
            and batch.sampling_info.linear_penalty.size > 0
        ):
            do_penalties = True
            original_linear_penalty = batch.sampling_info.linear_penalty
            if pad_size > 0:
                # Pad with zero rows for vocabulary dimension
                pad_rows = np.zeros(
                    (pad_size, original_linear_penalty.shape[1]),
                    dtype=original_linear_penalty.dtype,
                )
                padded_linear_penalty = np.concat([original_linear_penalty, pad_rows], axis=0)
            else:
                padded_linear_penalty = original_linear_penalty

            linear_penalty_device = device_array(
                padded_linear_penalty,
                sharding=linear_penalty_sharding,
            )

        # Handle individual penalties from orchestrator
        if (
            linear_penalty_device is None
            and batch.sampling_info.penalizer_orchestrator
            and batch.sampling_info.penalizer_orchestrator.is_required
        ):
            do_penalties = True
            orchestrator = batch.sampling_info.penalizer_orchestrator

            original_linear_penalty = orchestrator.apply()
            if pad_size > 0:
                pad_rows = np.zeros(
                    (pad_size, original_linear_penalty.shape[1]),
                    dtype=original_linear_penalty.dtype,
                )
                padded_linear_penalty = np.concat([original_linear_penalty, pad_rows], axis=0)
            else:
                padded_linear_penalty = original_linear_penalty

            linear_penalty_device = device_array(
                padded_linear_penalty,
                sharding=linear_penalty_sharding,
            )
        if linear_penalty_device is None:
            linear_penalty_device = _create_zero_penalty(
                (padded_temperatures.shape[0], vocab_size), linear_penalty_sharding
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
            linear_penalty=linear_penalty_device,
            do_penalties=do_penalties,
        )


@partial(jax.jit, static_argnames=["shape", "sharding"])
def _create_zero_penalty(shape, sharding):
    return jnp.zeros(shape=shape, dtype=jnp.bfloat16, device=sharding)


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

    vocab_size: int

    # Whether all requests use greedy sampling
    is_all_greedy: bool = False

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool = False

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool = False

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool = False

    # An event used for overlap schedule
    sampling_info_done: threading.Event | None = None

    sampling_seeds: np.ndarray | None = None

    # Penalizer
    penalizer_orchestrator: penaltylib.BatchedPenalizerOrchestrator | None = None
    linear_penalty: np.ndarray = None

    @classmethod
    def _get_global_server_args_dict(cls):
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        return global_server_args_dict

    @classmethod
    def generate_for_precompile(
        cls,
        bs: int,
        vocab_size: int = 32000,
    ):
        temperatures = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([0.9 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([30 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([0.6 for _ in range(bs)], dtype=np.float32)
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_SAMPLING"):
            sampling_seeds = np.array([DEFAULT_SAMPLING_SEED for _ in range(bs)], dtype=np.int32)
        else:
            sampling_seeds = None

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=True,
            sampling_info_done=None,
            sampling_seeds=sampling_seeds,
            penalizer_orchestrator=None,
            linear_penalty=None,
        )
        return ret

    @classmethod
    def from_schedule_batch(
        cls,
        batch: ScheduleBatch,
        vocab_size: int,
    ):
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

        # Initialize penalty orchestrator
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
            },
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
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            # Get penalty array directly from orchestrator - no np.zeros() needed!
            self.linear_penalty = self.penalizer_orchestrator.apply()
        else:
            self.linear_penalty = None

    def filter_batch(self, keep_indices: np.ndarray):
        self.penalizer_orchestrator.filter(keep_indices)

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

    def merge_batch(self, other: SamplingBatchInfo):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)
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

    def cumulate_output_tokens(self, output_ids: jax.Array):
        """
        Feed the output tokens to the penalty orchestrator.

        Args:
            output_ids (jax.Array): The output tokens.
        """
        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            self.penalizer_orchestrator.cumulate_output_tokens(output_ids)
