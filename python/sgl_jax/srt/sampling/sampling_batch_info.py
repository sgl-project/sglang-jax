from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.sampling import penaltylib
from sgl_jax.srt.sampling.sampling_params import TOP_K_ALL
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
    is_all_greedy: bool = False
    need_min_p_sampling: bool = False

    # penalty
    linear_penalty: Optional[jax.Array] = None
    frequency_penalties: Optional[jax.Array] = None
    presence_penalties: Optional[jax.Array] = None
    min_new_tokens: Optional[jax.Array] = None
    stop_token_penalties: Optional[jax.Array] = None
    len_output_tokens: Optional[jax.Array] = None

    # Cumulative penalty matrices for application in sampler
    cumulated_frequency_penalties: Optional[jax.Array] = None
    cumulated_presence_penalties: Optional[jax.Array] = None

    def tree_flatten(self):
        children = (
            self.temperatures,
            self.top_ps,
            self.top_ks,
            self.min_ps,
            self.frequency_penalties,
            self.presence_penalties,
            self.min_new_tokens,
            self.stop_token_penalties,
            self.len_output_tokens,
            self.cumulated_frequency_penalties,
            self.cumulated_presence_penalties,
            self.linear_penalty,
        )

        aux_data = {
            "return_logprob": self.return_logprob,
            "top_logprobs_nums": self.top_logprobs_nums,
            "token_ids_logprobs": self.token_ids_logprobs,
            "is_all_greedy": self.is_all_greedy,
            "need_min_p_sampling": self.need_min_p_sampling,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.temperatures = children[0]
        obj.top_ps = children[1]
        obj.top_ks = children[2]
        obj.min_ps = children[3]
        obj.frequency_penalties = children[4]
        obj.presence_penalties = children[5]
        obj.min_new_tokens = children[6]
        obj.stop_token_penalties = children[7]
        obj.len_output_tokens = children[8]
        obj.cumulated_frequency_penalties = children[9]
        obj.cumulated_presence_penalties = children[10]
        obj.linear_penalty = children[11]

        obj.return_logprob = aux_data["return_logprob"]
        obj.top_logprobs_nums = aux_data["top_logprobs_nums"]
        obj.token_ids_logprobs = aux_data["token_ids_logprobs"]
        obj.is_all_greedy = aux_data["is_all_greedy"]
        obj.need_min_p_sampling = aux_data["need_min_p_sampling"]

        return obj

    @classmethod
    def from_model_worker_batch(
        cls,
        batch: ModelWorkerBatch,
        pad_size: int = 0,
        mesh: Mesh = None,
    ) -> SamplingMetadata:
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
                np.array([-1] * pad_size, dtype=batch.sampling_info.top_ks.dtype),
            ]
        )
        padded_min_ps = np.concat(
            [
                batch.sampling_info.min_ps,
                np.array([0.0] * pad_size, dtype=batch.sampling_info.min_ps.dtype),
            ]
        )

        (temperatures_device, top_ps_device, top_ks_device, min_ps_device) = (
            device_array(
                (padded_temperatures, padded_top_ps, padded_top_ks, padded_min_ps),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        )

        # Extract penalty information from penalizer orchestrator
        frequency_penalties_device = None
        presence_penalties_device = None
        min_new_tokens_device = None
        stop_token_penalties_device = None
        len_output_tokens_device = None
        cumulated_frequency_penalties_device = None
        cumulated_presence_penalties_device = None

        if (
            batch.sampling_info.penalizer_orchestrator
            and batch.sampling_info.penalizer_orchestrator.is_required
        ):
            orchestrator = batch.sampling_info.penalizer_orchestrator

            # Extract frequency penalty data
            freq_penalizer = orchestrator.penalizers.get(
                penaltylib.BatchedFrequencyPenalizer
            )
            if freq_penalizer and freq_penalizer.is_prepared():
                # Apply same padding logic as other parameters
                original_freq_penalties = freq_penalizer.frequency_penalties.squeeze(
                    axis=1
                )
                padded_freq_penalties = np.concat(
                    [
                        original_freq_penalties,
                        np.array([0.0] * pad_size, dtype=original_freq_penalties.dtype),
                    ]
                )

                original_cumulated_freq_penalties = (
                    freq_penalizer.cumulated_frequency_penalties
                )
                # Pad with zero rows for vocabulary dimension
                pad_rows = np.zeros(
                    (pad_size, original_cumulated_freq_penalties.shape[1]),
                    dtype=original_cumulated_freq_penalties.dtype,
                )
                padded_cumulated_freq_penalties = np.concat(
                    [original_cumulated_freq_penalties, pad_rows], axis=0
                )

                frequency_penalties_device = device_array(
                    padded_freq_penalties,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                cumulated_frequency_penalties_device = device_array(
                    padded_cumulated_freq_penalties,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )

            # Extract presence penalty data
            pres_penalizer = orchestrator.penalizers.get(
                penaltylib.BatchedPresencePenalizer
            )
            if pres_penalizer and pres_penalizer.is_prepared():
                # Apply same padding logic as other parameters
                original_pres_penalties = pres_penalizer.presence_penalties.squeeze(
                    axis=1
                )
                padded_pres_penalties = np.concat(
                    [
                        original_pres_penalties,
                        np.array([0.0] * pad_size, dtype=original_pres_penalties.dtype),
                    ]
                )

                original_cumulated_pres_penalties = (
                    pres_penalizer.cumulated_presence_penalties
                )
                # Pad with zero rows for vocabulary dimension
                pad_rows = np.zeros(
                    (pad_size, original_cumulated_pres_penalties.shape[1]),
                    dtype=original_cumulated_pres_penalties.dtype,
                )
                padded_cumulated_pres_penalties = np.concat(
                    [original_cumulated_pres_penalties, pad_rows], axis=0
                )

                presence_penalties_device = device_array(
                    padded_pres_penalties,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                cumulated_presence_penalties_device = device_array(
                    padded_cumulated_pres_penalties,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )

            # Extract min new tokens penalty data
            min_tokens_penalizer = orchestrator.penalizers.get(
                penaltylib.BatchedMinNewTokensPenalizer
            )
            if min_tokens_penalizer and min_tokens_penalizer.is_prepared():
                # Apply same padding logic as other parameters
                original_min_new_tokens = min_tokens_penalizer.min_new_tokens.squeeze(
                    axis=1
                )
                padded_min_new_tokens = np.concat(
                    [
                        original_min_new_tokens,
                        np.array([0] * pad_size, dtype=original_min_new_tokens.dtype),
                    ]
                )

                original_stop_token_penalties = (
                    min_tokens_penalizer.stop_token_penalties
                )
                # Pad with zero rows for vocabulary dimension
                pad_rows = np.zeros(
                    (pad_size, original_stop_token_penalties.shape[1]),
                    dtype=original_stop_token_penalties.dtype,
                )
                padded_stop_token_penalties = np.concat(
                    [original_stop_token_penalties, pad_rows], axis=0
                )

                original_len_output_tokens = (
                    min_tokens_penalizer.len_output_tokens.squeeze(axis=1)
                )
                padded_len_output_tokens = np.concat(
                    [
                        original_len_output_tokens,
                        np.array(
                            [0] * pad_size, dtype=original_len_output_tokens.dtype
                        ),
                    ]
                )

                min_new_tokens_device = device_array(
                    padded_min_new_tokens,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                stop_token_penalties_device = device_array(
                    padded_stop_token_penalties,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                len_output_tokens_device = device_array(
                    padded_len_output_tokens,
                    sharding=(
                        NamedSharding(mesh, PartitionSpec())
                        if jax.process_count() == 1
                        else None
                    ),
                )

        # For JIT compilation stability, provide default zero tensors instead of None
        # when penalty data is not available (e.g., during precompile)
        # batch_size should include padding to match other parameters
        batch_size = len(batch.sampling_info.temperatures) + pad_size

        # Create default zero tensors for penalty data if needed
        if frequency_penalties_device is None:
            frequency_penalties_device = device_array(
                jnp.zeros(batch_size, dtype=jnp.float32),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if presence_penalties_device is None:
            presence_penalties_device = device_array(
                jnp.zeros(batch_size, dtype=jnp.float32),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if min_new_tokens_device is None:
            min_new_tokens_device = device_array(
                jnp.zeros(batch_size, dtype=jnp.int32),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if stop_token_penalties_device is None:
            stop_token_penalties_device = device_array(
                jnp.zeros(
                    (batch_size, batch.sampling_info.vocab_size), dtype=jnp.float32
                ),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if len_output_tokens_device is None:
            len_output_tokens_device = device_array(
                jnp.zeros(batch_size, dtype=jnp.int32),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if cumulated_frequency_penalties_device is None:
            cumulated_frequency_penalties_device = device_array(
                jnp.zeros(
                    (batch_size, batch.sampling_info.vocab_size), dtype=jnp.float32
                ),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )
        if cumulated_presence_penalties_device is None:
            cumulated_presence_penalties_device = device_array(
                jnp.zeros(
                    (batch_size, batch.sampling_info.vocab_size), dtype=jnp.float32
                ),
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )

        # Handle linear_penalty from overlap mode
        linear_penalty_device = None
        if (
            batch.sampling_info.linear_penalty is not None
            and batch.sampling_info.linear_penalty.size > 0
        ):
            # Apply same padding logic to linear_penalty
            original_linear_penalty = batch.sampling_info.linear_penalty
            if pad_size > 0:
                # Pad with zero rows for vocabulary dimension
                pad_rows = np.zeros(
                    (pad_size, original_linear_penalty.shape[1]),
                    dtype=original_linear_penalty.dtype,
                )
                padded_linear_penalty = np.concat(
                    [original_linear_penalty, pad_rows], axis=0
                )
            else:
                padded_linear_penalty = original_linear_penalty

            linear_penalty_device = device_array(
                padded_linear_penalty,
                sharding=(
                    NamedSharding(mesh, PartitionSpec())
                    if jax.process_count() == 1
                    else None
                ),
            )

        return cls(
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            temperatures=temperatures_device,
            top_ps=top_ps_device,
            top_ks=top_ks_device,
            min_ps=min_ps_device,
            is_all_greedy=batch.sampling_info.is_all_greedy,
            need_min_p_sampling=batch.sampling_info.need_min_p_sampling,
            frequency_penalties=frequency_penalties_device,
            presence_penalties=presence_penalties_device,
            min_new_tokens=min_new_tokens_device,
            stop_token_penalties=stop_token_penalties_device,
            len_output_tokens=len_output_tokens_device,
            cumulated_frequency_penalties=cumulated_frequency_penalties_device,
            cumulated_presence_penalties=cumulated_presence_penalties_device,
            linear_penalty=linear_penalty_device,
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
    sampling_info_done: Optional[threading.Event] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    linear_penalty: np.ndarray = None

    @classmethod
    def generate_for_precompile(cls, bs: int, vocab_size: int = 32000):
        temperatures = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([-1 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([0.0 for _ in range(bs)], dtype=np.float32)

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            vocab_size=vocab_size,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            sampling_info_done=None,
            penalizer_orchestrator=None,
            linear_penalty=None,
        )
        return ret

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        reqs = batch.reqs
        temperatures = np.array(
            [r.sampling_params.temperature for r in reqs],
            dtype=np.float32,
        )
        top_ps = np.array([r.sampling_params.top_p for r in reqs], dtype=np.float32)
        top_ks = np.array([r.sampling_params.top_k for r in reqs], dtype=np.int32)
        min_ps = np.array([r.sampling_params.min_p for r in reqs], dtype=np.float32)

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
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.linear_penalty = np.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=np.float32,
            )
            self.penalizer_orchestrator.apply(self.linear_penalty)
        else:
            self.linear_penalty = None

    def filter_batch(self, keep_indices: np.ndarray):
        self.penalizer_orchestrator.filter(keep_indices)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices])

    def merge_batch(self, other: "SamplingBatchInfo", mesh: Mesh = None):
        del mesh  # Parameter not used in current implementation
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)
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


def merge_bias_tensor(
    lhs: Optional[jax.Array],
    rhs: Optional[jax.Array],
    bs1: int,
    bs2: int,
    default: float,
    mesh: mesh_lib.Mesh = None,
):
    del mesh  # Parameter not used in current implementation
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
