from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.sampling import penaltylib
from sgl_jax.srt.sampling.sampling_params import DEFAULT_SAMPLING_SEED, TOP_K_ALL
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

    # penalty
    do_penalties: bool = False
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
            self.sampling_seeds,
            self.is_all_greedy,
            self.need_min_p_sampling,
            self.linear_penalty,
            self.frequency_penalties,
            self.presence_penalties,
            self.min_new_tokens,
            self.stop_token_penalties,
            self.len_output_tokens,
            self.cumulated_frequency_penalties,
            self.cumulated_presence_penalties,
        )

        aux_data = {
            "return_logprob": self.return_logprob,
            "top_logprobs_nums": self.top_logprobs_nums,
            "token_ids_logprobs": self.token_ids_logprobs,
            "is_all_greedy": self.is_all_greedy,
            "need_min_p_sampling": self.need_min_p_sampling,
            "do_penalties": self.do_penalties,
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
        obj.frequency_penalties = children[8]
        obj.presence_penalties = children[9]
        obj.min_new_tokens = children[10]
        obj.stop_token_penalties = children[11]
        obj.len_output_tokens = children[12]
        obj.cumulated_frequency_penalties = children[13]
        obj.cumulated_presence_penalties = children[14]

        obj.return_logprob = aux_data["return_logprob"]
        obj.top_logprobs_nums = aux_data["top_logprobs_nums"]
        obj.token_ids_logprobs = aux_data["token_ids_logprobs"]
        obj.is_all_greedy = aux_data["is_all_greedy"]
        obj.need_min_p_sampling = aux_data["need_min_p_sampling"]
        obj.do_penalties = aux_data["do_penalties"]

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

        # Extract penalty information from penalizer orchestrator
        frequency_penalties_device = None
        presence_penalties_device = None
        min_new_tokens_device = None
        stop_token_penalties_device = None
        len_output_tokens_device = None
        cumulated_frequency_penalties_device = None
        cumulated_presence_penalties_device = None
        linear_penalty_device = None
        do_penalties = False

        # Handle linear penalty independently (created by update_penalties)
        if (
            batch.sampling_info.linear_penalty is not None
            and batch.sampling_info.linear_penalty.size > 0
        ):
            do_penalties = True
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
                sharding=sharding,
            )

        # Handle individual penalties from orchestrator
        if (
            batch.sampling_info.penalizer_orchestrator
            and batch.sampling_info.penalizer_orchestrator.is_required
        ):
            do_penalties = True
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

                (
                    frequency_penalties_device,
                    cumulated_frequency_penalties_device,
                ) = device_array(
                    (padded_freq_penalties, padded_cumulated_freq_penalties),
                    sharding=sharding,
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

                (
                    presence_penalties_device,
                    cumulated_presence_penalties_device,
                ) = device_array(
                    (padded_pres_penalties, padded_cumulated_pres_penalties),
                    sharding=sharding,
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

                (
                    min_new_tokens_device,
                    len_output_tokens_device,
                    stop_token_penalties_device,
                ) = device_array(
                    (
                        padded_min_new_tokens,
                        padded_len_output_tokens,
                        padded_stop_token_penalties,
                    ),
                    sharding=sharding,
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
            frequency_penalties=frequency_penalties_device,
            presence_penalties=presence_penalties_device,
            min_new_tokens=min_new_tokens_device,
            stop_token_penalties=stop_token_penalties_device,
            len_output_tokens=len_output_tokens_device,
            cumulated_frequency_penalties=cumulated_frequency_penalties_device,
            cumulated_presence_penalties=cumulated_presence_penalties_device,
            linear_penalty=linear_penalty_device,
            do_penalties=do_penalties,
        )

    @classmethod
    def from_model_worker_batch_for_precompile(
        cls,
        batch: ModelWorkerBatch,
        pad_size: int = 0,
        mesh: Mesh = None,
    ) -> SamplingMetadata:
        """
        Create SamplingMetadata for precompile with all possible penalty shapes.
        Since JAX compilation only cares about shapes, we create tensors with appropriate
        shapes for all penalty types to ensure comprehensive compilation coverage.
        """
        # Basic sampling parameters (same as original method)
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
                sharding=sharding,
            )
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
                padded_sampling_seeds,
                sharding=sharding,
            )
        else:
            sampling_seeds_device = None

        # Calculate batch size and vocab size
        batch_size = len(batch.sampling_info.temperatures) + pad_size
        vocab_size = batch.sampling_info.vocab_size

        # Create all possible penalty tensors with appropriate shapes
        # This ensures JAX compiles all penalty application branches

        # Scalar penalty parameters
        padded_frequency_penalties = jnp.ones(batch_size, dtype=jnp.float32) * 0.1
        padded_presence_penalties = jnp.ones(batch_size, dtype=jnp.float32) * 0.15
        padded_min_new_tokens = jnp.ones(batch_size, dtype=jnp.int32) * 5
        padded_len_output_tokens = jnp.ones(batch_size, dtype=jnp.int32) * 3

        (
            frequency_penalties_device,
            presence_penalties_device,
            min_new_tokens_device,
            len_output_tokens_device,
        ) = device_array(
            (
                padded_frequency_penalties,
                padded_presence_penalties,
                padded_min_new_tokens,
                padded_len_output_tokens,
            ),
            sharding=sharding,
        )

        # Matrix penalty parameters
        padded_cumulated_frequency_penalties = (
            jnp.ones((batch_size, vocab_size), dtype=jnp.float32) * 0.05
        )
        padded_cumulated_presence_penalties = (
            jnp.ones((batch_size, vocab_size), dtype=jnp.float32) * 0.1
        )
        padded_stop_token_penalties = jnp.ones(
            (batch_size, vocab_size), dtype=jnp.float32
        ) * (-1000.0)
        padded_linear_penalty = (
            jnp.ones((batch_size, vocab_size), dtype=jnp.float32) * 0.2
        )

        (
            cumulated_frequency_penalties_device,
            cumulated_presence_penalties_device,
            stop_token_penalties_device,
            linear_penalty_device,
        ) = device_array(
            (
                padded_cumulated_frequency_penalties,
                padded_cumulated_presence_penalties,
                padded_stop_token_penalties,
                padded_linear_penalty,
            ),
            sharding=sharding,
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
            sampling_seeds=sampling_seeds_device,
            do_penalties=True,
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

    sampling_seeds: Optional[np.ndarray] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    linear_penalty: np.ndarray = None

    @classmethod
    def _get_global_server_args_dict(cls):
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        return global_server_args_dict

    @classmethod
    def generate_for_precompile(
        cls, bs: int, vocab_size: int = 32000, do_penalties: bool = False
    ):
        temperatures = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        top_ps = np.array([1.0 for _ in range(bs)], dtype=np.float32)
        top_ks = np.array([-1 for _ in range(bs)], dtype=np.int32)
        min_ps = np.array([0.0 for _ in range(bs)], dtype=np.float32)
        sampling_seeds = np.array([0 for _ in range(bs)], dtype=np.int32)

        # Create mock batch for precompile with penalty-enabled requests
        mock_batch = cls._create_mock_batch_for_precompile(bs, do_penalties)

        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=mock_batch,
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
            vocab_size=vocab_size,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=True,
            sampling_info_done=None,
            sampling_seeds=sampling_seeds,
            penalizer_orchestrator=penalizer_orchestrator,
            linear_penalty=None,
        )
        return ret

    @classmethod
    def _create_mock_batch_for_precompile(cls, bs: int, do_penalties: bool = False):
        """Create a mock batch with penalty-enabled requests for precompile."""
        from sgl_jax.srt.sampling.sampling_params import SamplingParams

        class MockReq:
            def __init__(self, idx):
                # Create sampling params with various penalty settings to ensure
                # orchestrator recognizes penalties as required
                if do_penalties:
                    self.sampling_params = SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                        frequency_penalty=0.1,  # Non-zero to trigger orchestrator
                        presence_penalty=0.1,  # Non-zero to trigger orchestrator
                        min_new_tokens=5,  # Non-zero to trigger orchestrator
                    )
                else:
                    self.sampling_params = SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                    )

                # Create mock tokenizer for min_new_tokens penalizer
                class MockTokenizer:
                    eos_token_id = 0
                    additional_stop_token_ids = [1, 2]

                self.tokenizer = MockTokenizer()

        class MockBatch:
            def __init__(self, bs):
                self.reqs = [MockReq(i) for i in range(bs)]

        return MockBatch(bs)

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
            "sampling_seeds",
        ]:
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[keep_indices])

    def merge_batch(self, other: "SamplingBatchInfo"):
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
