"""
Description: End-to-end tests for BgmvLoRABackend with Cartesian product combinations.
Note: Refer to https://github.com/sgl-project/sglang/blob/main/test/srt/lora/test_chunked_sgmv_backend.py
"""

import itertools
import random
import unittest
from enum import Enum
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.lora.backend.bgmv_backend import BgmvLoRABackend
from sgl_jax.srt.lora.utils import (
    get_lora_a_output_sharding,
    get_lora_b_output_sharding,
)
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


def safe_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Matrix multiplication with mixed precision handling for bfloat16."""
    result = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))
    return result.astype(a.dtype)


class BatchComposition(Enum):
    UNIFORM = "uniform"
    MIXED = "mixed"
    SKEWED = "skewed"
    NONE = "_NO_LORA_"


class BatchMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def reference_lora_a_gemm(
    x: jax.Array,
    weights: jax.Array,
    seq_lengths: List[int],
    lora_assignments: List[str],
    lora_configs: Dict[str, Tuple[int, int, int, int]],
    scalings: List[float],
) -> jax.Array:
    """Reference implementation of LoRA A GEMM."""
    # if weights.size == 0:
    #     return jnp.zeros((x.shape[0], 0), dtype=x.dtype)

    total_seq_len, input_dim = x.shape
    num_loras, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim

    output = jnp.zeros((total_seq_len, max_rank), dtype=x.dtype)

    unique_loras = sorted(set(lora_assignments))
    lora_name_to_idx = {name: idx for idx, name in enumerate(unique_loras)}

    token_offset = 0
    for seq_idx, (seq_len, lora_name) in enumerate(zip(seq_lengths, lora_assignments)):
        if seq_len == 0:
            continue

        lora_idx = lora_name_to_idx[lora_name]
        rank = lora_configs[lora_name][0]
        scaling = scalings[seq_idx]

        if rank > 0:
            x_seq = x[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, :rank, :]

            result = safe_matmul(x_seq, w_seq.T) * scaling
            output = output.at[token_offset : token_offset + seq_len, :rank].set(result)

        token_offset += seq_len

    return output


def reference_lora_b_gemm(
    x: jax.Array,
    weights: jax.Array,
    seq_lengths: List[int],
    lora_assignments: List[str],
    lora_configs: Dict[str, Tuple[int, int, int, int]],
    base_output: jax.Array,
) -> jax.Array:
    """Reference implementation of LoRA B GEMM."""
    if weights.size == 0:
        return base_output

    output = base_output.copy()
    unique_loras = sorted(set(lora_assignments))
    lora_name_to_idx = {name: idx for idx, name in enumerate(unique_loras)}

    token_offset = 0
    for seq_len, lora_name in zip(seq_lengths, lora_assignments):
        if seq_len == 0:
            continue

        lora_idx = lora_name_to_idx[lora_name]
        rank = lora_configs[lora_name][0]

        if rank > 0:
            x_seq = x[token_offset : token_offset + seq_len, :rank]
            w_seq = weights[lora_idx, :, :rank]

            result = safe_matmul(x_seq, w_seq.T)
            output = output.at[token_offset : token_offset + seq_len, :].add(result)

        token_offset += seq_len

    return output


def reference_qkv_lora(
    x: jax.Array,
    qkv_lora_a: jax.Array,
    qkv_lora_b: jax.Array,
    seq_lengths: List[int],
    lora_assignments: List[str],
    lora_configs: Dict[str, Tuple[int, int, int, int]],
    scalings: List[float],
    output_slices: Tuple[int, int, int],
    base_output: jax.Array,
) -> jax.Array:
    """Reference implementation of QKV LoRA."""
    if qkv_lora_a.size == 0:
        return base_output

    output = base_output.copy()
    unique_loras = sorted(set(lora_assignments))
    lora_name_to_idx = {name: idx for idx, name in enumerate(unique_loras)}

    token_offset = 0
    for seq_idx, (seq_len, lora_name) in enumerate(zip(seq_lengths, lora_assignments)):
        if seq_len == 0:
            continue

        lora_idx = lora_name_to_idx[lora_name]
        rank = lora_configs[lora_name][0]
        scaling = scalings[seq_idx]

        if rank > 0:
            # LoRA A pass
            x_seq = x[token_offset : token_offset + seq_len, :]
            w_a_seq = qkv_lora_a[lora_idx, : 3 * rank, :]
            intermediate = safe_matmul(x_seq, w_a_seq.T) * scaling  # (seq_len, 3*rank)

            # LoRA B pass for each slice (Q, K, V)
            for slice_idx, slice_dim in enumerate(output_slices):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank
                slice_start_output = sum(output_slices[:slice_idx])
                slice_end_output = slice_start_output + slice_dim

                inter_slice = intermediate[:, slice_start_input:slice_end_input]
                w_b_slice = qkv_lora_b[lora_idx, slice_start_output:slice_end_output, :rank]

                result = safe_matmul(inter_slice, w_b_slice.T)
                output = output.at[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ].add(result)

        token_offset += seq_len

    return output


def reference_gate_up_lora(
    x: jax.Array,
    gate_up_lora_a: jax.Array,
    gate_up_lora_b: jax.Array,
    seq_lengths: List[int],
    lora_assignments: List[str],
    lora_configs: Dict[str, Tuple[int, int, int, int]],
    scalings: List[float],
    output_dim: int,
    base_output: jax.Array,
) -> jax.Array:
    """Reference implementation of gate-up LoRA."""
    if gate_up_lora_a.size == 0:
        return base_output

    output = base_output.copy()
    unique_loras = sorted(set(lora_assignments))
    lora_name_to_idx = {name: idx for idx, name in enumerate(unique_loras)}

    token_offset = 0
    for seq_idx, (seq_len, lora_name) in enumerate(zip(seq_lengths, lora_assignments)):
        if seq_len == 0:
            continue

        lora_idx = lora_name_to_idx[lora_name]
        rank = lora_configs[lora_name][0]
        scaling = scalings[seq_idx]

        if rank > 0:
            # LoRA A pass
            x_seq = x[token_offset : token_offset + seq_len, :]
            w_a_seq = gate_up_lora_a[lora_idx, : 2 * rank, :]
            intermediate = safe_matmul(x_seq, w_a_seq.T) * scaling  # (seq_len, 2*rank)

            # LoRA B pass - process two slices (gate and up)
            for slice_idx in range(2):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank
                slice_start_output = slice_idx * output_dim
                slice_end_output = (slice_idx + 1) * output_dim

                inter_slice = intermediate[:, slice_start_input:slice_end_input]
                w_b_slice = gate_up_lora_b[lora_idx, slice_start_output:slice_end_output, :rank]

                result = safe_matmul(inter_slice, w_b_slice.T)
                output = output.at[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ].add(result)

        token_offset += seq_len

    return output


class TestBgmvLoRABackend(CustomTestCase):
    """End-to-end tests for BgmvLoRABackend with Cartesian product of conditions."""

    RTOL = 1e-3
    ATOL = 1e-3

    def setUp(self):
        """Set up common test parameters."""
        # jax.config.update("jax_platform_name", "cpu")
        np.random.seed(42)
        random.seed(42)

        self.dtype = jnp.bfloat16
        self.input_dim = 512
        self.max_seq_len = 128

        # LoRA configurations: name -> (rank, output_q, output_k, output_v)
        self.lora_configs = {
            "lora_A": (8, 256, 128, 128),
            "lora_B": (16, 256, 128, 128),
            "lora_C": (32, 256, 128, 128),
            "_NO_LORA_": (0, 4096, 1024, 1024),
        }

        self.qkv_output_slices = (256, 128, 128)
        self.gate_up_output_dim = 512

        self.mesh = create_device_mesh(
            ici_parallelism=[1, 1],
            dcn_parallelism=[1, 1],
        )

    def get_lora_batch_info_on_device(self, batch: ModelWorkerBatch):
        (
            lora_scalings,
            lora_token_indices,
        ) = device_array(
            (
                batch.lora_scalings,
                batch.lora_token_indices,
            ),
            sharding=(
                NamedSharding(self.mesh, PartitionSpec()) if jax.process_count() == 1 else None
            ),
        )
        return lora_scalings, lora_token_indices

    def generate_sequence_lengths(
        self, batch_size: int, batch_mode: BatchMode, max_len: int = None
    ) -> List[int]:
        """Generate sequence lengths based on batch mode."""
        if batch_mode == BatchMode.DECODE:
            return [1] * batch_size
        else:
            if max_len is None:
                max_len = self.max_seq_len
            return [random.randint(1, max_len) for _ in range(batch_size)]

    def create_lora_weights(
        self, lora_name: str, num_slices: int = 1, include_missing_k: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        """Create LoRA A and B weights."""
        rank, out_q, out_k, out_v = self.lora_configs[lora_name]

        if rank == 0:
            lora_a = jnp.empty((0, self.input_dim), dtype=self.dtype)

        if num_slices == 3:  # QKV
            lora_a = jnp.array(
                np.random.randn(num_slices * rank, self.input_dim) * 0.01,
                dtype=self.dtype,
            )
            total_output_dim = out_q + out_k + out_v
            lora_b = jnp.array(
                np.random.randn(total_output_dim, rank) * 0.01,
                dtype=self.dtype,
            )

            # Zero out k_proj for Qwen3 scenario
            if include_missing_k:
                lora_a = lora_a.at[rank : 2 * rank, :].set(0.0)
                lora_b = lora_b.at[out_q : out_q + out_k, :].set(0.0)

            if rank == 0:
                lora_b = jnp.empty((out_q + out_k + out_v, 0), dtype=self.dtype)

        elif num_slices == 2:  # gate-up
            lora_a = jnp.array(
                np.random.randn(num_slices * rank, self.input_dim) * 0.01,
                dtype=self.dtype,
            )
            lora_b = jnp.array(
                np.random.randn(2 * self.gate_up_output_dim, rank) * 0.01,
                dtype=self.dtype,
            )
            if rank == 0:
                lora_b = jnp.empty((2 * self.gate_up_output_dim, 0), dtype=self.dtype)
        else:  # Standard linear
            lora_a = jnp.array(
                np.random.randn(rank, self.input_dim) * 0.01,
                dtype=self.dtype,
            )
            lora_b = jnp.array(
                np.random.randn(self.input_dim, rank) * 0.01,
                dtype=self.dtype,
            )
            if rank == 0:
                lora_b = jnp.empty((self.input_dim, 0), dtype=self.dtype)

        return lora_a, lora_b

    def create_model_worker_batch(
        self, x: np.ndarray, seq_lengths: List[int], batch_mode: BatchMode
    ) -> ModelWorkerBatch:
        """Create a ForwardBatch for testing."""
        batch_size = len(seq_lengths)
        total_tokens = sum(seq_lengths)
        forward_mode = ForwardMode.EXTEND if batch_mode == BatchMode.PREFILL else ForwardMode.DECODE

        return ModelWorkerBatch(
            bid=0,
            forward_mode=forward_mode,
            input_ids=np.arange(total_tokens, dtype=jnp.int32),
            real_input_ids_len=total_tokens,
            seq_lens=np.array(seq_lengths, dtype=jnp.int32),
            out_cache_loc=np.arange(total_tokens, dtype=jnp.int32),
            req_pool_indices=np.arange(batch_size, dtype=jnp.int32),
            sampling_info=SamplingBatchInfo.generate_for_precompile(batch_size, 32000),
            positions=None,
            cache_loc=None,
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=1,
            token_ids_logprobs=None,
            extend_seq_lens=np.array(seq_lengths, dtype=np.int32),
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            real_bs=batch_size,
        )

    def stack_lora_weights(
        self, weight_list: List[jax.Array], is_lora_a: bool, max_rank: int, num_slices: int = 1
    ) -> jax.Array:
        """Stack LoRA weights from different adapters."""
        if not weight_list or all(w.size == 0 for w in weight_list):
            return jnp.empty((len(weight_list), 0, 0), dtype=self.dtype)

        first_non_empty = next((w for w in weight_list if w.size > 0), None)
        if first_non_empty is None:
            return jnp.empty((len(weight_list), 0, 0), dtype=self.dtype)

        if is_lora_a:
            final_shape = (len(weight_list), num_slices * max_rank, self.input_dim)
        else:
            output_dim = first_non_empty.shape[0]
            final_shape = (len(weight_list), output_dim, max_rank)

        stacked = jnp.zeros(final_shape, dtype=self.dtype)

        for i, weight in enumerate(weight_list):
            if weight.size > 0:
                if is_lora_a:
                    stacked = stacked.at[i, : weight.shape[0], :].set(weight)
                else:
                    stacked = stacked.at[i, :, : weight.shape[1]].set(weight)

        return stacked

    def create_test_batch(
        self,
        batch_composition: BatchComposition,
        batch_size: int,
        batch_mode: BatchMode,
        num_slices: int = 1,
        include_missing_k: bool = False,
    ) -> Tuple:
        """Create test batch with specified composition and mode."""
        seq_lengths = self.generate_sequence_lengths(batch_size, batch_mode, self.max_seq_len)

        if batch_composition == BatchComposition.UNIFORM:
            lora_assignments = ["lora_A"] * batch_size
        elif batch_composition == BatchComposition.MIXED:
            lora_names = ["lora_A", "lora_B", "lora_C", None]
            lora_assignments = [lora_names[i % len(lora_names)] for i in range(batch_size)]
        elif batch_composition == BatchComposition.SKEWED:
            num_minority = max(1, batch_size // 8)
            lora_assignments = ["lora_A"] * num_minority + ["lora_B"] * (batch_size - num_minority)
            random.shuffle(lora_assignments)
        else:
            raise ValueError(f"Unknown batch composition: {batch_composition}")

        total_seq_len = sum(seq_lengths)
        x = np.array(
            np.random.randn(total_seq_len, self.input_dim) * 0.01,
            dtype=self.dtype,
        )

        normalized_assignments = [
            name if name is not None else "_NO_LORA_" for name in lora_assignments
        ]
        unique_loras = sorted(set(normalized_assignments))

        weight_indices = [0] * len(normalized_assignments)
        lora_ranks = [0] * len(self.lora_configs)
        scalings = [0] * len(self.lora_configs)
        lora_name_to_idx = {}

        # no_lora_count=1

        for i, lora_name in enumerate(unique_loras):
            # if lora_name == "_NO_LORA_":
            lora_name_to_idx[lora_name] = i
            # else:
            #    lora_name_to_idx[lora_name] = no_lora_count
            #    no_lora_count+=1

        for i, lora_name in enumerate(normalized_assignments):
            weight_indices[i] = lora_name_to_idx[lora_name]
            lora_ranks[weight_indices[i]] = self.lora_configs[lora_name][0]
            scalings[weight_indices[i]] = 1.0

        scalings_seq = [1.0 for _ in normalized_assignments]

        weights = {}
        for lora_name in unique_loras:
            weights[lora_name] = self.create_lora_weights(lora_name, num_slices, include_missing_k)

        model_worker_batch = self.create_model_worker_batch(x, seq_lengths, batch_mode)

        return (
            jnp.array(x),
            weights,
            model_worker_batch,
            seq_lengths,
            normalized_assignments,
            weight_indices,
            lora_ranks,
            scalings,
            scalings_seq,
        )

    # === Test run_lora_a_gemm and run_lora_b_gemm ===

    def test_lora_a_b_gemm(self):
        """Test run_lora_a_gemm and run_lora_b_gemm with Cartesian product."""
        batch_sizes = [1, 2, 16, 64]
        compositions = [BatchComposition.UNIFORM, BatchComposition.MIXED, BatchComposition.SKEWED]
        modes = [BatchMode.PREFILL, BatchMode.DECODE]

        for batch_size, composition, mode in itertools.product(batch_sizes, compositions, modes):
            with self.subTest(
                batch_size=batch_size, composition=composition.value, mode=mode.value
            ):
                (
                    x,
                    weights,
                    model_worker_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                    scalings_seq,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=1)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(model_worker_batch, weight_indices, lora_ranks, scalings)
                lora_scalings_device, lora_token_indces_device = self.get_lora_batch_info_on_device(
                    model_worker_batch
                )

                # Stack weights
                lora_a_weights = [weights[name][0] for name in sorted(weights.keys())]
                stacked_lora_a = self.stack_lora_weights(
                    lora_a_weights, is_lora_a=True, max_rank=max_rank, num_slices=1
                )

                lora_b_weights = [weights[name][1] for name in sorted(weights.keys())]
                stacked_lora_b = self.stack_lora_weights(
                    lora_b_weights, is_lora_a=False, max_rank=max_rank
                )

                lora_a_output_sharding = get_lora_a_output_sharding("q_proj", self.mesh)
                lora_b_output_sharding = get_lora_b_output_sharding("q_proj", self.mesh)

                # Test run_lora_a_gemm
                backend_a_output = backend.run_lora_a_gemm(
                    x,
                    stacked_lora_a,
                    lora_a_output_sharding,
                    lora_scalings_device,
                    lora_token_indces_device,
                )
                reference_a_output = reference_lora_a_gemm(
                    x,
                    stacked_lora_a,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings_seq,
                )

                # Compare only valid portions for shrink
                token_offset = 0
                for seq_len, lora_name in zip(seq_lengths, lora_assignments):
                    if seq_len > 0:
                        rank = self.lora_configs[lora_name][0]
                        if rank > 0:
                            backend_seq = backend_a_output[
                                token_offset : token_offset + seq_len, :rank
                            ]
                            reference_seq = reference_a_output[
                                token_offset : token_offset + seq_len, :rank
                            ]
                            np.testing.assert_allclose(
                                np.array(backend_seq),
                                np.array(reference_seq),
                                rtol=self.RTOL,
                                atol=self.ATOL,
                                err_msg=f"LoRA A failed for {composition.value}, {mode.value}, batch_size={batch_size}",
                                strict=True,
                            )
                    token_offset += seq_len

                # Test run_lora_b_gemm
                base_output = jnp.ones((x.shape[0], self.input_dim), dtype=self.dtype)
                backend_b_output = backend.run_lora_b_gemm(
                    reference_a_output,
                    stacked_lora_b,
                    base_output,
                    lora_b_output_sharding,
                    lora_token_indces_device,
                )
                reference_b_output = reference_lora_b_gemm(
                    reference_a_output,
                    stacked_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    base_output,
                )

                np.testing.assert_allclose(
                    np.array(backend_b_output),
                    np.array(reference_b_output),
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    err_msg=f"LoRA B failed for {composition.value}, {mode.value}, batch_size={batch_size}",
                    strict=True,
                )

    # === Test run_qkv_lora ===

    def test_qkv_lora(self):
        """Test run_qkv_lora with Cartesian product."""
        batch_sizes = [1, 2, 16, 64]
        compositions = [BatchComposition.UNIFORM, BatchComposition.MIXED, BatchComposition.SKEWED]
        modes = [BatchMode.PREFILL, BatchMode.DECODE]

        for batch_size, composition, mode in itertools.product(batch_sizes, compositions, modes):
            with self.subTest(
                batch_size=batch_size, composition=composition.value, mode=mode.value
            ):
                (
                    x,
                    weights,
                    model_worker_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                    scalings_seq,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=3)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(model_worker_batch, weight_indices, lora_ranks, scalings)
                lora_scalings_device, lora_token_indces_device = self.get_lora_batch_info_on_device(
                    model_worker_batch
                )

                # Stack weights
                qkv_lora_a_weights = [weights[name][0] for name in sorted(weights.keys())]
                stacked_qkv_lora_a = self.stack_lora_weights(
                    qkv_lora_a_weights, is_lora_a=True, max_rank=max_rank, num_slices=3
                )

                qkv_lora_b_weights = [weights[name][1] for name in sorted(weights.keys())]
                stacked_qkv_lora_b = self.stack_lora_weights(
                    qkv_lora_b_weights, is_lora_a=False, max_rank=max_rank
                )

                base_output = jnp.ones((x.shape[0], sum(self.qkv_output_slices)), dtype=self.dtype)

                lora_a_output_sharding = get_lora_a_output_sharding("qkv_proj", self.mesh)
                lora_b_output_sharding = get_lora_b_output_sharding("qkv_proj", self.mesh)

                # Test run_qkv_lora
                backend_output = backend.run_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    self.qkv_output_slices,
                    base_output,
                    lora_a_output_sharding,
                    lora_b_output_sharding,
                    lora_scalings_device,
                    lora_token_indces_device,
                )
                reference_output = reference_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings_seq,
                    self.qkv_output_slices,
                    base_output,
                )

                np.testing.assert_allclose(
                    np.array(backend_output),
                    np.array(reference_output),
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    err_msg=f"QKV LoRA failed for {composition.value}, {mode.value}, batch_size={batch_size}",
                    strict=True,
                )

    # === Test run_gate_up_lora ===

    def test_gate_up_lora(self):
        """Test run_gate_up_lora with Cartesian product."""
        batch_sizes = [1, 2, 16, 64]
        compositions = [BatchComposition.UNIFORM, BatchComposition.MIXED, BatchComposition.SKEWED]
        modes = [BatchMode.PREFILL, BatchMode.DECODE]

        for batch_size, composition, mode in itertools.product(batch_sizes, compositions, modes):
            with self.subTest(
                batch_size=batch_size, composition=composition.value, mode=mode.value
            ):
                (
                    x,
                    weights,
                    model_worker_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                    scalings_seq,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=2)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(model_worker_batch, weight_indices, lora_ranks, scalings)
                lora_scalings_device, lora_token_indces_device = self.get_lora_batch_info_on_device(
                    model_worker_batch
                )

                # Stack weights
                gate_up_lora_a_weights = [weights[name][0] for name in sorted(weights.keys())]
                stacked_gate_up_lora_a = self.stack_lora_weights(
                    gate_up_lora_a_weights, is_lora_a=True, max_rank=max_rank, num_slices=2
                )

                gate_up_lora_b_weights = [weights[name][1] for name in sorted(weights.keys())]
                stacked_gate_up_lora_b = self.stack_lora_weights(
                    gate_up_lora_b_weights, is_lora_a=False, max_rank=max_rank
                )

                base_output = jnp.ones((x.shape[0], 2 * self.gate_up_output_dim), dtype=self.dtype)

                lora_a_output_sharding = get_lora_a_output_sharding("gate_up", self.mesh)
                lora_b_output_sharding = get_lora_b_output_sharding("gate_up", self.mesh)

                # Test run_gate_up_lora
                backend_output = backend.run_gate_up_lora(
                    x,
                    stacked_gate_up_lora_a,
                    stacked_gate_up_lora_b,
                    base_output,
                    lora_a_output_sharding,
                    lora_b_output_sharding,
                    lora_scalings_device,
                    lora_token_indces_device,
                )
                reference_output = reference_gate_up_lora(
                    x,
                    stacked_gate_up_lora_a,
                    stacked_gate_up_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings_seq,
                    self.gate_up_output_dim,
                    base_output,
                )

                np.testing.assert_allclose(
                    np.array(backend_output),
                    np.array(reference_output),
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    err_msg=f"Gate-up LoRA failed for {composition.value}, {mode.value}, batch_size={batch_size}",
                    strict=True,
                )

    # === Test QKV with missing k_proj ===

    def test_qkv_missing(self):
        """Test QKV operations with missing k_proj (Qwen3 scenario)."""
        batch_sizes = [1, 2, 16, 64]
        compositions = [BatchComposition.UNIFORM, BatchComposition.MIXED, BatchComposition.SKEWED]
        modes = [BatchMode.PREFILL, BatchMode.DECODE]

        for batch_size, composition, mode in itertools.product(batch_sizes, compositions, modes):
            with self.subTest(
                batch_size=batch_size, composition=composition.value, mode=mode.value
            ):
                (
                    x,
                    weights,
                    model_worker_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                    scalings_seq,
                ) = self.create_test_batch(
                    composition, batch_size, mode, num_slices=3, include_missing_k=True
                )

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(model_worker_batch, weight_indices, lora_ranks, scalings)
                lora_scalings_device, lora_token_indces_device = self.get_lora_batch_info_on_device(
                    model_worker_batch
                )

                # Stack weights
                qkv_lora_a_weights = [weights[name][0] for name in sorted(weights.keys())]
                stacked_qkv_lora_a = self.stack_lora_weights(
                    qkv_lora_a_weights, is_lora_a=True, max_rank=max_rank, num_slices=3
                )

                qkv_lora_b_weights = [weights[name][1] for name in sorted(weights.keys())]
                stacked_qkv_lora_b = self.stack_lora_weights(
                    qkv_lora_b_weights, is_lora_a=False, max_rank=max_rank
                )

                base_output = jnp.ones((x.shape[0], sum(self.qkv_output_slices)), dtype=self.dtype)

                lora_a_output_sharding = get_lora_a_output_sharding("qkv_proj", self.mesh)
                lora_b_output_sharding = get_lora_b_output_sharding("qkv_proj", self.mesh)

                # Test run_qkv_lora with missing k_proj
                backend_output = backend.run_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    self.qkv_output_slices,
                    base_output,
                    lora_a_output_sharding,
                    lora_b_output_sharding,
                    lora_scalings_device,
                    lora_token_indces_device,
                )
                reference_output = reference_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings_seq,
                    self.qkv_output_slices,
                    base_output,
                )

                np.testing.assert_allclose(
                    np.array(backend_output),
                    np.array(reference_output),
                    rtol=self.RTOL,
                    atol=self.ATOL,
                    err_msg=f"QKV missing k_proj failed for {composition.value}, {mode.value}, batch_size={batch_size}",
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()
