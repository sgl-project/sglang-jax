"""End-to-end tests for BgmvLoRABackend with Cartesian product combinations."""

import itertools
import random
import unittest
from enum import Enum
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.lora.backend.bgmv_backend import BgmvLoRABackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.test.test_utils import CustomTestCase


def safe_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Matrix multiplication with mixed precision handling for bfloat16."""
    result = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))
    return result.astype(a.dtype)


class BatchComposition(Enum):
    UNIFORM = "uniform"
    MIXED = "mixed"
    SKEWED = "skewed"


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


class TestBgmvLoRABackendE2E(CustomTestCase):
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
        }

        self.qkv_output_slices = (256, 128, 128)
        self.gate_up_output_dim = 512

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

        elif num_slices == 2:  # gate-up
            lora_a = jnp.array(
                np.random.randn(num_slices * rank, self.input_dim) * 0.01,
                dtype=self.dtype,
            )
            lora_b = jnp.array(
                np.random.randn(2 * self.gate_up_output_dim, rank) * 0.01,
                dtype=self.dtype,
            )
        else:  # Standard linear
            lora_a = jnp.array(
                np.random.randn(rank, self.input_dim) * 0.01,
                dtype=self.dtype,
            )
            lora_b = jnp.array(
                np.random.randn(self.input_dim, rank) * 0.01,
                dtype=self.dtype,
            )

        return lora_a, lora_b

    def create_forward_batch(self, seq_lengths: List[int], batch_mode: BatchMode) -> ForwardBatch:
        """Create a ForwardBatch for testing."""
        batch_size = len(seq_lengths)
        total_tokens = sum(seq_lengths)
        forward_mode = ForwardMode.EXTEND if batch_mode == BatchMode.PREFILL else ForwardMode.DECODE

        return ForwardBatch(
            bid=0,
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=jnp.arange(total_tokens, dtype=jnp.int32),
            req_pool_indices=jnp.arange(batch_size, dtype=jnp.int32),
            seq_lens=jnp.array(seq_lengths, dtype=jnp.int32),
            out_cache_loc=jnp.arange(total_tokens, dtype=jnp.int32),
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
            lora_names = ["lora_A", "lora_B", "lora_C"]
            lora_assignments = [lora_names[i % len(lora_names)] for i in range(batch_size)]
        elif batch_composition == BatchComposition.SKEWED:
            num_minority = max(1, batch_size // 8)
            lora_assignments = ["lora_A"] * num_minority + ["lora_B"] * (batch_size - num_minority)
            random.shuffle(lora_assignments)
        else:
            raise ValueError(f"Unknown batch composition: {batch_composition}")

        total_seq_len = sum(seq_lengths)
        x = jnp.array(
            np.random.randn(total_seq_len, self.input_dim) * 0.01,
            dtype=self.dtype,
        )

        unique_loras = sorted(set(lora_assignments))
        lora_name_to_idx = {name: idx for idx, name in enumerate(unique_loras)}

        weights = {}
        for lora_name in unique_loras:
            weights[lora_name] = self.create_lora_weights(lora_name, num_slices, include_missing_k)

        weight_indices = [lora_name_to_idx[name] for name in lora_assignments]
        lora_ranks = [self.lora_configs[name][0] for name in lora_assignments]
        scalings = [1.0 for _ in lora_assignments]

        forward_batch = self.create_forward_batch(seq_lengths, batch_mode)

        return (
            x,
            weights,
            forward_batch,
            seq_lengths,
            lora_assignments,
            weight_indices,
            lora_ranks,
            scalings,
        )

    # === Test run_lora_a_gemm and run_lora_b_gemm ===

    def test_lora_a_b_gemm_cartesian(self):
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
                    forward_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=1)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(forward_batch, weight_indices, lora_ranks, scalings)

                # Stack weights
                lora_a_weights = [weights[name][0] for name in sorted(weights.keys())]
                stacked_lora_a = self.stack_lora_weights(
                    lora_a_weights, is_lora_a=True, max_rank=max_rank, num_slices=1
                )

                lora_b_weights = [weights[name][1] for name in sorted(weights.keys())]
                stacked_lora_b = self.stack_lora_weights(
                    lora_b_weights, is_lora_a=False, max_rank=max_rank
                )

                # Test run_lora_a_gemm
                backend_a_output = backend.run_lora_a_gemm(x, stacked_lora_a)
                reference_a_output = reference_lora_a_gemm(
                    x, stacked_lora_a, seq_lengths, lora_assignments, self.lora_configs, scalings
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
                    reference_a_output, stacked_lora_b, base_output
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

    def test_qkv_lora_cartesian(self):
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
                    forward_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=3)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(forward_batch, weight_indices, lora_ranks, scalings)

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

                # Test run_qkv_lora
                backend_output = backend.run_qkv_lora(
                    x, stacked_qkv_lora_a, stacked_qkv_lora_b, self.qkv_output_slices, base_output
                )
                reference_output = reference_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings,
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

    def test_gate_up_lora_cartesian(self):
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
                    forward_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                ) = self.create_test_batch(composition, batch_size, mode, num_slices=2)

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(forward_batch, weight_indices, lora_ranks, scalings)

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

                # Test run_gate_up_lora
                backend_output = backend.run_gate_up_lora(
                    x, stacked_gate_up_lora_a, stacked_gate_up_lora_b, base_output
                )
                reference_output = reference_gate_up_lora(
                    x,
                    stacked_gate_up_lora_a,
                    stacked_gate_up_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings,
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

    def test_qkv_missing_projections(self):
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
                    forward_batch,
                    seq_lengths,
                    lora_assignments,
                    weight_indices,
                    lora_ranks,
                    scalings,
                ) = self.create_test_batch(
                    composition, batch_size, mode, num_slices=3, include_missing_k=True
                )

                max_rank = max(self.lora_configs[name][0] for name in weights.keys())
                backend = BgmvLoRABackend(max_loras_per_batch=len(weights), max_lora_rank=max_rank)
                backend.prepare_lora_batch(forward_batch, weight_indices, lora_ranks, scalings)

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

                # Test run_qkv_lora with missing k_proj
                backend_output = backend.run_qkv_lora(
                    x, stacked_qkv_lora_a, stacked_qkv_lora_b, self.qkv_output_slices, base_output
                )
                reference_output = reference_qkv_lora(
                    x,
                    stacked_qkv_lora_a,
                    stacked_qkv_lora_b,
                    seq_lengths,
                    lora_assignments,
                    self.lora_configs,
                    scalings,
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
