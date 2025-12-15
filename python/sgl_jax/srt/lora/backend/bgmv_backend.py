import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding

from sgl_jax.srt.lora.backend.base_backend import BaseLoRABackend
from sgl_jax.srt.lora.utils import LoRABatchInfo
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

MIN_CHUNK_SIZE = 16


class BatchInfo(nnx.Variable):
    pass


class BgmvLoRABackend(BaseLoRABackend):
    """
    Bgmv LoRA backend using batched grouped matrix-vector multiplication.
    """

    name = "bgmv"

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
    ):
        super().__init__(max_loras_per_batch)
        self.max_lora_rank = max_lora_rank

        # Initialize with dummy arrays to ensure consistent PyTree structure for JIT
        # using empty arrays (size 0) is sufficient for structure matching
        dummy_arr = jnp.array([], dtype=jnp.int32)
        dummy_scalings = jnp.array([], dtype=jnp.float32)
        self.batch_info = BatchInfo(
            LoRABatchInfo(
                scalings=dummy_scalings,
                token_lora_indices=dummy_arr,
                lora_ranks=dummy_arr,
            )
        )

    def run_lora_a_gemm(
        self,
        x: jax.Array,
        weights: jax.Array,
        sharding: NamedSharding,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run gemm of lora a modules with current backend.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, r, input_dim), r is lora rank,
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, r)
        """
        info = self.batch_info.value
        return shrink(x, weights, info.token_lora_indices, info.scalings, sharding).astype(x.dtype)

    def run_lora_b_gemm(
        self,
        x: jax.Array,
        weights: jax.Array,
        base_output: jax.Array,
        sharding: NamedSharding,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run gemm of lora b modules with current backend.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
             base_output: (s, output_dim)
        Returns:
             result with shape (s, output_dim)
        """
        info = self.batch_info.value
        return jnp.add(
            base_output,
            expand(
                x,
                weights,
                info.token_lora_indices,
                (weights.shape[1],),
                self.max_lora_rank,
                sharding,
            ).astype(x.dtype),
        )

    def run_qkv_lora(
        self,
        x: jax.Array,
        qkv_lora_a: jax.Array,
        qkv_lora_b: jax.Array | tuple[jax.Array],
        output_slices: tuple,
        base_output: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
            output_slices: a fixed tuple which has three items, (output_dim_q, output_dim_kv, output_dim_kv)
            base_output: (s, 2 * output_dim)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        if isinstance(qkv_lora_b, tuple):
            q_lora_b = qkv_lora_b[0]
            kv_lora_b = qkv_lora_b[1]
            reshaped_q_lora_b = jnp.squeeze(q_lora_b, axis=0)
            swapped_kv_lora_b = jnp.transpose(kv_lora_b, (1, 0, 2, 3))
            reshaped_kv_lora_b = swapped_kv_lora_b.reshape(
                swapped_kv_lora_b.shape[0], -1, swapped_kv_lora_b.shape[-1]
            )
            qkv_lora_b_concated = jnp.concatenate([reshaped_q_lora_b, reshaped_kv_lora_b], axis=1)
        else:
            qkv_lora_b_concated = qkv_lora_b

        # (s, 3*r)
        info = self.batch_info.value
        lora_a_output = bgmv_shrink(x, qkv_lora_a, info.token_lora_indices, info.scalings)

        return jnp.add(
            base_output,
            expand(
                lora_a_output,
                qkv_lora_b_concated,
                info.token_lora_indices,
                output_slices,
                self.max_lora_rank,
            ).astype(x.dtype),
        )

    def run_gate_up_lora(
        self,
        x: jax.Array,
        gate_up_lora_a: jax.Array,
        gate_up_lora_b: jax.Array | tuple[jax.Array],
        base_output: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run the lora pass for gate_up_proj.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
            base_output: (s, 2 * output_dim)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        if isinstance(gate_up_lora_b, tuple):
            gate_up_lora_b_concated = jnp.concat([gate_up_lora_b[0], gate_up_lora_b[1]], axis=1)
        else:
            gate_up_lora_b_concated = gate_up_lora_b

        # (s, 2*r)
        info = self.batch_info.value
        lora_a_output = bgmv_shrink(x, gate_up_lora_a, info.token_lora_indices, info.scalings)

        return jnp.add(
            base_output,
            expand(
                lora_a_output,
                gate_up_lora_b_concated,
                info.token_lora_indices,
                (gate_up_lora_b_concated.shape[1] // 2, gate_up_lora_b_concated.shape[1] // 2),
                self.max_lora_rank,
            ).astype(x.dtype),
        )

    def prepare_lora_batch(
        self,
        model_worker_batch: ModelWorkerBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
    ):
        lora_ranks_bs = []
        scalings_bs = []
        for indice in weight_indices:
            if indice != -1:
                lora_ranks_bs.append(lora_ranks[indice])
                scalings_bs.append(scalings[indice])
            else:
                lora_ranks_bs.append(0)
                scalings_bs.append(0.0)

        assert len(model_worker_batch.seq_lens) == len(weight_indices)
        assert len(model_worker_batch.seq_lens) == len(lora_ranks_bs)
        assert len(model_worker_batch.seq_lens) == len(scalings_bs)

        target_len = model_worker_batch.input_ids.shape[0]

        if model_worker_batch.forward_mode == ForwardMode.EXTEND:
            scalings_cpu = np.repeat(
                np.array(scalings_bs, dtype=np.float32), model_worker_batch.seq_lens
            )
            token_lora_indices_cpu = np.repeat(
                np.array(weight_indices, dtype=np.int32), model_worker_batch.seq_lens
            )
            lora_ranks_cpu = np.repeat(
                np.array(lora_ranks_bs, dtype=np.int32), model_worker_batch.seq_lens
            )

            num_to_pad = target_len - jnp.sum(model_worker_batch.seq_lens)

            padded_scalings_cpu = scalings_cpu
            padded_token_lora_indices_cpu = token_lora_indices_cpu
            padded_lora_ranks_cpu = lora_ranks_cpu

            if num_to_pad > 0:
                padded_scalings_cpu = np.pad(
                    scalings_cpu, [0, num_to_pad], mode="constant", constant_values=0.0
                )
                padded_token_lora_indices_cpu = np.pad(
                    token_lora_indices_cpu, [0, num_to_pad], mode="constant", constant_values=0
                )
                padded_lora_ranks_cpu = np.pad(
                    lora_ranks_cpu, [0, num_to_pad], mode="constant", constant_values=0
                )
        elif model_worker_batch.forward_mode == ForwardMode.DECODE:
            padded_scalings_cpu = np.array(scalings_bs, dtype=np.float32)
            padded_token_lora_indices_cpu = np.array(weight_indices, dtype=np.int32)
            padded_lora_ranks_cpu = np.array(lora_ranks_bs, dtype=np.int32)

        self.batch_info = LoRABatchInfo(
            scalings=jnp.array(padded_scalings_cpu, dtype=jnp.float32),
            token_lora_indices=jnp.array(padded_token_lora_indices_cpu, dtype=jnp.int32),
            lora_ranks=jnp.array(padded_lora_ranks_cpu, dtype=jnp.int32),
        )


def shrink(
    x: jax.Array,  # (num_tokens, in_features)
    lora_a_stacked: jax.Array,  # (max_loras, num_slices*max_lora_rank, in_features)
    token_lora_indices: jax.Array,  # (num_tokens,)
    scalings: jax.Array,  # (num_tokens,)
    sharding: NamedSharding,
):
    return bgmv_shrink(x, lora_a_stacked, token_lora_indices, sharding, scalings)


def expand(
    x: jax.Array,  # (num_tokens, num_slices*max_lora_rank)
    lora_b_stacked: jax.Array,  # (max_loras, sum(output_slices), max_lora_rank)
    token_lora_indices: jax.Array,  # (num_tokens,)
    output_slices: tuple,
    max_lora_rank: int,
    sharding: NamedSharding,
):
    """Optimized: Loop with slicing."""
    # y_shape = output_shape
    # y = y.reshape(-1, y.shape[-1])
    offset_output = 0
    offset_rank = 0
    output = jnp.zeros((x.shape[0], sum(output_slices)), dtype=x.dtype)

    num_slices = len(output_slices)

    for slice_idx in range(num_slices):
        # Slice the buffer
        x_slice = x[:, offset_rank : offset_rank + max_lora_rank]

        # Slice lora_b
        lora_b_slice = lora_b_stacked[
            :, offset_output : offset_output + output_slices[slice_idx], :
        ]
        # lora_b_slice = jnp.expand_dims(lora_b_slice, axis=1)

        output = bgmv_expand_slice(
            x_slice,
            lora_b_slice,
            output,
            token_lora_indices,
            offset_output,
            output_slices[slice_idx],
            sharding,
        )

        offset_output += output_slices[slice_idx]
        offset_rank += max_lora_rank

    return output


def bgmv_shrink(
    inputs,
    lora_weights,
    lora_indices,
    sharding: NamedSharding,
    scaling: float = 1.0,
):
    """
    Shrink operation: maps input to low-rank space.

    Args:
        inputs: (s, input_dim)
        lora_weights: (num_lora, c * r, input_dim), c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
        lora_indices: (num_tokens)
    Returns:
        [s, c * r]
    """
    if isinstance(scaling, jax.Array) and scaling.ndim == 1:
        scaling = scaling[:, jnp.newaxis]
    return scaling * bgmv_jax(inputs, lora_weights, lora_indices, sharding)


def bgmv_expand_slice(
    inputs,  # [num_tokens, lora_rank]
    lora_weights,  # [num_loras, out_features, lora_rank]
    output_array,  # [num_tokens, total_out_features]
    lora_indices,  # [num_tokens]
    slice_offset: int,
    slice_size: int,
    sharding: NamedSharding,
):
    """
    Expand operation: maps from low-rank space to output space.

    Args:
        inputs: [num_tokens, lora_rank]
        lora_weights: [num_loras, 1, out_features, lora_rank]
        output_tensor: [num_tokens, total_out_features]
        lora_indices: [num_tokens]
    """
    # if len(lora_weights.shape) == 4:
    #     lora_weights = jnp.squeeze(lora_weights, axis=1)

    outputs = bgmv_jax(inputs, lora_weights, lora_indices, sharding)

    # Pad the outputs
    pad_left = slice_offset
    pad_right = output_array.shape[-1] - (slice_offset + slice_size)
    outputs = jnp.pad(outputs, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

    if output_array is not None:
        return output_array + outputs
    else:
        return outputs


def bgmv_jax(
    inputs,  # (num_tokens, input_dim)
    loras,  # (num_lora, output_dim, input_dim)
    idxs,  # (num_tokens)
    sharding,
):
    """
    Batched grouped matrix-vector multiplication.
    For each token, select the corresponding LoRA and apply matrix multiplication.
    """
    return jnp.einsum(
        "td,tX,Xld->tl",
        inputs,
        jax.nn.one_hot(idxs, loras.shape[0], dtype=inputs.dtype),
        loras,
        out_sharding=sharding,
    )
