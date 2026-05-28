import jax
from flax import nnx
from jax.sharding import NamedSharding

from sgl_jax.srt.lora.utils import LoRABatchPlan
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


class BaseLoRABackend(nnx.Module):
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        max_loras_per_batch: number of LoRA buffer slots available to the backend.
                             Dynamic LoRA reserves one slot for base-model requests.
    """

    def __init__(self, max_loras_per_batch: int):
        super().__init__()
        self.max_loras_per_batch = max_loras_per_batch

    def run_lora_a_gemm(
        self,
        x: jax.Array,
        weights: jax.Array,
        sharding: NamedSharding,
        scalings: jax.Array,
        token_indices: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run gemm of lora a modules with current backend.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, r, input_dim), r is lora rank,
                      usually input_dim is much larger than r
             sharding: lora_a_output sharding
             scalings: vector with shape (s,), alpha/rank
             token_indices: vector with shape (s,)
        Returns:
             result with shape (s, r)
        """
        raise NotImplementedError

    def run_lora_b_gemm(
        self,
        x: jax.Array,
        weights: jax.Array,
        base_output: jax.Array,
        sharding: NamedSharding,
        token_indices: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run gemm of lora b modules with current backend.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
             base_output: base layer output with shape (s, output_dim)
             sharding: lora_b_output sharding
             token_indices: vector with shape (s,)
        Returns:
             result with shape (s, output_dim)
        """
        raise NotImplementedError

    def run_qkv_lora(
        self,
        x: jax.Array,
        qkv_lora_a: jax.Array,
        qkv_lora_b: jax.Array | tuple[jax.Array],
        output_slices: tuple,
        base_output: jax.Array,
        lora_a_output_sharding: NamedSharding,
        lora_b_output_sharding: NamedSharding,
        scalings: jax.Array,
        token_indices: jax.Array,
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
            output_slices: a fixed tuple which has three item, (output_dim_q, output_dim_kv, output_dim_kv)
            base_output: base layer output with shape (s, output_dim_q + 2 * output_dim_kv)
            lora_a_output_sharding: lora_a_output sharding
            lora_b_output_sharding: lora_b_output sharding
            scalings: vector with shape (s,), alpha/rank
            token_indices: vector with shape (s,)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        raise NotImplementedError

    def run_gate_up_lora(
        self,
        x: jax.Array,
        gate_up_lora_a: jax.Array,
        gate_up_lora_b: jax.Array | tuple[jax.Array],
        base_output: jax.Array,
        lora_a_output_sharding: NamedSharding,
        lora_b_output_sharding: NamedSharding,
        scalings: jax.Array,
        token_indices: jax.Array,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Run the lora pass for gate_up_proj.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for gate_up_proj.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
            base_output: (s, 2 * output_dim)
            lora_a_output_sharding: lora_a_output sharding
            lora_b_output_sharding: lora_b_output sharding
            scalings: vector with shape (s,), alpha/rank
            token_indices: vector with shape (s,)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        raise NotImplementedError

    def prepare_lora_batch(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_plan: LoRABatchPlan,
    ):
        """Prepare the lora weights and batch info for current forward batch.

        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.

        Args:
            model_worker_batch: the batch object for current forward pass.
            batch_plan: CPU-side request-to-slot LoRA metadata to expand.
        """
        raise NotImplementedError
