"""Fused MoE layer using optimized TPU kernel."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe


def _get_default_tile_sizes(hidden_size: int, intermediate_size: int) -> dict[str, int]:
    """
    Select appropriate tile sizes based on model dimensions.

    These values are derived from benchmarking in the test suite and optimized
    for TPU performance with different model sizes.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: MoE intermediate (FFN) dimension

    Returns:
        Dictionary containing tile size parameters for the fused kernel
    """
    if hidden_size >= 4096:
        # Large models (e.g., Qwen 2.5B)
        return {
            "bt": 64,
            "bf": 768,
            "bd1": 2048,
            "bd2": 2048,
            "btc": 64,
            "bfc": 768,
            "bd1c": 2048,
            "bd2c": 2048,
        }
    elif hidden_size >= 2048:
        # Medium models (e.g., Qwen 30B A3B)
        return {
            "bt": 16,
            "bf": 384,
            "bd1": 512,
            "bd2": 512,
            "btc": 16,
            "bfc": 384,
            "bd1c": 256,
            "bd2c": 256,
        }
    else:
        # Small models
        return {
            "bt": 32,
            "bf": 512,
            "bd1": 512,
            "bd2": 512,
            "btc": 32,
            "bfc": 256,
            "bd1c": 256,
            "bd2c": 256,
        }


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1 is 4D (num_experts, 2, hidden_size, intermediate_size)
      where dimension 2 contains [gate_proj, up_proj]
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        config: Model configuration
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

    def __init__(
        self,
        config,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        renormalize_topk_logits: bool = False,
        # Tile size parameters - auto-selected if None
        bt: int | None = None,
        bf: int | None = None,
        bd1: int | None = None,
        bd2: int | None = None,
        btc: int | None = None,
        bfc: int | None = None,
        bd1c: int | None = None,
        bd2c: int | None = None,
    ):
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.renormalize_topk_logits = renormalize_topk_logits
        self.mesh = mesh

        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        # Auto-select tile sizes if not provided
        if any(param is None for param in [bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c]):
            default_sizes = _get_default_tile_sizes(config.hidden_size, intermediate_dim)
            bt = bt or default_sizes["bt"]
            bf = bf or default_sizes["bf"]
            bd1 = bd1 or default_sizes["bd1"]
            bd2 = bd2 or default_sizes["bd2"]
            btc = btc or default_sizes["btc"]
            bfc = bfc or default_sizes["bfc"]
            bd1c = bd1c or default_sizes["bd1c"]
            bd2c = bd2c or default_sizes["bd2c"]

        self.bt = bt
        self.bf = bf
        self.bd1 = bd1
        self.bd2 = bd2
        self.btc = btc
        self.bfc = bfc
        self.bd1c = bd1c
        self.bd2c = bd2c

        # Initialize weights in fused format
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, 2, config.hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, intermediate_dim, config.hidden_size),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

    def __call__(self, hidden_states: jax.Array, router_logits: jax.Array) -> jax.Array:
        """
        Forward pass through the fused MoE layer.

        Args:
            hidden_states: Input tokens, shape (num_tokens, hidden_size) or
                          (batch_size, seq_len, hidden_size)
            router_logits: Router output logits, shape (num_tokens, num_experts)
                          Note: Should be raw logits, not after softmax or top-k

        Returns:
            MoE layer output, same shape as hidden_states
        """
        assert hidden_states.ndim == 2

        hidden_states = jax.sharding.reshard(hidden_states, P("tensor", None))
        router_logits = jax.sharding.reshard(router_logits, P("tensor", None))

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            gating_output=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize_topk_logits=self.renormalize_topk_logits,
            act_fn=self.activation,
            # Tile sizes
            bt=self.bt,
            bf=self.bf,
            bd1=self.bd1,
            bd2=self.bd2,
            btc=self.btc,
            bfc=self.bfc,
            bd1c=self.bd1c,
            bd2c=self.bd2c,
            # Optional parameters (not used in basic case)
            subc_quant_wsz=None,
            w1_scale=None,
            w2_scale=None,
            b1=None,
            b2=None,
            ep_axis_name="tensor",
            # tp_axis_name="data",
        )

        final_output = jax.sharding.reshard(output, P(None))
        return final_output
