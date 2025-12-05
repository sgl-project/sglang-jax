"""Configuration utilities for MoE benchmark."""

from dataclasses import dataclass


@dataclass
class MoEBenchmarkConfig:
    """Configuration for MoE benchmark."""

    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    renormalize_topk_logits: bool = True
    dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"

    # Distributed config
    ep_size: int = 1
    tp_size: int = 1

    @classmethod
    def from_model_path(
        cls, model_path: str, ep_size: int = 1, tp_size: int = 1
    ) -> "MoEBenchmarkConfig":
        """
        Load configuration from model path using AutoConfig.from_pretrained().

        Downloads from HuggingFace if needed, or loads from local models directory.

        Args:
            model_path: Path or name of HuggingFace model
            ep_size: Expert parallel size
            tp_size: Total number of devices to use

        Returns:
            MoEBenchmarkConfig instance
        """
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Extract MoE-specific parameters with fallbacks
        num_experts = getattr(
            hf_config,
            "num_experts",
            getattr(hf_config, "num_local_experts", 8),
        )

        num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 2)

        intermediate_size = getattr(
            hf_config,
            "moe_intermediate_size",
            getattr(hf_config, "intermediate_size", 2048),
        )

        activation = getattr(hf_config, "hidden_act", "silu")
        renormalize = getattr(hf_config, "norm_topk_prob", True)

        return cls(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hf_config.hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            renormalize_topk_logits=renormalize,
            ep_size=ep_size,
            tp_size=tp_size,
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) must be <= num_experts ({self.num_experts})"
            )

        # Check activation is supported
        supported_activations = ["silu", "gelu", "swigluoai"]
        if self.activation not in supported_activations:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. Supported: {supported_activations}"
            )

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"MoEBenchmarkConfig(\n"
            f"  num_experts={self.num_experts},\n"
            f"  num_experts_per_tok={self.num_experts_per_tok},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  activation={self.activation},\n"
            f"  renormalize_topk_logits={self.renormalize_topk_logits},\n"
            f"  ep_size={self.ep_size},\n"
            f"  tp_size={self.tp_size}\n"
            f")"
        )
