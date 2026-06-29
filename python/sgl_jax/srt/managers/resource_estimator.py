"""Resource estimation for shape-aware best-fit DP inference scheduling.

https://jax-ml.github.io/scaling-book/transformers/ provides estimation formulas for
training. This module adapts them for inference:
    * FLOPs estimation for prefill and decode is based on peak usage.
    * HBM estimation for KV cache is based on total usage.

This module also provides stranding calculations for best-fit scheduling score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TPUCapacity:
    """TPU hardware capacity specifications.

    Attributes:
        name: TPU generation name (e.g., "v4", "v5e", "v5p")
        flops_tflops: Peak TFLOPs/s for bf16 operations
        hbm_gb: HBM capacity in GB per chip
    """

    name: str
    flops_tflops: float
    hbm_gb: float

    @property
    def compute_intensity_ratio(self) -> float:
        """Hardware compute intensity ratio (TFLOPs/GB).

        This ratio determines the balance point between compute and memory.
        Workloads with higher intensity are compute-bound (HBM stranded).
        Workloads with lower intensity are memory-bound (FLOPs stranded).
        """
        return self.flops_tflops / self.hbm_gb


# Pre-defined TPU capacity configurations
TPU_V4 = TPUCapacity("v4", flops_tflops=275, hbm_gb=32)  # intensity = 8.6
TPU_V5E = TPUCapacity("v5e", flops_tflops=197, hbm_gb=16)  # intensity = 12.3
TPU_V5P = TPUCapacity("v5p", flops_tflops=459, hbm_gb=95)  # intensity = 4.8
TPU_V6E = TPUCapacity("v6e", flops_tflops=918, hbm_gb=32)  # intensity = 28.7

# Default TPU for when hardware type is not specified
DEFAULT_TPU = TPU_V4

# Mapping from JAX device_kind patterns to TPU capacities
_TPU_CAPACITY_MAP: dict[str, TPUCapacity] = {
    "v6e": TPU_V6E,
    "v6 lite": TPU_V6E,
    "v5p": TPU_V5P,
    "v5 litepod": TPU_V5E,  # v5e appears as "v5 lite" or "v5 litepod" in JAX
    "v5e": TPU_V5E,
    "v5lite": TPU_V5E,
    "v4": TPU_V4,
}


def detect_tpu_capacity() -> TPUCapacity:
    """Auto-detect TPU capacity from JAX runtime.

    Queries the JAX device and returns the corresponding TPUCapacity.
    Falls back to DEFAULT_TPU (v4) if detection fails or not on TPU.

    Returns:
        TPUCapacity for the detected TPU type.
    """
    try:
        import jax

        devices = jax.devices()
        if not devices:
            return DEFAULT_TPU

        device = devices[0]
        if device.platform != "tpu":
            return DEFAULT_TPU

        device_kind = device.device_kind.lower()

        # Match against known patterns
        for pattern, capacity in _TPU_CAPACITY_MAP.items():
            if pattern in device_kind:
                return capacity

        # Unknown TPU type, fall back to default
        return DEFAULT_TPU

    except ImportError:
        # JAX not available
        return DEFAULT_TPU
    except Exception:
        # Any other error during detection
        return DEFAULT_TPU


@dataclass
class ModelResourceConfig:
    """Model parameters relevant to resource estimation.

    Attributes:
        num_hidden_layers: Number of transformer layers (L)
        hidden_size: Hidden dimension (d)
        num_attention_heads: Number of attention heads
        head_dim: Dimension per attention head (d / num_heads)
        num_key_value_heads: Number of KV heads (for GQA/MQA)
        intermediate_size: FFN intermediate dimension (typically 4d)
        vocab_size: Vocabulary size
        num_params: Total parameter count (approximate)
        bytes_per_element: Bytes per KV cache element (2 for bf16/fp16)
        num_experts: Number of experts for MoE models (1 for dense models)
        num_active_experts: Number of active experts per token (1 for dense)
    """

    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    num_params: int
    bytes_per_element: int = 2  # bf16/fp16 default
    num_experts: int = 1  # Dense model default
    num_active_experts: int = 1  # Dense model default

    @property
    def is_moe(self) -> bool:
        """Whether this is a Mixture of Experts model."""
        return self.num_experts > 1

    @classmethod
    def from_model_config(
        cls, model_config: ModelConfig, bytes_per_element: int = 2
    ) -> "ModelResourceConfig":
        """Create from SGLang ModelConfig."""
        hf_config = model_config.hf_config

        # Extract model dimensions
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", 32)
        hidden_size = getattr(hf_config, "hidden_size", 4096)
        num_attention_heads = getattr(hf_config, "num_attention_heads", 32)
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = getattr(
            hf_config, "num_key_value_heads", num_attention_heads
        )
        intermediate_size = getattr(hf_config, "intermediate_size", hidden_size * 4)
        vocab_size = getattr(hf_config, "vocab_size", 32000)

        # Extract MoE parameters (Mixtral uses num_local_experts and num_experts_per_tok)
        num_experts = getattr(hf_config, "num_local_experts", 1)
        num_active_experts = getattr(hf_config, "num_experts_per_tok", 1)

        # Estimate parameter count
        # Rough formula: embedding + L * (attn + ffn) + output
        embed_params = vocab_size * hidden_size
        attn_params = num_hidden_layers * 4 * hidden_size * hidden_size  # Q, K, V, O
        # For MoE, multiply FFN params by number of experts
        ffn_params = num_hidden_layers * 3 * hidden_size * intermediate_size * num_experts
        num_params = embed_params + attn_params + ffn_params

        return cls(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            num_params=num_params,
            bytes_per_element=bytes_per_element,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
        )


@dataclass
class ResourceEstimate:
    """Estimated resource requirements for a request.

    Attributes:
        flops: Estimated FLOPs for prefill computation
        hbm_bytes: Estimated HBM bytes for KV cache storage
        input_tokens: Number of input tokens (FLOPs proxy)
        output_tokens: Estimated output tokens (HBM proxy)
    """

    flops: int
    hbm_bytes: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total sequence length (input + output)."""
        return self.input_tokens + self.output_tokens


class ResourceEstimator:
    """Estimates FLOPs and HBM requirements for LLM inference requests.

    The estimator uses model architecture parameters to compute:
    - FLOPs: Dominated by prefill phase attention and FFN operations
    - HBM: Dominated by KV cache storage for all tokens

    For scheduling purposes, we use simplified proxies:
    - FLOPs proxy: input_tokens (prefill dominates compute)
    - HBM proxy: input_tokens + output_tokens (KV cache is linear)

    Example:
        >>> config = ModelResourceConfig(
        ...     num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
        ...     head_dim=128, num_key_value_heads=8, intermediate_size=14336,
        ...     vocab_size=128000, num_params=70_000_000_000
        ... )
        >>> estimator = ResourceEstimator(config)
        >>> estimate = estimator.estimate(input_tokens=1024, output_tokens=512)
        >>> print(f"FLOPs: {estimate.flops:.2e}, HBM: {estimate.hbm_bytes / 1e9:.2f} GB")
    """

    def __init__(
        self,
        model_config: ModelResourceConfig,
        tpu_capacity: TPUCapacity | None = None,
        max_total_tokens: int | None = None,
    ):
        """Initialize the resource estimator.

        Args:
            model_config: Model architecture parameters
            tpu_capacity: TPU hardware specifications (defaults to TPU_V4)
            max_total_tokens: Maximum total tokens per DP replica (for capacity checks)
        """
        self.config = model_config
        self.tpu_capacity = tpu_capacity or DEFAULT_TPU
        self.max_total_tokens = max_total_tokens

    @classmethod
    def from_server_args(
        cls,
        model_config: ModelConfig,
        server_args: ServerArgs,
        dp_size: int = 1,
        tpu_capacity: TPUCapacity | None = None,
    ) -> "ResourceEstimator":
        """Create estimator from server configuration.

        Args:
            model_config: Model configuration
            server_args: Server arguments containing capacity settings
            dp_size: Data parallel degree for capacity calculation
            tpu_capacity: TPU hardware specifications (auto-detected if None)
        """
        resource_config = ModelResourceConfig.from_model_config(model_config)

        # Compute max tokens per DP replica
        max_tokens = getattr(server_args, "max_total_num_tokens", 20000)
        max_total_tokens = max_tokens // max(dp_size, 1)

        # Auto-detect TPU capacity if not provided
        if tpu_capacity is None:
            tpu_capacity = detect_tpu_capacity()

        return cls(
            model_config=resource_config,
            tpu_capacity=tpu_capacity,
            max_total_tokens=max_total_tokens,
        )

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> ResourceEstimate:
        """Estimate resource requirements for a request.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Estimated number of output tokens

        Returns:
            ResourceEstimate with FLOPs and HBM requirements
        """
        flops = self.estimate_flops(input_tokens)
        hbm_bytes = self.estimate_hbm_bytes(input_tokens, output_tokens)

        return ResourceEstimate(
            flops=flops,
            hbm_bytes=hbm_bytes,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def estimate_flops(self, input_tokens: int) -> int:
        """Estimate FLOPs for prefill phase.

        Attention and FFN run sequentially within each layer. For sequential
        operations, the bottleneck (max) determines throughput, not the sum.

        - Attention: 4 * L * n^2 * d (Q@K, softmax, attn@V, output proj)
        - FFN (dense): 6 * L * n * d * d_ffn (for gated FFN with 3 projections)
        - FFN (MoE): 6 * L * n * d * d_ffn * K (K = num_active_experts)

        For MoE models, only K experts activate per token, so FFN FLOPs scale
        with the number of active experts, not the total number of experts.

        At short sequences, FFN dominates. At long sequences (n > ~1.5 * d_ffn * K),
        attention dominates due to quadratic scaling.

        Args:
            input_tokens: Number of input tokens (n)

        Returns:
            Estimated peak FLOPs for prefill (bottleneck operation)
        """
        n = input_tokens
        L = self.config.num_hidden_layers
        d = self.config.hidden_size
        d_ffn = self.config.intermediate_size
        K = self.config.num_active_experts  # 1 for dense, typically 2 for MoE

        # Attention FLOPs: 4 * L * n^2 * d
        # (Q@K^T: n*d @ d*n = n^2*d, scaled for all heads and layers)
        attn_flops = 4 * L * n * n * d

        # FFN FLOPs: 6 * L * n * d * d_ffn * K
        # For MoE, K experts activate per token (K=1 for dense models)
        ffn_flops = 6 * L * n * d * d_ffn * K

        # Sequential operations: bottleneck determines peak compute demand
        total_flops = max(attn_flops, ffn_flops)

        return total_flops

    def estimate_hbm_bytes(self, input_tokens: int, output_tokens: int) -> int:
        """Estimate HBM bytes for KV cache.

        KV cache stores key and value vectors for each token:
        HBM = 2 * L * kv_dim * (n + m) * bytes_per_element

        Where:
        - 2 = key + value
        - L = number of layers
        - kv_dim = num_kv_heads * head_dim
        - n + m = total sequence length
        - bytes_per_element = 2 for bf16/fp16

        Args:
            input_tokens: Number of input tokens (n)
            output_tokens: Number of output tokens (m)

        Returns:
            Estimated HBM bytes for KV cache
        """
        total_tokens = input_tokens + output_tokens
        L = self.config.num_hidden_layers
        kv_dim = self.config.num_key_value_heads * self.config.head_dim
        bytes_per_elem = self.config.bytes_per_element

        # KV cache: 2 (K+V) * layers * kv_dim * seq_len * bytes
        hbm_bytes = 2 * L * kv_dim * total_tokens * bytes_per_elem

        return hbm_bytes

    def compute_intensity_ratio(self, input_tokens: int, output_tokens: int) -> float:
        """Compute workload intensity ratio (TFLOPs/GB).

        Intensity = Prefill FLOPs (TFLOPs) / KV Cache HBM (GB)

        Higher intensity means compute-heavy workload (HBM stranded).
        Lower intensity means memory-heavy workload (FLOPs stranded).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Workload intensity in TFLOPs/GB
        """
        if input_tokens <= 0:
            return 0.0

        flops = self.estimate_flops(input_tokens)
        hbm_bytes = self.estimate_hbm_bytes(input_tokens, output_tokens)

        if hbm_bytes <= 0:
            return float("inf")

        # Convert to TFLOPs / GB
        flops_tflops = flops / 1e12
        hbm_gb = hbm_bytes / 1e9

        return flops_tflops / hbm_gb

    def compute_stranding(
        self,
        input_tokens: int,
        output_tokens: int,
        flops_weight: float = 1.0,
        hbm_weight: float = 1.0,
    ) -> float:
        """Compute stranding score using intensity ratios.

        Stranding measures resource imbalance between workload and hardware:
        - FLOPs stranding = max(0, 1 - workload_compute_intensity / hardware_compute_intensity)
        - HBM stranding = max(0, 1 - hardware_compute_intensity / workload_compute_intensity)

        When workload compute intensity < hardware compute intensity: memory bottleneck (FLOPs stranded)
        When workload compute intensity > hardware compute intensity: compute bottleneck (HBM stranded)

        Lower stranding = better fit.

        Args:
            input_tokens: Total input tokens (determines FLOPs)
            output_tokens: Total output tokens (with input, determines HBM)
            flops_weight: Cost weight for FLOPs stranding
            hbm_weight: Cost weight for HBM stranding

        Returns:
            Weighted stranding score in [0, 1] for each dimension
        """
        # Handle empty load case
        if input_tokens <= 0 and output_tokens <= 0:
            return flops_weight + hbm_weight  # Maximum stranding for empty bin

        workload_compute_intensity = self.compute_intensity_ratio(input_tokens, output_tokens)
        hardware_compute_intensity = self.tpu_capacity.compute_intensity_ratio

        # Handle edge cases
        if workload_compute_intensity <= 0:
            # No compute, pure memory -> maximum FLOPs stranding
            return flops_weight
        if workload_compute_intensity == float("inf"):
            # No memory, pure compute -> maximum HBM stranding
            return hbm_weight

        # FLOPs stranding: memory is bottleneck (workload intensity < hardware)
        flops_stranding = max(0.0, 1.0 - workload_compute_intensity / hardware_compute_intensity)

        # HBM stranding: compute is bottleneck (workload intensity > hardware)
        hbm_stranding = max(0.0, 1.0 - hardware_compute_intensity / workload_compute_intensity)

        return flops_weight * flops_stranding + hbm_weight * hbm_stranding

    def select_best_dp_rank(
        self,
        dp_loads: list[tuple[int, int]],
        new_input_tokens: int,
        new_output_tokens: int,
        flops_weight: float = 1.0,
        hbm_weight: float = 1.0,
        respect_capacity: bool = True,
    ) -> int | None:
        """Select the best DP rank for a new request.

        Evaluates all DP ranks and returns the one that minimizes
        stranding after placing the request.

        Args:
            dp_loads: List of (input_tokens, output_tokens) per DP rank
            new_input_tokens: Input tokens for the new request
            new_output_tokens: Output tokens for the new request
            flops_weight: Cost weight for FLOPs stranding
            hbm_weight: Cost weight for HBM stranding
            respect_capacity: If True, skip ranks that would exceed max_total_tokens

        Returns:
            Best DP rank index, or None if no feasible placement exists
        """
        best_rank = None
        best_stranding = float("inf")

        for rank, (existing_input, existing_output) in enumerate(dp_loads):
            # Compute total load after adding new request
            after_input = existing_input + new_input_tokens
            after_output = existing_output + new_output_tokens
            after_total = after_input + after_output

            # Skip if would exceed token capacity
            if respect_capacity and self.max_total_tokens is not None:
                if after_total > self.max_total_tokens:
                    continue

            # Compute stranding using intensity ratios
            stranding = self.compute_stranding(
                after_input, after_output, flops_weight, hbm_weight
            )

            if stranding < best_stranding:
                best_stranding = stranding
                best_rank = rank

        return best_rank

    def format_estimate(self, estimate: ResourceEstimate) -> str:
        """Format resource estimate for logging."""
        flops_str = f"{estimate.flops:.2e}" if estimate.flops > 1e9 else str(estimate.flops)
        hbm_gb = estimate.hbm_bytes / (1024**3)
        return (
            f"FLOPs={flops_str}, HBM={hbm_gb:.3f}GB, "
            f"tokens={estimate.input_tokens}+{estimate.output_tokens}"
        )
