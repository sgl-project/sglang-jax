"""Unified quantization configuration.

Quantization settings are explicit - no fallbacks between components.
Config files should specify both qwix (for dense layers) and moe sections.
Dense models will use qwix rules only; MoE models will use both.
"""

import os
from dataclasses import dataclass

import jax.numpy as jnp
import yaml

# Map string dtype names to JAX numpy dtypes
DTYPE_MAP = {
    "int8": jnp.int8,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    None: None,
}

# Path to built-in quantization config files
BUILTIN_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "utils", "quantization", "configs"
)


def _str_to_dtype(dtype_str: str | None) -> jnp.dtype | None:
    """Convert a string dtype name to a JAX numpy dtype."""
    if dtype_str is None:
        return None
    if dtype_str not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported: {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[dtype_str]


def _resolve_config_path(config_path: str) -> str:
    """Resolve a config path, checking both absolute and built-in locations."""
    # If it's an absolute path or exists as-is, use it directly
    if os.path.isabs(config_path) or os.path.exists(config_path):
        if os.path.exists(config_path):
            return config_path
        raise FileNotFoundError(f"Quantization config file not found: {config_path}")

    # Try looking in the built-in configs directory
    builtin_path = os.path.join(BUILTIN_CONFIG_PATH, config_path)
    if os.path.exists(builtin_path):
        return builtin_path

    raise FileNotFoundError(
        f"Quantization config file not found: {config_path}. "
        f"Searched in current directory and {BUILTIN_CONFIG_PATH}"
    )


@dataclass
class QuantizationConfig:
    """Quantization configuration with explicit settings (no fallbacks).

    Attributes:
        qwix_rules: List of qwix quantization rules for dense layers
        moe_weight_dtype: Dtype for MoE weight quantization (None = no quantization)
        moe_activation_dtype: Dtype for MoE activation quantization (None = no quantization)
    """

    qwix_rules: list[dict] | None = None
    moe_weight_dtype: jnp.dtype | None = None
    moe_activation_dtype: jnp.dtype | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "QuantizationConfig":
        """Load quantization config from a YAML file.

        Expected YAML format:
        ```yaml
        quantization:
          qwix:
            rules:
              - module_path: '.*'
                weight_qtype: 'int8'
                # act_qtype: 'int8'  # optional

          moe:
            weight_dtype: 'int8'
            activation_dtype: null  # null = no activation quantization
        ```
        """
        resolved_path = _resolve_config_path(yaml_path)

        with open(resolved_path) as f:
            cfg = yaml.safe_load(f)

        if "quantization" not in cfg:
            raise ValueError(
                f"Invalid quantization config format in {resolved_path}. "
                "Expected 'quantization' key at top level."
            )

        quant = cfg["quantization"]

        # Parse qwix rules (required)
        qwix_section = quant.get("qwix", {})
        qwix_rules = qwix_section.get("rules")
        if not qwix_rules:
            raise ValueError(
                f"No qwix rules found in {resolved_path}. "
                "The 'quantization.qwix.rules' section is required."
            )

        # Parse MoE settings (required)
        moe_section = quant.get("moe", {})
        if not moe_section:
            raise ValueError(
                f"No moe section found in {resolved_path}. "
                "The 'quantization.moe' section is required."
            )
        moe_weight_dtype = _str_to_dtype(moe_section.get("weight_dtype"))
        moe_activation_dtype = _str_to_dtype(moe_section.get("activation_dtype"))

        return cls(
            qwix_rules=qwix_rules,
            moe_weight_dtype=moe_weight_dtype,
            moe_activation_dtype=moe_activation_dtype,
        )

    @classmethod
    def from_path(cls, config_path: str | None) -> "QuantizationConfig | None":
        """Load quantization config from a path.

        Args:
            config_path: Path to the YAML config file, or None to disable quantization.

        Returns:
            QuantizationConfig if config_path is specified, None otherwise
        """
        if config_path is None:
            return None
        return cls.from_yaml(config_path)

    def get_moe_weight_dtype(self) -> jnp.dtype | None:
        """Get the dtype for MoE weight quantization."""
        return self.moe_weight_dtype

    def get_moe_activation_dtype(self) -> jnp.dtype | None:
        """Get the dtype for MoE activation quantization."""
        return self.moe_activation_dtype

    def get_qwix_rules(self) -> list[dict]:
        """Get the qwix rules for dense layer quantization."""
        return self.qwix_rules or []

    def has_moe_quantization(self) -> bool:
        """Check if MoE quantization is configured."""
        return self.moe_weight_dtype is not None or self.moe_activation_dtype is not None
