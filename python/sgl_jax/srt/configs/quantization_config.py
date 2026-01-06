"""Unified quantization configuration."""

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
    """Resolve a config path, checking both absolute and built-in locations.

    Args:
        config_path: Either an absolute path, relative path, or just a filename
                    (which will be looked up in the built-in configs directory)

    Returns:
        The resolved absolute path to the config file

    Raises:
        FileNotFoundError: If the config file cannot be found
    """
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
    """Single source of truth for all quantization settings.

    This class encapsulates all quantization-related configuration,
    making it easy to:
    - Define custom quantization rules via YAML files
    - Access quantization settings from any model component
    - Override settings for specific components (e.g., MoE vs dense layers)

    Attributes:
        weight_dtype: Default dtype for quantized weights (e.g., jnp.int8)
        activation_dtype: Default dtype for quantized activations (None = no quant)
        qwix_rules: List of qwix quantization rules for dense layers
        moe_weight_dtype: Override weight dtype for MoE layers
        moe_activation_dtype: Override activation dtype for MoE layers
    """

    weight_dtype: jnp.dtype | None = None
    activation_dtype: jnp.dtype | None = None
    qwix_rules: list[dict] | None = None
    moe_weight_dtype: jnp.dtype | None = None
    moe_activation_dtype: jnp.dtype | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "QuantizationConfig":
        """Load quantization config from a YAML file.

        Args:
            yaml_path: Path to the YAML config file. Can be:
                - An absolute path
                - A relative path from the current directory
                - Just a filename (will be looked up in built-in configs)

        Supports both the new unified format and backward-compatible qwix-only format.

        New format:
        ```yaml
        quantization:
          weight_dtype: 'int8'
          activation_dtype: null
          qwix:
            rules:
              - module_path: '.*'
                weight_qtype: 'int8'
          moe:
            weight_dtype: 'int8'
            activation_dtype: 'int8'
        ```

        Legacy format (still supported):
        ```yaml
        qwix:
          rules:
            - module_path: '.*'
              weight_qtype: 'int8'
        ```
        """
        resolved_path = _resolve_config_path(yaml_path)

        with open(resolved_path) as f:
            cfg = yaml.safe_load(f)

        # Support both new 'quantization' key and legacy 'qwix' key
        if "quantization" in cfg:
            quant = cfg["quantization"]
            qwix_section = quant.get("qwix", {})
            moe_section = quant.get("moe", {})

            return cls(
                weight_dtype=_str_to_dtype(quant.get("weight_dtype")),
                activation_dtype=_str_to_dtype(quant.get("activation_dtype")),
                qwix_rules=qwix_section.get("rules"),
                moe_weight_dtype=_str_to_dtype(moe_section.get("weight_dtype")),
                moe_activation_dtype=_str_to_dtype(moe_section.get("activation_dtype")),
            )
        elif "qwix" in cfg:
            # Legacy format: infer dtype from qwix rules
            qwix_rules = cfg["qwix"].get("rules", [])
            weight_dtype = None
            if qwix_rules:
                # Get weight_qtype from first rule as default
                weight_qtype = qwix_rules[0].get("weight_qtype")
                weight_dtype = _str_to_dtype(weight_qtype)

            return cls(
                weight_dtype=weight_dtype,
                activation_dtype=None,
                qwix_rules=qwix_rules,
                moe_weight_dtype=None,
                moe_activation_dtype=None,
            )
        else:
            raise ValueError(
                f"Invalid quantization config format in {resolved_path}. "
                "Expected 'quantization' or 'qwix' key."
            )

    @classmethod
    def from_path(cls, config_path: str | None) -> "QuantizationConfig | None":
        """Load quantization config from a path.

        This is the primary factory method. Pass a path to a YAML config file.

        Args:
            config_path: Path to the YAML config file, or None to disable quantization.
                        Can be an absolute path, relative path, or just a filename
                        (which will be looked up in the built-in configs directory).

        Returns:
            QuantizationConfig if config_path is specified, None otherwise

        Examples:
            # Use a built-in config
            config = QuantizationConfig.from_path("int8_all_modules_w_only.yaml")

            # Use a custom config with absolute path
            config = QuantizationConfig.from_path("/path/to/my_custom_config.yaml")

            # Disable quantization
            config = QuantizationConfig.from_path(None)
        """
        if config_path is None:
            return None
        return cls.from_yaml(config_path)

    def get_moe_weight_dtype(self) -> jnp.dtype | None:
        """Get the dtype for MoE weight quantization.

        Returns MoE-specific override if set, otherwise falls back to global weight_dtype.
        """
        return self.moe_weight_dtype if self.moe_weight_dtype is not None else self.weight_dtype

    def get_moe_activation_dtype(self) -> jnp.dtype | None:
        """Get the dtype for MoE activation quantization.

        Returns MoE-specific override if set, otherwise falls back to global activation_dtype.
        """
        return self.moe_activation_dtype if self.moe_activation_dtype is not None else self.activation_dtype

    def get_qwix_rules(self) -> list[dict]:
        """Get the qwix rules for dense layer quantization."""
        return self.qwix_rules or []
