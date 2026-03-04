from typing import Any

import jax.numpy as jnp

STR_DTYPE_TO_JAX_DTYPE = {
    "half": jnp.float16,
    "float16": jnp.float16,
    "float": jnp.float32,
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


class DtypeConfig:
    def __init__(
        self, config_dict: dict[str, Any] | None = None, default_dtype: jnp.dtype | None = None
    ):
        # Validate at least one of config_dict and default_dtype is provided
        if config_dict is None and default_dtype is None:
            raise ValueError("At least one of config_dict and default_dtype must be provided.")

        self.config_dict = self._parse_dict(config_dict or {})

        # Resolve the default dtype for this level
        self.default_dtype = self.config_dict.get("default", default_dtype)

    def _parse_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively parses a dictionary, converting string dtypes to jnp.dtype."""
        parsed = {}
        for k, v in d.items():
            if isinstance(v, dict):
                parsed[k] = self._parse_dict(v)
            elif isinstance(v, str) and v.lower() in STR_DTYPE_TO_JAX_DTYPE:
                parsed[k] = STR_DTYPE_TO_JAX_DTYPE[v.lower()]
            else:
                raise ValueError(f"Unknown dtype: {v}")
        return parsed

    def get_config(self, key: str) -> "DtypeConfig":
        """Returns a child config covering the sub-dictionary, preserving the default."""
        return DtypeConfig(
            config_dict=self.config_dict.get(key, {}), default_dtype=self.default_dtype
        )

    def get_dtype(self, key: str) -> jnp.dtype | None:
        """Returns the specific dtype, or falls back to the default."""
        val = self.config_dict.get(key, self.default_dtype)
        if not isinstance(val, jnp.dtype):
            return self.default_dtype
        return val
