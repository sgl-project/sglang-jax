"""Layer-wise intermediate results dumping utility for debugging."""

import os
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


class LayerDumper:
    """Dump intermediate layer outputs for debugging."""

    def __init__(self, dump_dir: str | None = None, enabled: bool = False):
        self.enabled = enabled and dump_dir is not None
        self.dump_dir = Path(dump_dir) if dump_dir else None
        if self.enabled:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"[LayerDumper] Enabled, saving to {self.dump_dir}")

    def dump(self, name: str, tensor: jax.Array, step: int = 0):
        """Dump a tensor to disk.

        Args:
            name: Name of the tensor (e.g., "layer_0_attn_output")
            tensor: JAX array to dump
            step: Step number (for decode steps)
        """
        if not self.enabled:
            return

        # Convert to numpy on CPU
        tensor_np = jax.device_get(tensor)

        # Save with step number
        filename = f"{name}_step{step}.pkl"
        filepath = self.dump_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump({
                "name": name,
                "step": step,
                "shape": tensor_np.shape,
                "dtype": str(tensor_np.dtype),
                "data": tensor_np,
                "stats": {
                    "mean": float(tensor_np.mean()),
                    "std": float(tensor_np.std()),
                    "min": float(tensor_np.min()),
                    "max": float(tensor_np.max()),
                    "absmax": float(jnp.abs(tensor_np).max()),
                }
            }, f)

    def dump_dict(self, prefix: str, tensors: dict[str, jax.Array], step: int = 0):
        """Dump multiple tensors with a common prefix."""
        for key, tensor in tensors.items():
            self.dump(f"{prefix}_{key}", tensor, step)


# Global dumper instance
_global_dumper: LayerDumper | None = None


def init_dumper(dump_dir: str | None = None):
    """Initialize global dumper from environment variable or argument."""
    global _global_dumper

    # Check environment variable
    env_dir = os.environ.get("SGLANG_DUMP_DIR")
    dump_dir = dump_dir or env_dir

    enabled = dump_dir is not None
    _global_dumper = LayerDumper(dump_dir=dump_dir, enabled=enabled)
    return _global_dumper


def get_dumper() -> LayerDumper:
    """Get the global dumper instance."""
    global _global_dumper
    if _global_dumper is None:
        _global_dumper = init_dumper()
    return _global_dumper


def dump_tensor(name: str, tensor: jax.Array, step: int = 0):
    """Convenience function to dump a tensor using global dumper."""
    get_dumper().dump(name, tensor, step)


def dump_dict(prefix: str, tensors: dict[str, jax.Array], step: int = 0):
    """Convenience function to dump multiple tensors using global dumper."""
    get_dumper().dump_dict(prefix, tensors, step)
