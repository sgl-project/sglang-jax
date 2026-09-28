"""Checkpoint declarations and the shared JAX weight loading entry point."""

from .loader import WeightLoader
from .source import LocalSource
from .specs import WeightSpec

__all__ = ["WeightLoader", "WeightSpec", "LocalSource"]
