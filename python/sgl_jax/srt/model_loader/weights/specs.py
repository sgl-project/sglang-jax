from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WeightSpec:
    """Declare checkpoint inputs, final parameter targets and their layout.

    Ordinary entries use the mapping key as their source. ``sources`` groups
    separate experts, or inputs consumed together by a recipe. Device recipes
    return JAX arrays; ``host_recipe`` splits/reshapes independent prefused
    experts as NumPy arrays and lets the loader read only local expert ranges.
    ``sharding`` describes output axes and otherwise comes from the parameter.
    """

    target_path: str | list[str]
    sharding: tuple[Any, ...] | None = None
    transpose: bool = False
    transpose_axes: tuple[int, ...] | None = (
        None  # For multi-dimensional transpose (e.g., conv weights)
    )
    reshape: tuple | None = None
    repeat: tuple[int, int] | None = None
    head_dim_padding: bool = False
    kv_head_padding: bool = False
    concat_axis: int | None = None
    physical_to_logical_map: np.ndarray | None = None
    pad_width: tuple[tuple[int, int], ...] | None = None
    split_sizes: tuple[int, ...] | None = None
    split_axis: int = -1  # Applied after transpose, before each target's layout.

    sources: tuple[str, ...] = ()
    recipe: Any = None
    host_recipe: Any = None  # Independent experts; pure NumPy arrays in/out.
    optional: bool = False
