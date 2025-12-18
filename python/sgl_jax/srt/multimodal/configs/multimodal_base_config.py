# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class MultiModalModelConfigs:
    model_path: str | None = None
    revision: str | None = None
    dtype: jnp.dtype = jnp.bfloat16
