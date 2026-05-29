# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""Base classes and structures for multimodal encoders in JAX."""

from dataclasses import dataclass

import jax


@dataclass
class BaseEncoderOutput:
    """
    Base class for outputs of multimodal encoder models.
    Aligned with SGLang and Hugging Face semantics.
    """

    last_hidden_state: jax.Array | None = None
    pooler_output: jax.Array | None = None
    hidden_states: list[jax.Array] | None = None
    attentions: list[jax.Array] | None = None
    attention_mask: jax.Array | None = None
