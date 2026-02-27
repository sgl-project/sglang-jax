"""
LTX-2 Scheduler implementation in JAX.

This module provides the LTX2Scheduler class, which generates a sigma schedule
for diffusion sampling with token-count-dependent shifting and optional stretching.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp

# Constants for shift calculation
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """
    Default scheduler for LTX-2 diffusion sampling.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value. The scheduler uses a linear base schedule
    that is then transformed using an exponential shift based on the number of
    tokens in the latent representation.

    The shifting formula adjusts the noise schedule based on the latent size,
    allowing the model to adapt to different input resolutions and sequence lengths.
    """

    def execute(
        self,
        steps: int,
        latent: Optional[jax.Array] = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        **_kwargs,
    ) -> jax.Array:
        """
        Generate a sigma schedule for diffusion sampling.

        Args:
            steps: Number of sampling steps. The schedule will have steps+1 values.
            latent: Optional latent tensor. If provided, the number of tokens is
                   calculated from its spatial dimensions (shape[2:]). If None,
                   uses MAX_SHIFT_ANCHOR as the token count.
            max_shift: Maximum shift value used when tokens = MAX_SHIFT_ANCHOR.
                      Controls the upper bound of the shift range.
            base_shift: Base shift value used when tokens = BASE_SHIFT_ANCHOR.
                       Controls the lower bound of the shift range.
            stretch: If True, stretches the sigma schedule so the final non-zero
                    value matches the terminal value.
            terminal: Target value for the final non-zero sigma when stretch=True.
            **_kwargs: Additional keyword arguments (ignored).

        Returns:
            A jax.Array of shape (steps+1,) containing the sigma schedule.
            Values range from 1.0 (pure noise) to 0.0 (clean signal).

        Notes:
            The scheduling algorithm:
            1. Creates a linear schedule from 1.0 to 0.0
            2. Calculates a shift value based on the number of tokens
            3. Applies an exponential transformation to shift the schedule
            4. Optionally stretches the schedule to match a terminal value
        """
        # Calculate number of tokens from latent spatial dimensions
        if latent is not None:
            # Product of all spatial dimensions (excluding batch and channel dims)
            tokens = math.prod(latent.shape[2:])
        else:
            tokens = MAX_SHIFT_ANCHOR

        # Create linear sigma schedule from 1.0 to 0.0
        sigmas = jnp.linspace(1.0, 0.0, steps + 1)

        # Calculate token-dependent shift using linear interpolation
        # This creates a shift value that scales linearly with the number of tokens
        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        # Apply exponential shift transformation
        # Formula: exp(shift) / (exp(shift) + (1/sigma - 1)^power)
        # This shifts the schedule towards higher or lower noise levels
        power = 1
        sigmas = jnp.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]

            if non_zero_sigmas.size > 0:
                # Calculate scaling to match terminal value
                one_minus_z = 1.0 - non_zero_sigmas
                scale_factor = one_minus_z[-1] / (1.0 - terminal)
                stretched = 1.0 - (one_minus_z / scale_factor)

                # Use jnp.where to update only non-zero elements (pure function)
                # Create stretched version of full array
                stretched_full = jnp.zeros_like(sigmas)
                # Place stretched values at non-zero positions
                indices = jnp.where(non_zero_mask)[0]
                stretched_full = stretched_full.at[indices].set(stretched)

                # Update sigmas with stretched values where mask is True
                sigmas = jnp.where(non_zero_mask, stretched_full, sigmas)

        return sigmas.astype(jnp.float32)
