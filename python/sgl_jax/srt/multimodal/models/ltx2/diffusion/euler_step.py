"""
Euler diffusion step implementation in JAX.

This module implements the first-order Euler method for diffusion sampling,
ported from the PyTorch reference implementation.
"""

import jax
import jax.numpy as jnp
from jax import Array


def to_velocity(
    sample: Array,
    sigma: float | Array,
    denoised_sample: Array,
    calc_dtype: jnp.dtype = jnp.float32,
) -> Array:
    """
    Convert the sample and its denoised version to velocity.

    The velocity represents the rate of change from the noisy sample to the
    denoised sample, scaled by the noise level (sigma).

    Args:
        sample: The noisy sample tensor.
        sigma: The noise level (sigma) at the current step. Can be a scalar or array.
        denoised_sample: The denoised prediction from the model.
        calc_dtype: The dtype to use for intermediate calculations. Defaults to float32
                   for numerical stability.

    Returns:
        The velocity tensor with the same dtype as the input sample.

    Raises:
        ValueError: If sigma is 0.0, as this would result in division by zero.
    """
    # Extract scalar value if sigma is an array
    if isinstance(sigma, Array):
        sigma = sigma.astype(calc_dtype).item()

    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")

    # Cast to calc_dtype for numerical stability, compute velocity, then cast back
    sample_calc = sample.astype(calc_dtype)
    denoised_calc = denoised_sample.astype(calc_dtype)
    velocity = (sample_calc - denoised_calc) / sigma

    return velocity.astype(sample.dtype)


class EulerDiffusionStep:
    """
    First-order Euler method for diffusion sampling.

    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.

    The Euler method is a simple numerical integration technique that approximates
    the solution to an ODE by taking steps proportional to the derivative (velocity)
    at the current point.
    """

    def step(
        self,
        sample: Array,
        denoised_sample: Array,
        sigmas: Array,
        step_index: int,
    ) -> Array:
        """
        Perform a single Euler integration step.

        Args:
            sample: The current noisy sample at the current sigma level.
            denoised_sample: The denoised prediction from the model at the current step.
            sigmas: Array of sigma (noise level) values for all steps in the schedule.
            step_index: The current step index in the diffusion process.

        Returns:
            The updated sample after taking an Euler step towards the next sigma level.
        """
        # Get current and next sigma values
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]

        # Calculate time step (change in sigma)
        dt = sigma_next - sigma

        # Compute velocity from current sample and denoised prediction
        velocity = to_velocity(sample, sigma, denoised_sample)

        # Apply Euler integration: x_next = x_current + velocity * dt
        # Use float32 for numerical stability during the update
        sample_fp32 = sample.astype(jnp.float32)
        velocity_fp32 = velocity.astype(jnp.float32)
        next_sample = sample_fp32 + velocity_fp32 * dt

        # Cast back to original dtype
        return next_sample.astype(sample.dtype)
