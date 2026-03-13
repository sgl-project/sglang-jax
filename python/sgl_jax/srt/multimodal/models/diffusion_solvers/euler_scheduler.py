import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

@dataclass
class FlowSchedulerOutput:
    """Output class for scheduler step."""
    prev_sample: jax.Array

class EulerScheduler:
    """
    Euler Scheduler for LTX-2 diffusion sampling.
    
    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value. Uses a first-order Euler method for integration.
    """
    
    def __init__(
        self,
        base_shift: float = 0.95,
        max_shift: float = 2.05,
        stretch: bool = True,
        terminal: float = 0.1,
        timestep_scale_multiplier: int = 1000,
    ):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.stretch = stretch
        self.terminal = terminal
        self.timestep_scale_multiplier = timestep_scale_multiplier
        
        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[jax.Array] = None
        self._sigmas: Optional[jax.Array] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    def set_begin_index(self, begin_index: int = 0):
        """Sets the begin index for the scheduler."""
        self._begin_index = begin_index
        self._step_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        shape: tuple,
        mu: Optional[float] = None,
        **kwargs,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.
        
        Args:
            num_inference_steps: Number of diffusion steps for inference
            shape: Shape of the latent tensor
            mu: Pre-calculated dynamic shift value. If None, it's calculated from shape.
        """
        self.num_inference_steps = num_inference_steps
        
        if mu is not None:
            sigma_shift = mu
        else:
            # LTX-2 token calculation if mu isn't provided directly
            tokens = math.prod(shape[2:]) if len(shape) >= 3 else 4096
            x1 = 1024
            x2 = 4096
            mm = (self.max_shift - self.base_shift) / (x2 - x1)
            b = self.base_shift - mm * x1
            sigma_shift = tokens * mm + b

        # Create linear sigma schedule from 1.0 to 0.0
        sigmas = jnp.linspace(1.0, 0.0, num_inference_steps + 1)

        # Apply exponential shift transformation
        power = 1
        sigmas = jnp.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value
        if self.stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]

            if non_zero_sigmas.size > 0:
                one_minus_z = 1.0 - non_zero_sigmas
                scale_factor = one_minus_z[-1] / (1.0 - self.terminal)
                stretched = 1.0 - (one_minus_z / scale_factor)

                stretched_full = jnp.zeros_like(sigmas)
                indices = jnp.where(non_zero_mask)[0]
                stretched_full = stretched_full.at[indices].set(stretched)

                sigmas = jnp.where(non_zero_mask, stretched_full, sigmas)

        self._sigmas = sigmas.astype(jnp.float32)
        # Multiply by timestep_scale_multiplier (1000) for the model timesteps
        self.timesteps = (self._sigmas[:-1] * self.timestep_scale_multiplier).astype(jnp.int32)
        self._step_index = None

    def step(
        self,
        model_output: jax.Array,
        timestep: Union[int, jax.Array],
        sample: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlowSchedulerOutput, Tuple[jax.Array]]:
        """
        Take a single Euler step.
        
        Args:
            model_output: Direct output from the diffusion model (velocity prediction)
            timestep: Current discrete timestep
            sample: Current noisy sample (latent)
            return_dict: Whether to return FlowSchedulerOutput or tuple
        """
        if self._step_index is None:
            self._step_index = self._begin_index if self._begin_index is not None else 0

        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]

        # Calculate time step (change in sigma)
        dt = sigma_next - sigma
        
        # Apply Euler integration: x_next = x_current + velocity * dt
        # Use float32 for numerical stability
        sample_fp32 = sample.astype(jnp.float32)
        velocity_fp32 = model_output.astype(jnp.float32)
        
        prev_sample = sample_fp32 + velocity_fp32 * dt
        prev_sample = prev_sample.astype(sample.dtype)
        
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowSchedulerOutput(prev_sample=prev_sample)
