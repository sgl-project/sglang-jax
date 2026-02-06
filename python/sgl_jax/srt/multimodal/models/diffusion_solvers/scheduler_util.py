# Adapted from https://github.com/huggingface/diffusers
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0
import importlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import flax
import jax
import jax.numpy as jnp

SCHEDULER_CONFIG_NAME = "scheduler_config.json"


class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSchedulerMultistepScheduler = 5


@dataclass
class SchedulerOutput:
    prev_sample: jax.Array


class SchedulerMixin:
    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config: ClassVar = ["dtype"]
    _compatibles: ClassVar = []
    has_compatibles = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        subfolder: str | None = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            **kwargs,
        )
        scheduler, unused_kwargs = cls.from_config(config, return_unused_kwargs=True, **kwargs)

        if hasattr(scheduler, "create_state") and getattr(scheduler, "has_state", False):
            state = scheduler.create_state()

        if return_unused_kwargs:
            return scheduler, state, unused_kwargs

        return scheduler, state

    def save_pretrained(
        self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs
    ):
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    @property
    def compatibles(self):
        return self._get_compatibles()

    @classmethod
    def _get_compatibles(cls):
        compatible_classes_str = list(set([cls.__name__, *cls._compatibles]))
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        compatible_classes = [
            getattr(diffusers_library, c)
            for c in compatible_classes_str
            if hasattr(diffusers_library, c)
        ]
        return compatible_classes


def broadcast_to_shape_from_left(x: jax.Array, shape: tuple[int]) -> jax.Array:
    assert len(shape) >= x.ndim
    return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def betas_for_alpha_bar(
    num_diffusion_timesteps: int, max_beta=0.999, dtype=jnp.float32
) -> jax.Array:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas, dtype=dtype)


def rescale_betas_zero_snr(betas):
    """Rescale betas for zero terminal SNR. See https://arxiv.org/pdf/2305.08891.pdf"""
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_bar_sqrt = jnp.sqrt(alphas_cumprod)

    alphas_bar_sqrt_0 = jnp.copy(alphas_bar_sqrt[0])
    alphas_bar_sqrt_T = jnp.copy(alphas_bar_sqrt[-1])

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = jnp.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


@flax.struct.dataclass
class CommonSchedulerState:
    alphas: jax.Array
    betas: jax.Array
    alphas_cumprod: jax.Array

    @classmethod
    def create(cls, scheduler):
        config = scheduler.config

        if config.trained_betas is not None:
            betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
        elif config.beta_schedule == "linear":
            betas = jnp.linspace(
                config.beta_start,
                config.beta_end,
                config.num_train_timesteps,
                dtype=scheduler.dtype,
            )
        elif config.beta_schedule == "scaled_linear":
            betas = (
                jnp.linspace(
                    config.beta_start**0.5,
                    config.beta_end**0.5,
                    config.num_train_timesteps,
                    dtype=scheduler.dtype,
                )
                ** 2
            )
        elif config.beta_schedule == "squaredcos_cap_v2":
            betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=scheduler.dtype)
        else:
            raise NotImplementedError(
                f"beta_schedule {config.beta_schedule} is not implemented for {scheduler.__class__.__name__}"
            )

        if not config.rescale_zero_terminal_snr:
            betas = rescale_betas_zero_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)

        return cls(alphas=alphas, betas=betas, alphas_cumprod=alphas_cumprod)


def get_sqrt_alpha_prod(
    state: CommonSchedulerState,
    original_samples: jax.Array,
    noise: jax.Array,
    timesteps: jax.Array,
):
    alphas_cumprod = state.alphas_cumprod

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(
        sqrt_one_minus_alpha_prod, original_samples.shape
    )

    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def add_noise_common(
    state: CommonSchedulerState,
    original_samples: jax.Array,
    noise: jax.Array,
    timesteps: jax.Array,
):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(
        state, original_samples, noise, timesteps
    )
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def get_velocity_common(
    state: CommonSchedulerState, sample: jax.Array, noise: jax.Array, timesteps: jax.Array
):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(
        state, sample, noise, timesteps
    )
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
