import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.configuration_utils import register_to_config
from sgl_jax.srt.multimodal.schedulers.scheduling_utils import SchedulerMixin
from sgl_jax.srt.multimodal.utils import BaseOutput

try:
    import scipy.stats
except ImportError:  # pragma: no cover
    scipy = None


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: jax.Array


class FlowMatchEulerDiscreteScheduler(SchedulerMixin):
    _compatibles: list[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
        stochastic_sampling: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

        if use_beta_sigmas and scipy is None:
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if (
            sum(
                [
                    use_beta_sigmas,
                    use_exponential_sigmas,
                    use_karras_sigmas,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of `use_beta_sigmas`, `use_exponential_sigmas`, `use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[
            ::-1
        ].copy()
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = jnp.asarray(sigmas * num_train_timesteps, dtype=dtype)
        self.sigmas = jnp.asarray(sigmas, dtype=dtype)
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self._shift = shift
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])
        self.num_inference_steps: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        self._shift = shift

    def calculate_mu(self, image_seq_len: int | float) -> float:
        """
        Compute the dynamic shifting parameter `mu` from image/token sequence length.

        This maps `image_seq_len` linearly from
        `[base_image_seq_len, max_image_seq_len]` to `[base_shift, max_shift]`.
        Values outside the configured range are clipped to the nearest endpoint.
        """
        if self.config.base_shift is None or self.config.max_shift is None:
            raise ValueError(
                "`base_shift` and `max_shift` must be set to calculate dynamic shifting `mu`."
            )

        base_seq = float(self.config.base_image_seq_len)
        max_seq = float(self.config.max_image_seq_len)
        if max_seq <= base_seq:
            raise ValueError("`max_image_seq_len` must be larger than `base_image_seq_len`.")

        seq = float(np.clip(image_seq_len, base_seq, max_seq))
        slope = (float(self.config.max_shift) - float(self.config.base_shift)) / (
            max_seq - base_seq
        )
        mu = float(self.config.base_shift) + slope * (seq - base_seq)
        return mu

    def _sigma_to_t(self, sigma: jax.Array | float) -> jax.Array:
        return jnp.asarray(sigma) * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: jax.Array) -> jax.Array:
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: jax.Array) -> jax.Array:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        shape: tuple = (),
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        del shape
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to True")

        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, "
                    "if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps
        is_timesteps_provided = timesteps is not None

        timesteps_np = np.asarray(timesteps, dtype=np.float32) if is_timesteps_provided else None

        if sigmas is None:
            if timesteps_np is None:
                timesteps_np = np.linspace(
                    float(self._sigma_to_t(self.sigma_max)),
                    float(self._sigma_to_t(self.sigma_min)),
                    num_inference_steps,
                ).astype(np.float32)
            sigmas_np = timesteps_np / self.config.num_train_timesteps
        else:
            sigmas_np = np.asarray(sigmas, dtype=np.float32)
            num_inference_steps = len(sigmas_np)

        if self.config.use_dynamic_shifting:
            sigmas_np = np.asarray(self.time_shift(mu, 1.0, jnp.asarray(sigmas_np)))
        else:
            sigmas_np = self.shift * sigmas_np / (1 + (self.shift - 1) * sigmas_np)

        if self.config.shift_terminal:
            sigmas_np = np.asarray(self.stretch_shift_to_terminal(jnp.asarray(sigmas_np)))

        if self.config.use_karras_sigmas:
            sigmas_np = self._convert_to_karras(sigmas_np, num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas_np = self._convert_to_exponential(sigmas_np, num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas_np = self._convert_to_beta(sigmas_np, num_inference_steps)

        sigmas_jax = jnp.asarray(sigmas_np, dtype=self.dtype)
        if not is_timesteps_provided:
            timesteps_jax = sigmas_jax * self.config.num_train_timesteps
        else:
            timesteps_jax = jnp.asarray(timesteps_np, dtype=self.dtype)

        if self.config.invert_sigmas:
            sigmas_jax = 1.0 - sigmas_jax
            timesteps_jax = sigmas_jax * self.config.num_train_timesteps
            sigmas_jax = jnp.concatenate([sigmas_jax, jnp.ones((1,), dtype=self.dtype)])
        else:
            sigmas_jax = jnp.concatenate([sigmas_jax, jnp.zeros((1,), dtype=self.dtype)])

        self.timesteps = timesteps_jax
        self.sigmas = sigmas_jax
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self,
        timestep: float | jax.Array,
        schedule_timesteps: jax.Array | None = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        timestep_arr = np.asarray(timestep)
        schedule_arr = np.asarray(schedule_timesteps)
        indices = np.argwhere(schedule_arr == timestep_arr).flatten()
        if len(indices) == 0:
            return int(np.argmin(np.abs(schedule_arr - timestep_arr)))
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos])

    def _init_step_index(self, timestep: float | jax.Array) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_noise(
        self,
        sample: jax.Array,
        timestep: float | jax.Array,
        noise: jax.Array | None = None,
    ) -> jax.Array:
        if noise is None:
            raise ValueError("`noise` must be provided for `scale_noise`.")

        sigmas = self.sigmas.astype(sample.dtype)
        schedule_timesteps = self.timesteps.astype(sample.dtype)
        timestep = jnp.asarray(timestep, dtype=sample.dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in np.asarray(timestep)
            ]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[jnp.asarray(step_indices)].flatten()
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: jax.Array,
        timestep: float | jax.Array,
        sample: jax.Array,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        noise: jax.Array | None = None,
        rng: jax.Array | None = None,
        per_token_timesteps: jax.Array | None = None,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple[jax.Array]:
        del s_churn, s_tmin, s_tmax, s_noise
        timestep = jnp.asarray(timestep)
        if jnp.issubdtype(timestep.dtype, jnp.integer):
            timestep = timestep.astype(self.timesteps.dtype)

        if self.step_index is None:
            self._init_step_index(timestep)

        sample_f32 = sample.astype(jnp.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = jnp.where(lower_mask, sigmas, 0.0)
            lower_sigmas = jnp.max(lower_sigmas, axis=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = next_sigma - current_sigma
        else:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            if noise is None:
                if rng is None:
                    raise ValueError(
                        "`rng` or explicit `noise` must be provided for stochastic sampling."
                    )
                noise = jax.random.normal(rng, sample_f32.shape, dtype=sample_f32.dtype)
            x0 = sample_f32 - current_sigma * model_output
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = sample_f32 + dt * model_output

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.astype(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def _convert_to_karras(self, in_sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(
        self, in_sigmas: np.ndarray, num_inference_steps: int
    ) -> np.ndarray:
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))

    def _convert_to_beta(
        self,
        in_sigmas: np.ndarray,
        num_inference_steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
    ) -> np.ndarray:
        if scipy is None:
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        return np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ],
            dtype=np.float32,
        )

    def _time_shift_exponential(self, mu: float, sigma: float, t: jax.Array) -> jax.Array:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu: float, sigma: float, t: jax.Array) -> jax.Array:
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self) -> int:
        return self.config.num_train_timesteps
