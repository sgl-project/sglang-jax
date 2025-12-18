# SPDX-License-Identifier: Apache-2.0
# Adapted from sglang's FlowUniPCMultistepScheduler (PyTorch version)
# Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py
#
# This is a JAX/Flax implementation of the UniPC scheduler specifically designed for Flow Matching models.
# Unlike the generic UniPCMultistepScheduler, this version:
# 1. Uses linear interpolation path (Flow Matching) instead of VP-SDE
# 2. Only supports flow_prediction type
# 3. Has simplified initialization without beta schedule parameters
# 4. API compatible with PyTorch version (no explicit state passing)

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass
class FlowSchedulerOutput:
    """Output class for scheduler step."""

    prev_sample: jax.Array


class FlowUniPCMultistepScheduler:
    """
    `FlowUniPCMultistepScheduler` is a JAX implementation of the UniPC scheduler
    specifically designed for Flow Matching diffusion models (e.g., Wan2.1).

    This implementation uses JIT-compiled core computations with a non-JIT wrapper
    to maintain PyTorch-compatible API (no explicit state passing required).

    Key differences from UniPCMultistepScheduler:
    - Uses linear interpolation path: x_t = (1-t)*x_0 + t*noise
    - Only supports flow_prediction type
    - No beta schedule - uses direct sigma computation
    - Simplified _sigma_to_alpha_sigma_t: alpha_t = 1 - sigma, sigma_t = sigma
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float | None = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: tuple[int, ...] = (),
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: str = "zero",
        dtype: jnp.dtype = jnp.float32,
    ):
        # Store config
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = list(disable_corrector)
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.final_sigmas_type = final_sigmas_type
        self.dtype = dtype

        # Validation
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.solver_type = "bh2"
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        if prediction_type != "flow_prediction":
            raise ValueError(
                f"FlowUniPCMultistepScheduler only supports prediction_type='flow_prediction', "
                f"got '{prediction_type}'"
            )

        # Initialize sigmas for Flow Matching
        alphas = jnp.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1]
        sigmas = 1.0 - alphas

        if not use_dynamic_shifting and shift is not None:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas.astype(dtype)
        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

        # Mutable state (will be initialized in set_timesteps)
        self.num_inference_steps: int | None = None
        self.timesteps: jax.Array | None = None
        self._sigmas: jax.Array | None = None  # sigmas for inference
        self.model_outputs: jax.Array | None = None
        self.timestep_list: jax.Array | None = None
        self.lower_order_nums: int = 0
        self.last_sample: jax.Array | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.this_order: int = 0

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Sets the begin index for the scheduler."""
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        """Update the shift value."""
        self.shift = shift

    def time_shift(self, mu: float, sigma: float, t: jax.Array) -> jax.Array:
        """Apply exponential time shifting for dynamic shifting."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int,
        shape: tuple,
        mu: float | None = None,
        shift: float | None = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps: Number of diffusion steps for inference
            shape: Shape of the latent tensor (for initializing model_outputs buffer)
            mu: Dynamic shifting parameter (required if use_dynamic_shifting=True)
            shift: Optional override for the shift parameter
        """
        if self.use_dynamic_shifting and mu is None:
            raise ValueError(
                "You have to pass a value for `mu` when `use_dynamic_shifting` is set to True"
            )

        self.num_inference_steps = num_inference_steps

        # Compute sigmas for inference steps
        sigmas = jnp.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]

        # Apply shifting
        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.shift
            if shift is not None:
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Determine final sigma
        if self.final_sigmas_type == "sigma_min":
            sigma_last = sigmas[-1]
        elif self.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero' or 'sigma_min', "
                f"but got {self.final_sigmas_type}"
            )

        # Compute timesteps from sigmas
        self.timesteps = (sigmas * self.num_train_timesteps).astype(jnp.int32)

        # Append final sigma
        self._sigmas = jnp.concatenate([sigmas, jnp.array([sigma_last])]).astype(jnp.float32)

        # Initialize history buffers
        self.model_outputs = jnp.zeros((self.solver_order, *shape), dtype=self.dtype)
        self.timestep_list = jnp.zeros((self.solver_order,), dtype=jnp.int32)

        # Reset state
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.last_sample = None
        self.this_order = 0

    def _sigma_to_alpha_sigma_t(self, sigma: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Convert sigma to alpha_t and sigma_t for Flow Matching."""
        return 1 - sigma, sigma

    def _index_for_timestep(self, timestep: int | jax.Array) -> int:
        """Gets the step_index for a given timestep."""
        indices = jnp.where(self.timesteps == timestep, size=1, fill_value=-1)[0]
        pos = jnp.where(len(indices) > 1, 1, 0)
        step_index = jnp.where(
            indices[0] == -1,
            len(self.timesteps) - 1,
            indices[pos],
        )
        return int(step_index)

    def _init_step_index(self, timestep: int | jax.Array):
        """Initialize the step_index counter for the scheduler."""
        if self._begin_index is None:
            self._step_index = self._index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # ============== JIT-compiled core computations ==============

    @staticmethod
    @partial(jax.jit, static_argnames=["predict_x0"])
    def _convert_model_output_jit(
        model_output: jax.Array,
        sample: jax.Array,
        sigma: jax.Array,
        predict_x0: bool,
    ) -> jax.Array:
        """JIT-compiled model output conversion."""
        if predict_x0:
            x0_pred = sample - sigma * model_output
            return x0_pred
        else:
            epsilon = sample - (1 - sigma) * model_output
            return epsilon

    @staticmethod
    @partial(jax.jit, static_argnames=["order", "solver_order", "predict_x0", "solver_type"])
    def _uni_p_update_jit(
        sample: jax.Array,
        m0: jax.Array,
        model_outputs: jax.Array,
        sigmas: jax.Array,
        step_index: int,
        order: int,
        solver_order: int,
        predict_x0: bool,
        solver_type: str,
    ) -> jax.Array:
        """JIT-compiled UniP predictor update."""
        x = sample

        sigma_t_val = sigmas[step_index + 1]
        sigma_s0_val = sigmas[step_index]

        # Flow Matching: alpha_t = 1 - sigma, sigma_t = sigma
        alpha_t, sigma_t = 1 - sigma_t_val, sigma_t_val
        alpha_s0, sigma_s0 = 1 - sigma_s0_val, sigma_s0_val

        lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
        lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)

        h = lambda_t - lambda_s0

        # Build rks and D1s
        def rk_d1_loop_body(i, carry):
            rks, D1s = carry
            history_idx = solver_order - 2 - i
            mi = model_outputs[history_idx]

            sigma_si = sigmas[step_index - 1 - i]
            alpha_si, sigma_si_t = 1 - sigma_si, sigma_si
            lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si_t + 1e-10)

            rk = (lambda_si - lambda_s0) / h
            Di = (mi - m0) / rk

            rks = rks.at[i].set(rk)
            D1s = D1s.at[i].set(Di)
            return rks, D1s

        rks_init = jnp.zeros(solver_order, dtype=h.dtype)
        D1s_shape = (solver_order - 1,) + m0.shape if solver_order > 1 else (1,) + m0.shape
        D1s_init = jnp.zeros(D1s_shape, dtype=m0.dtype)

        rks, D1s = jax.lax.fori_loop(0, order - 1, rk_d1_loop_body, (rks_init, D1s_init))
        rks = rks.at[order - 1].set(1.0)

        hh = -h if predict_x0 else h
        h_phi_1 = jnp.expm1(hh)

        B_h = hh if solver_type == "bh1" else jnp.expm1(hh)

        # Build R and b matrices
        def rb_loop_body(i, carry):
            R, b, current_h_phi_k, factorial_val = carry
            R = R.at[i].set(jnp.power(rks, i))
            b = b.at[i].set(current_h_phi_k * factorial_val / B_h)

            next_fac = factorial_val * (i + 2)
            next_h_phi_k = current_h_phi_k / hh - 1.0 / next_fac

            current_h_phi_k = jnp.where(i < order - 1, next_h_phi_k, current_h_phi_k)
            factorial_val = jnp.where(i < order - 1, next_fac, factorial_val)

            return R, b, current_h_phi_k, factorial_val

        R_init = jnp.zeros((solver_order, solver_order), dtype=h.dtype)
        b_init = jnp.zeros(solver_order, dtype=h.dtype)
        init_h_phi_k = h_phi_1 / hh - 1.0
        init_factorial = 1.0

        R, b, _, _ = jax.lax.fori_loop(
            0, order, rb_loop_body, (R_init, b_init, init_h_phi_k, init_factorial)
        )

        # Solve for rhos_p
        def solve_for_rhos_p(R_mat, b_vec, current_order):
            mask_size = solver_order - 1
            mask = jnp.arange(mask_size) < (current_order - 1)
            mask_2d = mask[:, None] & mask[None, :]

            R_safe = jnp.where(
                mask_2d,
                R_mat[:mask_size, :mask_size],
                jnp.eye(mask_size, dtype=R_mat.dtype),
            )
            b_safe = jnp.where(mask, b_vec[:mask_size], 0.0)

            solved_rhos = jnp.linalg.solve(R_safe, b_safe)
            return jnp.where(mask, solved_rhos, 0.0)

        rhos_p_order2 = jnp.zeros(max(solver_order - 1, 1), dtype=x.dtype)
        rhos_p_order2 = rhos_p_order2.at[0].set(0.5) if solver_order > 1 else rhos_p_order2

        rhos_p_general = solve_for_rhos_p(R, b, order)
        rhos_p = jnp.where(order == 2, rhos_p_order2, rhos_p_general)

        # Compute prediction residual
        pred_res = jnp.where(
            order > 1,
            jnp.einsum("k,k...->...", rhos_p[: order - 1], D1s[: order - 1]),
            jnp.zeros_like(x),
        )

        if predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.astype(x.dtype)

    @staticmethod
    @partial(jax.jit, static_argnames=["order", "solver_order", "predict_x0", "solver_type"])
    def _uni_c_update_jit(
        this_model_output: jax.Array,
        last_sample: jax.Array,
        this_sample: jax.Array,
        m0: jax.Array,
        model_outputs: jax.Array,
        sigmas: jax.Array,
        step_index: int,
        order: int,
        solver_order: int,
        predict_x0: bool,
        solver_type: str,
    ) -> jax.Array:
        """JIT-compiled UniC corrector update."""
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t_val = sigmas[step_index]
        sigma_s0_val = sigmas[step_index - 1]

        alpha_t, sigma_t = 1 - sigma_t_val, sigma_t_val
        alpha_s0, sigma_s0 = 1 - sigma_s0_val, sigma_s0_val

        lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
        lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)

        h = lambda_t - lambda_s0

        # Fast path for order 1
        def build_order1():
            rks = jnp.ones(solver_order, dtype=h.dtype)
            D1s = jnp.zeros((max(solver_order - 1, 1),) + m0.shape, dtype=m0.dtype)
            return rks, D1s

        # Fast path for order 2
        def build_order2():
            si = step_index - 2
            mi = model_outputs[solver_order - 2]
            sigma_si = sigmas[si]
            alpha_si, sigma_si_t = 1 - sigma_si, sigma_si
            lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si_t + 1e-10)
            rk = (lambda_si - lambda_s0) / h
            rks = jnp.zeros(solver_order, dtype=h.dtype)
            rks = rks.at[0].set(rk).at[1].set(1.0)
            D1s = ((mi - m0) / rk)[None, ...]
            return rks, D1s

        # General case
        def build_general():
            def rk_d1_loop_body(i, carry):
                rks, D1s = carry
                history_idx = solver_order - (i + 2)
                mi = model_outputs[history_idx]

                sigma_si = sigmas[step_index - (i + 2)]
                alpha_si, sigma_si_t = 1 - sigma_si, sigma_si
                lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si_t + 1e-10)

                rk = (lambda_si - lambda_s0) / h
                Di = (mi - m0) / rk

                rks = rks.at[i].set(rk)
                D1s = D1s.at[i].set(Di)
                return rks, D1s

            rks_init = jnp.zeros(solver_order, dtype=h.dtype)
            D1s_init = jnp.zeros((max(solver_order - 1, 1),) + m0.shape, dtype=m0.dtype)

            rks, D1s = jax.lax.fori_loop(0, order - 1, rk_d1_loop_body, (rks_init, D1s_init))
            rks = rks.at[order - 1].set(1.0)
            return rks, D1s

        rks, D1s = jax.lax.cond(
            order == 1,
            build_order1,
            lambda: jax.lax.cond(order == 2, build_order2, build_general),
        )

        hh = -h if predict_x0 else h
        h_phi_1 = jnp.expm1(hh)
        B_h = hh if solver_type == "bh1" else jnp.expm1(hh)

        # Build R and b
        def rb_loop_body(i, carry):
            R, b, current_h_phi_k, factorial_val = carry
            R = R.at[i].set(jnp.power(rks, i))
            b = b.at[i].set(current_h_phi_k * factorial_val / B_h)

            next_fac = factorial_val * (i + 2)
            next_h_phi_k = current_h_phi_k / hh - 1.0 / next_fac

            current_h_phi_k = jnp.where(i < order - 1, next_h_phi_k, current_h_phi_k)
            factorial_val = jnp.where(i < order - 1, next_fac, factorial_val)

            return R, b, current_h_phi_k, factorial_val

        R_init = jnp.zeros((solver_order, solver_order), dtype=h.dtype)
        b_init = jnp.zeros(solver_order, dtype=h.dtype)
        init_h_phi_k = h_phi_1 / hh - 1.0
        init_factorial = 1.0

        R, b, _, _ = jax.lax.fori_loop(
            0, order, rb_loop_body, (R_init, b_init, init_h_phi_k, init_factorial)
        )

        # Solve for rhos_c with fast paths
        def solve_order1():
            rhos = jnp.zeros(solver_order, dtype=x_t.dtype)
            return rhos.at[0].set(0.5)

        def solve_order2():
            rk = rks[0]
            det = 1 - rk
            rhos_c_0 = (b[0] - b[1]) / det
            rhos_c_1 = (b[1] - rk * b[0]) / det
            rhos = jnp.zeros(solver_order, dtype=x_t.dtype)
            return rhos.at[0].set(rhos_c_0).at[1].set(rhos_c_1)

        def solve_general():
            mask = jnp.arange(solver_order) < order
            mask_2d = mask[:, None] & mask[None, :]
            R_safe = jnp.where(mask_2d, R, jnp.eye(solver_order, dtype=R.dtype))
            b_safe = jnp.where(mask, b, 0.0)
            solved_rhos = jnp.linalg.solve(R_safe, b_safe)
            return jnp.where(mask, solved_rhos, 0.0)

        rhos_c = jax.lax.cond(
            order == 1,
            solve_order1,
            lambda: jax.lax.cond(order == 2, solve_order2, solve_general),
        )

        D1_t = model_t - m0

        # Compute correction residual
        corr_res = jnp.where(
            order > 1,
            jnp.einsum("k,k...->...", rhos_c[: order - 1], D1s[: order - 1]),
            jnp.zeros_like(D1_t),
        )

        final_rho = rhos_c[order - 1]

        if predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + final_rho * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + final_rho * D1_t)

        return x_t.astype(x.dtype)

    # ============== Public API (non-JIT wrappers) ==============

    def step(
        self,
        model_output: jax.Array,
        timestep: int | jax.Array,
        sample: jax.Array,
        return_dict: bool = True,
    ) -> FlowSchedulerOutput | tuple[jax.Array]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the sample with the multistep UniPC.

        API is compatible with PyTorch version - no explicit state passing required.

        Args:
            model_output: Direct output from the diffusion model (flow prediction)
            timestep: Current discrete timestep from the scheduler's sequence
            sample: Current noisy sample (latent)
            return_dict: Whether to return FlowSchedulerOutput or tuple

        Returns:
            FlowSchedulerOutput with prev_sample, or tuple (prev_sample,)
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        sample = sample.astype(jnp.float32)
        timestep_scalar = int(timestep) if not isinstance(timestep, int) else timestep

        # Initialize step_index if it's the first step
        if self._step_index is None:
            self._init_step_index(timestep_scalar)

        # Determine if corrector should be used
        use_corrector = (
            self._step_index > 0
            and (self._step_index - 1) not in self.disable_corrector
            and self.last_sample is not None
        )

        # Convert model_output to x0_pred or epsilon_pred (JIT compiled)
        sigma = self._sigmas[self._step_index]
        model_output_for_history = self._convert_model_output_jit(
            model_output, sample, sigma, self.predict_x0
        )

        # Apply corrector if applicable
        if use_corrector:
            m0 = self.model_outputs[self.solver_order - 1]
            sample = self._uni_c_update_jit(
                this_model_output=model_output_for_history,
                last_sample=self.last_sample,
                this_sample=sample,
                m0=m0,
                model_outputs=self.model_outputs,
                sigmas=self._sigmas,
                step_index=self._step_index,
                order=self.this_order,
                solver_order=self.solver_order,
                predict_x0=self.predict_x0,
                solver_type=self.solver_type,
            )

        # Update history buffers
        if self._step_index == 0:
            self.model_outputs = self.model_outputs.at[-1].set(model_output_for_history)
            self.timestep_list = self.timestep_list.at[-1].set(timestep_scalar)
        else:
            self.model_outputs = jnp.roll(self.model_outputs, shift=-1, axis=0)
            self.model_outputs = self.model_outputs.at[-1].set(model_output_for_history)
            self.timestep_list = jnp.roll(self.timestep_list, shift=-1)
            self.timestep_list = self.timestep_list.at[-1].set(timestep_scalar)

        # Determine the order for the current step
        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self.solver_order

        # Warmup for multistep
        self.this_order = min(this_order, self.lower_order_nums + 1)

        # Store current sample for next step's corrector
        self.last_sample = sample

        # UniP predictor step (JIT compiled)
        m0 = self.model_outputs[self.solver_order - 1]
        prev_sample = self._uni_p_update_jit(
            sample=sample,
            m0=m0,
            model_outputs=self.model_outputs,
            sigmas=self._sigmas,
            step_index=self._step_index,
            order=self.this_order,
            solver_order=self.solver_order,
            predict_x0=self.predict_x0,
            solver_type=self.solver_type,
        )

        # Update lower_order_nums for warmup
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # Increment step index
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowSchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: jax.Array, *args, **kwargs) -> jax.Array:
        """UniPC does not scale model input, returns sample unchanged."""
        return sample

    def add_noise(
        self,
        original_samples: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        """
        Add noise to samples using Flow Matching formulation.

        For Flow Matching: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
        """
        sigmas = self._sigmas if self._sigmas is not None else self.sigmas

        # Get sigma values for the given timesteps
        sigma = sigmas[timesteps].flatten()

        # Broadcast sigma to match sample shape
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma[..., None]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self) -> int:
        return self.num_train_timesteps
