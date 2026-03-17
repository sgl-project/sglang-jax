from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler as HFFlowMatchEulerDiscreteScheduler,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFFlowMatchEulerDiscreteScheduler = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

if jax is not None:
    from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_match_euler_discrete_scheduler import (
        FlowMatchEulerDiscreteScheduler as JaxFlowMatchEulerDiscreteScheduler,
    )


FLUX1_SCHEDULER_ROOT = Path(os.environ.get("FLUX1_SCHEDULER_ROOT", "/models/FLUX1.0"))


@unittest.skipIf(
    torch is None or HFFlowMatchEulerDiscreteScheduler is None or jax is None,
    "torch/diffusers/jax not installed",
)
class TestFlowMatchEulerDiscreteSchedulerParity(unittest.TestCase):
    def _build_pair(self, **kwargs):
        torch_scheduler = HFFlowMatchEulerDiscreteScheduler(**kwargs)
        jax_scheduler = JaxFlowMatchEulerDiscreteScheduler(**kwargs)
        return torch_scheduler, jax_scheduler

    def _run_rollout_pair(
        self,
        *,
        scheduler_kwargs: dict,
        num_inference_steps: int,
        sample_shape: tuple[int, ...],
    ):
        torch_scheduler, jax_scheduler = self._build_pair(**scheduler_kwargs)
        torch_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        jax_scheduler.set_timesteps(num_inference_steps=num_inference_steps, shape=sample_shape)

        torch.manual_seed(0)
        sample_torch = torch.randn(*sample_shape, dtype=torch.float32)
        sample_jax = jnp.asarray(sample_torch.cpu().numpy())

        # Mimic runner behavior: integer timesteps and channel-first latent layout.
        for step in range(num_inference_steps):
            model_output_torch = torch.randn(*sample_shape, dtype=torch.float32)
            model_output_jax = jnp.asarray(model_output_torch.cpu().numpy())

            timestep_torch = torch_scheduler.timesteps[step]
            timestep_jax = jnp.array(jax_scheduler.timesteps, dtype=jnp.int32)[step]

            sample_torch = torch_scheduler.step(
                model_output=model_output_torch,
                timestep=timestep_torch,
                sample=sample_torch,
                return_dict=False,
            )[0]
            sample_jax = jax_scheduler.step(
                model_output=model_output_jax,
                timestep=timestep_jax,
                sample=sample_jax,
                return_dict=False,
            )[0]

        return sample_torch.cpu().numpy(), np.asarray(sample_jax)

    def test_initial_schedule_parity(self):
        torch_scheduler, jax_scheduler = self._build_pair(shift=1.0)
        np.testing.assert_allclose(
            torch_scheduler.timesteps.cpu().numpy(),
            np.asarray(jax_scheduler.timesteps),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            torch_scheduler.sigmas.cpu().numpy(),
            np.asarray(jax_scheduler.sigmas),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(jax_scheduler.config.shift, 1.0)
        self.assertEqual(jax_scheduler.config.num_train_timesteps, 1000)

    def test_set_timesteps_and_scale_noise_parity(self):
        torch_scheduler, jax_scheduler = self._build_pair(shift=1.0)
        torch_scheduler.set_timesteps(num_inference_steps=6)
        jax_scheduler.set_timesteps(num_inference_steps=6, shape=(2, 3, 4, 4))

        np.testing.assert_allclose(
            torch_scheduler.timesteps.cpu().numpy(),
            np.asarray(jax_scheduler.timesteps),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            torch_scheduler.sigmas.cpu().numpy(),
            np.asarray(jax_scheduler.sigmas),
            atol=1e-6,
            rtol=1e-6,
        )

        sample_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        noise_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        timestep_torch = torch_scheduler.timesteps[:2]

        scaled_torch = torch_scheduler.scale_noise(sample_torch, timestep_torch, noise_torch)
        scaled_jax = jax_scheduler.scale_noise(
            sample=jnp.asarray(sample_torch.cpu().numpy()),
            timestep=jnp.asarray(timestep_torch.cpu().numpy()),
            noise=jnp.asarray(noise_torch.cpu().numpy()),
        )

        np.testing.assert_allclose(
            scaled_torch.cpu().numpy(),
            np.asarray(scaled_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_step_parity(self):
        torch_scheduler, jax_scheduler = self._build_pair(shift=1.0)
        torch_scheduler.set_timesteps(num_inference_steps=4)
        jax_scheduler.set_timesteps(num_inference_steps=4, shape=(2, 3, 4, 4))

        model_output_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        sample_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        timestep_torch = torch_scheduler.timesteps[0]

        out_torch = torch_scheduler.step(
            model_output=model_output_torch,
            timestep=timestep_torch,
            sample=sample_torch,
            return_dict=True,
        ).prev_sample
        out_jax = jax_scheduler.step(
            model_output=jnp.asarray(model_output_torch.cpu().numpy()),
            timestep=jnp.asarray(timestep_torch.cpu().numpy()),
            sample=jnp.asarray(sample_torch.cpu().numpy()),
            return_dict=True,
        ).prev_sample

        np.testing.assert_allclose(
            out_torch.cpu().numpy(),
            np.asarray(out_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_full_rollout_runner_style_parity(self):
        torch_final, jax_final = self._run_rollout_pair(
            scheduler_kwargs={"shift": 1.0},
            num_inference_steps=6,
            sample_shape=(2, 4, 3, 8, 8),
        )
        np.testing.assert_allclose(
            torch_final,
            jax_final,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_full_rollout_nondefault_config_parity(self):
        torch_final, jax_final = self._run_rollout_pair(
            scheduler_kwargs={
                "shift": 1.1,
                "invert_sigmas": True,
                "use_karras_sigmas": True,
            },
            num_inference_steps=5,
            sample_shape=(1, 8, 2, 4, 4),
        )
        np.testing.assert_allclose(
            torch_final,
            jax_final,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dynamic_shifting_rollout_parity(self):
        scheduler_kwargs = {
            "use_dynamic_shifting": True,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
        }
        image_seq_len = 1024
        _, jax_scheduler = self._build_pair(**scheduler_kwargs)
        mu = jax_scheduler.calculate_mu(image_seq_len)

        torch_scheduler, jax_scheduler = self._build_pair(**scheduler_kwargs)
        torch_scheduler.set_timesteps(num_inference_steps=5, mu=mu)
        jax_scheduler.set_timesteps(num_inference_steps=5, shape=(1, 4, 2, 4, 4), mu=mu)

        torch.manual_seed(0)
        sample_torch = torch.randn(1, 4, 2, 4, 4, dtype=torch.float32)
        sample_jax = jnp.asarray(sample_torch.cpu().numpy())

        for step in range(5):
            model_output_torch = torch.randn(1, 4, 2, 4, 4, dtype=torch.float32)
            model_output_jax = jnp.asarray(model_output_torch.cpu().numpy())

            sample_torch = torch_scheduler.step(
                model_output=model_output_torch,
                timestep=torch_scheduler.timesteps[step],
                sample=sample_torch,
                return_dict=False,
            )[0]
            sample_jax = jax_scheduler.step(
                model_output=model_output_jax,
                timestep=jnp.asarray(jax_scheduler.timesteps[step]),
                sample=sample_jax,
                return_dict=False,
            )[0]

        np.testing.assert_allclose(
            sample_torch.cpu().numpy(),
            np.asarray(sample_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_stochastic_sampling_step_parity(self):
        torch_scheduler, jax_scheduler = self._build_pair(stochastic_sampling=True, shift=1.0)
        torch_scheduler.set_timesteps(num_inference_steps=4)
        jax_scheduler.set_timesteps(num_inference_steps=4, shape=(2, 3, 4, 4))

        model_output_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        sample_torch = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        timestep_torch = torch_scheduler.timesteps[0]

        torch.manual_seed(123)
        noise_torch = torch.randn_like(sample_torch)
        torch.manual_seed(123)
        out_torch = torch_scheduler.step(
            model_output=model_output_torch,
            timestep=timestep_torch,
            sample=sample_torch,
            return_dict=True,
        ).prev_sample

        out_jax = jax_scheduler.step(
            model_output=jnp.asarray(model_output_torch.cpu().numpy()),
            timestep=jnp.asarray(timestep_torch.cpu().numpy()),
            sample=jnp.asarray(sample_torch.cpu().numpy()),
            noise=jnp.asarray(noise_torch.cpu().numpy()),
            return_dict=True,
        ).prev_sample

        np.testing.assert_allclose(
            out_torch.cpu().numpy(),
            np.asarray(out_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_per_token_timesteps_matches_global_step_when_uniform(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(num_inference_steps=4, shape=(2, 4, 3))

        model_output = jnp.asarray(np.random.randn(2, 4, 3).astype(np.float32))
        sample = jnp.asarray(np.random.randn(2, 4, 3).astype(np.float32))
        timestep = scheduler.timesteps[0]

        out_global = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=True,
        ).prev_sample

        scheduler.set_timesteps(num_inference_steps=4, shape=(2, 4, 3))
        per_token_timesteps = jnp.full((2, 4), timestep, dtype=jnp.float32)
        out_per_token = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            per_token_timesteps=per_token_timesteps,
            return_dict=True,
        ).prev_sample

        np.testing.assert_allclose(
            np.asarray(out_global),
            np.asarray(out_per_token),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_integer_timestep_compatibility(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(num_inference_steps=4, shape=(2, 3, 4, 4))

        model_output = jnp.ones((2, 3, 4, 4), dtype=jnp.float32)
        sample = jnp.zeros((2, 3, 4, 4), dtype=jnp.float32)
        timestep = scheduler.timesteps[0]

        out_float = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=True,
        ).prev_sample

        scheduler.set_timesteps(num_inference_steps=4, shape=(2, 3, 4, 4))
        out_int = scheduler.step(
            model_output=model_output,
            timestep=timestep.astype(jnp.int32),
            sample=sample,
            return_dict=True,
        ).prev_sample

        np.testing.assert_allclose(
            np.asarray(out_float),
            np.asarray(out_int),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_integer_timestep_compatibility_with_noninteger_schedule(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(shift=1.1, use_karras_sigmas=True)
        scheduler.set_timesteps(num_inference_steps=5, shape=(2, 3, 4, 4))

        model_output = jnp.ones((2, 3, 4, 4), dtype=jnp.float32)
        sample = jnp.zeros((2, 3, 4, 4), dtype=jnp.float32)
        timestep = scheduler.timesteps[1]

        out_float = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=True,
        ).prev_sample

        scheduler.set_timesteps(num_inference_steps=5, shape=(2, 3, 4, 4))
        out_int = scheduler.step(
            model_output=model_output,
            timestep=timestep.astype(jnp.int32),
            sample=sample,
            return_dict=True,
        ).prev_sample

        np.testing.assert_allclose(
            np.asarray(out_float),
            np.asarray(out_int),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_config_round_trip(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(
            shift=1.15,
            use_dynamic_shifting=True,
            base_shift=0.7,
            max_shift=1.3,
        )

        rebuilt = JaxFlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
        self.assertEqual(rebuilt.config.shift, 1.15)
        self.assertTrue(rebuilt.config.use_dynamic_shifting)
        self.assertEqual(rebuilt.config.base_shift, 0.7)
        self.assertEqual(rebuilt.config.max_shift, 1.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded = JaxFlowMatchEulerDiscreteScheduler.load_config(tmpdir)

        self.assertEqual(loaded["shift"], 1.15)
        self.assertTrue(loaded["use_dynamic_shifting"])
        self.assertEqual(loaded["_class_name"], "FlowMatchEulerDiscreteScheduler")

    def test_scheduler_mixin_pretrained_round_trip(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(shift=1.05, invert_sigmas=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_pretrained(tmpdir)
            rebuilt = JaxFlowMatchEulerDiscreteScheduler.from_pretrained(tmpdir)

        self.assertEqual(rebuilt.config.shift, 1.05)
        self.assertTrue(rebuilt.config.invert_sigmas)
        self.assertIn(JaxFlowMatchEulerDiscreteScheduler, rebuilt.compatibles)

    def test_from_local_hf_scheduler_config_rollout(self):
        torch_scheduler = HFFlowMatchEulerDiscreteScheduler(
            shift=1.1,
            invert_sigmas=True,
            use_karras_sigmas=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_scheduler.save_pretrained(tmpdir)
            config_path = Path(tmpdir) / "scheduler_config.json"
            self.assertTrue(config_path.is_file())

            jax_scheduler = JaxFlowMatchEulerDiscreteScheduler.from_pretrained(tmpdir)

        torch_scheduler.set_timesteps(num_inference_steps=4)
        jax_scheduler.set_timesteps(num_inference_steps=4, shape=(1, 4, 2, 4, 4))

        torch.manual_seed(0)
        sample_torch = torch.randn(1, 4, 2, 4, 4, dtype=torch.float32)
        sample_jax = jnp.asarray(sample_torch.cpu().numpy())

        for step in range(4):
            model_output_torch = torch.randn(1, 4, 2, 4, 4, dtype=torch.float32)
            model_output_jax = jnp.asarray(model_output_torch.cpu().numpy())

            sample_torch = torch_scheduler.step(
                model_output=model_output_torch,
                timestep=torch_scheduler.timesteps[step],
                sample=sample_torch,
                return_dict=False,
            )[0]
            sample_jax = jax_scheduler.step(
                model_output=model_output_jax,
                timestep=jnp.asarray(jax_scheduler.timesteps[step]),
                sample=sample_jax,
                return_dict=False,
            )[0]

        np.testing.assert_allclose(
            sample_torch.cpu().numpy(),
            np.asarray(sample_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    @unittest.skipUnless(
        (FLUX1_SCHEDULER_ROOT / "scheduler" / "scheduler_config.json").is_file(),
        "local FLUX scheduler config not found",
    )
    def test_from_real_flux_scheduler_config_rollout(self):
        image_seq_len = 4096
        sample_shape = (2, 16, 4, 16, 16)
        num_inference_steps = 8
        torch_scheduler = HFFlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX1_SCHEDULER_ROOT,
            subfolder="scheduler",
        )
        jax_scheduler = JaxFlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX1_SCHEDULER_ROOT,
            subfolder="scheduler",
        )

        self.assertEqual(torch_scheduler.config.shift, 3.0)
        self.assertEqual(jax_scheduler.config.shift, 3.0)
        self.assertTrue(torch_scheduler.config.use_dynamic_shifting)
        self.assertTrue(jax_scheduler.config.use_dynamic_shifting)

        mu = jax_scheduler.calculate_mu(image_seq_len)
        torch_scheduler.set_timesteps(num_inference_steps=num_inference_steps, mu=mu)
        jax_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            shape=sample_shape,
            mu=mu,
        )

        np.testing.assert_allclose(
            torch_scheduler.timesteps.cpu().numpy(),
            np.asarray(jax_scheduler.timesteps),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            torch_scheduler.sigmas.cpu().numpy(),
            np.asarray(jax_scheduler.sigmas),
            atol=1e-6,
            rtol=1e-6,
        )

        torch.manual_seed(0)
        sample_torch = torch.randn(*sample_shape, dtype=torch.float32)
        sample_jax = jnp.asarray(sample_torch.cpu().numpy())

        for step in range(num_inference_steps):
            model_output_torch = torch.randn(*sample_shape, dtype=torch.float32)
            model_output_jax = jnp.asarray(model_output_torch.cpu().numpy())

            sample_torch = torch_scheduler.step(
                model_output=model_output_torch,
                timestep=torch_scheduler.timesteps[step],
                sample=sample_torch,
                return_dict=False,
            )[0]
            sample_jax = jax_scheduler.step(
                model_output=model_output_jax,
                timestep=jnp.asarray(jax_scheduler.timesteps[step]),
                sample=sample_jax,
                return_dict=False,
            )[0]

        np.testing.assert_allclose(
            sample_torch.cpu().numpy(),
            np.asarray(sample_jax),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_scheduler_output_base_output_behavior(self):
        scheduler = JaxFlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(num_inference_steps=4, shape=(1, 1, 2, 2))
        out = scheduler.step(
            model_output=jnp.ones((1, 1, 2, 2), dtype=jnp.float32),
            timestep=scheduler.timesteps[0],
            sample=jnp.zeros((1, 1, 2, 2), dtype=jnp.float32),
            return_dict=True,
        )

        self.assertTrue(isinstance(out, dict))
        np.testing.assert_allclose(np.asarray(out["prev_sample"]), np.asarray(out.prev_sample))
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(out.prev_sample))


if __name__ == "__main__":
    unittest.main()
