import json
import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Add python directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)
from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import WanTransformer3DModel
from sgl_jax.srt.utils.jax_utils import device_array


def compare_arrays(name, jax_arr, torch_arr, rtol=0.05, atol=0.1):
    """Compare two arrays and print detailed diff info."""
    jax_arr = np.asarray(jax_arr)
    torch_arr = np.asarray(torch_arr)

    diff = np.abs(jax_arr - torch_arr)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Find location of max diff
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    print(f"\n=== {name} ===")
    print(f"  JAX shape: {jax_arr.shape}, Torch shape: {torch_arr.shape}")
    print(f"  JAX  - mean: {jax_arr.mean():.6f}, std: {jax_arr.std():.6f}")
    print(f"  Torch- mean: {torch_arr.mean():.6f}, std: {torch_arr.std():.6f}")
    print(f"  Max diff: {max_diff:.6f} at {max_idx}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  JAX value at max: {jax_arr[max_idx]:.6f}, Torch value: {torch_arr[max_idx]:.6f}")

    is_close = np.allclose(jax_arr, torch_arr, rtol=rtol, atol=atol)
    print(f"  Match (rtol={rtol}, atol={atol}): {'PASS' if is_close else 'FAIL'}")

    return is_close, max_diff, mean_diff


class TestDiffusionPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load reference data and initialize runner once."""
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

        # Load debug JSON
        debug_json_path = os.path.join(cls.project_root, "DenoisingSteps_debug.json")
        if not os.path.exists(debug_json_path):
            raise unittest.SkipTest(f"Debug JSON not found: {debug_json_path}")

        with open(debug_json_path) as f:
            cls.debug_data = json.load(f)

        print(f"Loaded debug data with {len(cls.debug_data['steps'])} steps")
        print(f"Config: {cls.debug_data['config']}")

        # Load npy files
        try:
            cls.prompt_embeds = np.load(os.path.join(cls.project_root, "test_prompt_embeds.npy"))
            cls.negative_prompt_embeds = np.load(
                os.path.join(cls.project_root, "test_negative_prompt_embeds.npy")
            )
            cls.latents_in = np.load(os.path.join(cls.project_root, "test_latents_in.npy"))
            cls.latents_out_expected = np.load(
                os.path.join(cls.project_root, "test_latents_out.npy")
            )
        except FileNotFoundError as e:
            raise e

        # Setup server args and mesh
        cls.server_args = MultimodalServerArgs(
            model_path="/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/",
            download_dir="/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/",
        )

        devices = jax.devices()
        if len(devices) > 1:
            cls.mesh = Mesh(np.array(devices).reshape(1, -1), ("data", "model"))
        else:
            cls.mesh = Mesh(np.array([[devices[0]]]), ("data", "model"))

        # Initialize runner
        print("Initializing DiffusionModelRunner...")
        cls.runner = DiffusionModelRunner(
            server_args=cls.server_args,
            mesh=cls.mesh,
            model_class=WanTransformer3DModel,
        )

    def test_01_config_comparison(self):
        """Test 1: Compare scheduler config."""
        print("\n" + "=" * 60)
        print("TEST 1: Config Comparison")
        print("=" * 60)

        config = self.debug_data["config"]

        # Compare timesteps
        torch_timesteps = config["timesteps"]
        test_shape = (1, 16, 2, 8, 8)
        self.runner.solver.set_timesteps(num_inference_steps=len(torch_timesteps), shape=test_shape)
        jax_timesteps = list(self.runner.solver.timesteps)

        print(f"Torch timesteps: {torch_timesteps}")
        print(f"JAX timesteps:   {jax_timesteps}")

        self.assertEqual(torch_timesteps, jax_timesteps, "Timesteps mismatch!")

        # Compare sigmas
        torch_sigmas = config["sigmas"]
        jax_sigmas = list(self.runner.solver._sigmas)

        print(f"Torch sigmas: {torch_sigmas}")
        print(f"JAX sigmas:   {jax_sigmas}")

        np.testing.assert_allclose(
            jax_sigmas, torch_sigmas, rtol=1e-5, atol=1e-5, err_msg="Sigmas mismatch!"
        )
        print("Config comparison PASSED!")

    def test_02_step_by_step_comparison(self):
        """Test 2: Step-by-step comparison of denoising loop."""
        print("\n" + "=" * 60)
        print("TEST 2: Step-by-Step Denoising Comparison")
        print("=" * 60)

        config = self.debug_data["config"]
        steps = self.debug_data["steps"]
        num_inference_steps = config["num_inference_steps"]
        guidance_scale = config["guidance_scale"]
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare prompt embeddings
        prompt_embeds = np.squeeze(
            self.prompt_embeds, axis=1
        )  # (1, 1, 512, 4096) -> (1, 512, 4096)
        negative_prompt_embeds = np.squeeze(self.negative_prompt_embeds, axis=1)

        if do_classifier_free_guidance:
            prompt_embeds_combined = np.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
        else:
            prompt_embeds_combined = prompt_embeds

        text_embeds = device_array(
            jnp.array(prompt_embeds_combined), sharding=NamedSharding(self.mesh, PartitionSpec())
        )
        # Note: text_embeds should be (B, L, D), no transpose needed

        # Prepare initial latents - use latents_before from step 0
        # Torch format: (B, C, T, H, W), JAX format: (B, T, H, W, C)
        torch_latents_before_step0 = np.array(steps[0]["latents_before"]["data"])
        print(f"Torch latents_before shape (step 0): {torch_latents_before_step0.shape}")

        # Transpose from (B, C, T, H, W) to (B, T, H, W, C)
        latents = np.transpose(torch_latents_before_step0, (0, 2, 3, 4, 1))
        latents = device_array(
            jnp.array(latents), sharding=NamedSharding(self.mesh, PartitionSpec())
        )

        # Setup solver
        self.runner.solver.set_timesteps(
            num_inference_steps=num_inference_steps,
            shape=torch_latents_before_step0.shape,  # (B, C, T, H, W)
        )
        self.runner.solver.set_begin_index(0)

        all_passed = True
        first_fail_step = None

        for step_idx, step_data in enumerate(steps):
            print(f"\n--- Step {step_idx} (timestep={step_data['timestep']}) ---")

            # Get reference data from torch
            torch_latents_before = np.array(step_data["latents_before"]["data"])
            torch_noise_pred = np.array(step_data["noise_pred_final"]["data"])
            torch_latents_after = np.array(step_data["latents_after"]["data"])

            torch_noise_cond = (
                np.array(step_data["noise_pred_cond"]["data"])
                if "noise_pred_cond" in step_data
                else None
            )
            torch_noise_uncond = (
                np.array(step_data["noise_pred_uncond"]["data"])
                if "noise_pred_uncond" in step_data
                else None
            )

            # Compare latents_before (should match if previous step was correct)
            jax_latents_before = np.transpose(
                np.asarray(latents), (0, 4, 1, 2, 3)
            )  # to (B, C, T, H, W)

            # BF16 has limited precision, use relaxed tolerance
            is_close, max_diff, _ = compare_arrays(
                f"Step {step_idx} latents_before",
                jax_latents_before,
                torch_latents_before,
                rtol=0.05,
                atol=0.1,
            )

            if not is_close:
                print(f"  *** latents_before MISMATCH at step {step_idx}! ***")
                if first_fail_step is None:
                    first_fail_step = step_idx
                all_passed = False
                # Continue to see where the error originated

            # Run one denoising step
            t_scalar = jnp.array(self.runner.solver.timesteps, dtype=jnp.int32)[step_idx]

            if do_classifier_free_guidance:
                latents_for_model = jnp.concatenate([latents] * 2, axis=0)
            else:
                latents_for_model = latents

            t_batch = jnp.broadcast_to(t_scalar, (latents_for_model.shape[0],))
            latents_cf = latents_for_model.transpose(
                0, 4, 1, 2, 3
            )  # (B, T, H, W, C) -> (B, C, T, H, W)

            # Model forward
            noise_pred = self.runner.jitted_forward(
                hidden_states=latents_cf,
                encoder_hidden_states=text_embeds,
                timesteps=t_batch,
                encoder_hidden_states_image=None,
                guidance_scale=None,
            )

            # CFG
            if do_classifier_free_guidance:
                bsz = latents_for_model.shape[0] // 2
                noise_uncond = noise_pred[bsz:]
                noise_cond = noise_pred[:bsz]
                noise_pred_final = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = latents[:bsz] if latents.shape[0] > bsz else latents

                # Compare noise_pred_cond and noise_pred_uncond
                if torch_noise_cond is not None:
                    compare_arrays(
                        f"Step {step_idx} noise_pred_cond",
                        np.asarray(noise_cond),
                        torch_noise_cond,
                        rtol=0.05,
                        atol=0.1,
                    )
                if torch_noise_uncond is not None:
                    compare_arrays(
                        f"Step {step_idx} noise_pred_uncond",
                        np.asarray(noise_uncond),
                        torch_noise_uncond,
                        rtol=0.05,
                        atol=0.1,
                    )
            else:
                noise_pred_final = noise_pred

            # Compare noise_pred_final
            is_close_noise, max_diff_noise, _ = compare_arrays(
                f"Step {step_idx} noise_pred_final",
                np.asarray(noise_pred_final),
                torch_noise_pred,
                rtol=0.05,
                atol=0.1,
            )

            if not is_close_noise:
                print(f"  *** noise_pred MISMATCH at step {step_idx}! ***")
                if first_fail_step is None:
                    first_fail_step = step_idx
                all_passed = False

            # Solver step
            latents = self.runner.solver.step(
                model_output=noise_pred_final,
                timestep=t_scalar,
                sample=latents.transpose(0, 4, 1, 2, 3),  # to (B, C, T, H, W)
                return_dict=False,
            )[0]
            latents = latents.transpose(0, 2, 3, 4, 1)  # back to (B, T, H, W, C)

            # Compare latents_after
            jax_latents_after = np.transpose(np.asarray(latents), (0, 4, 1, 2, 3))
            is_close_after, max_diff_after, _ = compare_arrays(
                f"Step {step_idx} latents_after",
                jax_latents_after,
                torch_latents_after,
                rtol=0.05,
                atol=0.1,
            )

            if not is_close_after:
                print(f"  *** latents_after MISMATCH at step {step_idx}! ***")
                if first_fail_step is None:
                    first_fail_step = step_idx
                all_passed = False

        print("\n" + "=" * 60)
        if all_passed:
            print("All steps PASSED!")
        else:
            print(f"FAILED! First failure at step {first_fail_step}")
        print("=" * 60)

        self.assertTrue(all_passed, f"Step-by-step comparison failed at step {first_fail_step}")

    def test_03_final_output_comparison(self):
        """Test 3: Compare final output."""
        print("\n" + "=" * 60)
        print("TEST 3: Final Output Comparison")
        print("=" * 60)

        # Prepare inputs
        prompt_embeds = np.squeeze(self.prompt_embeds, axis=1)
        negative_prompt_embeds = np.squeeze(self.negative_prompt_embeds, axis=1)
        latents_in_transposed = np.transpose(self.latents_in, (0, 2, 3, 4, 1))
        _, C, T, H, W = self.latents_in.shape

        req = Req(
            prompt_embeds=jnp.array(prompt_embeds),
            negative_prompt_embeds=jnp.array(negative_prompt_embeds),
            latents=jnp.array(latents_in_transposed),
            guidance_scale=3.0,
            width=W * self.runner.model_config.scale_factor_spatial,
            height=H * self.runner.model_config.scale_factor_spatial,
            num_frames=(T - 1) * self.runner.model_config.scale_factor_temporal + 1,
            output_path="/tmp/test_output",
        )

        # Run forward
        out_batch = self.runner.forward(req, self.mesh)
        latents_out = np.transpose(out_batch.latents, (0, 4, 1, 2, 3))

        # Compare
        compare_arrays("Final latents", latents_out, self.latents_out_expected, rtol=0.05, atol=0.1)

        np.testing.assert_allclose(
            latents_out,
            self.latents_out_expected,
            rtol=0.05,
            atol=0.1,
            err_msg="Final latents mismatch",
        )
        print("Final output comparison PASSED!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
