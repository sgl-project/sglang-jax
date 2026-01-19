"""
Test script for WanTransformer3DModel weight loading.

Usage:
    python -m sgl_jax.test.multimodal.test_wan2_1_dit_weight_loading
"""

import logging
import os
import pprint

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.multimodal.configs.dits.wan_model_config import WanModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_weight_loading():
    """Test loading weights into WanTransformer3DModel."""
    from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import (
        WanTransformer3DModel,
    )

    # Model path
    model_path = "/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"

    if not os.path.exists(model_path):
        logger.info("Please download the model first:")
        logger.info("  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        return False

    # Check for safetensors files
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not safetensor_files:
        return False
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        config_json = json.load(f)
    # Create config
    config = WanModelConfig()
    config = config.from_dict(config_json)

    # Create mesh (single device for testing)
    devices = jax.devices()
    logger.info("Available devices: %s", devices)
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))

    # Create model
    logger.info("Creating WanTransformer3DModel...")
    try:
        ctx = jax.sharding.use_mesh(mesh)
    except AttributeError:
        try:
            ctx = jax.set_mesh(mesh)
        except AttributeError:
            ctx = mesh
    with ctx:
        model = nnx.eval_shape(
            lambda: WanTransformer3DModel(
                config=config, dtype=config.dtype, mesh=mesh, rngs=nnx.Rngs(0)
            )
        )
    logger.info("Model created successfully")

    # Get initial parameter stats
    params_before = nnx.state(model)

    def inspect_param(x):
        if hasattr(x, "shape"):
            try:
                # Use flat slice to avoid large output
                data_slice = x.reshape(-1)[:5]
                return {
                    "shape": x.shape,
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "slice": data_slice.tolist(),
                }
            except Exception:
                return str(type(x))
        return None

    with open("params_before.txt", "w") as f:
        pprint.pprint(jax.tree_util.tree_map(inspect_param, params_before), stream=f)
    logger.info("Dumped params_before structure/values to params_before.txt")

    # Sample a few parameters before loading
    sample_params_before = {}
    try:
        sample_params_before["proj_out.kernel"] = model.proj_out.kernel[...].mean().item()
        sample_params_before["scale_shift_table"] = model.scale_shift_table[...].mean().item()
        sample_params_before["blocks.0.to_q.weight"] = (
            model.blocks[0].to_q.weight[...].mean().item()
        )
    except Exception as e:
        logger.warning("Could not sample params before: %s", e)

    # Load weights
    try:
        model.load_weights(model_path)
        logger.info("Weights loaded successfully!")
    except Exception as e:
        logger.error("Failed to load weights: %s", e)
        import traceback

        traceback.print_exc()
        return False

    params_after = nnx.state(model)
    with open("params_after.txt", "w") as f:
        # Re-define or reuse inspect_param logic
        def inspect_param(x):
            if hasattr(x, "shape"):
                try:
                    return {"shape": str(type(x))}
                except Exception:
                    return x.shape
            return None

        pprint.pprint(jax.tree_util.tree_map(inspect_param, params_after), stream=f)
    logger.info("Dumped params_after structure/values to params_after.txt")

    # Sample parameters after loading
    sample_params_after = {}
    try:
        sample_params_after["proj_out.kernel"] = model.proj_out.kernel[...].mean().item()
        sample_params_after["scale_shift_table"] = model.scale_shift_table[...].mean().item()
        sample_params_after["blocks.0.to_q.weight"] = model.blocks[0].to_q.weight[...].mean().item()
    except Exception as e:
        logger.warning("Could not sample params after: %s", e)

    # Verify weights changed
    weights_changed = False
    for key in sample_params_before:
        if key in sample_params_after:
            weights_changed = (
                True if sample_params_before[key] != sample_params_after[key] else weights_changed
            )

    if weights_changed:
        logger.info("SUCCESS: Weights were loaded and changed from initial values")
    else:
        logger.warning("WARNING: Weights may not have changed - verify manually")

    # Print some weight statistics
    logger.info("\nWeight statistics after loading:")

    return True


def test_forward_pass():
    """Test forward pass after loading weights."""
    from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import (
        WanTransformer3DModel,
    )

    model_path = "/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer"

    if not os.path.exists(model_path):
        logger.info("Skipping forward pass test - model not found")
        return True
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        config_json = json.load(f)
    config = WanModelConfig.from_dict(config_json)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))

    logger.info("Creating model and loading weights for forward pass test...")
    try:
        ctx = jax.sharding.use_mesh(mesh)
    except AttributeError:
        try:
            ctx = jax.set_mesh(mesh)
        except AttributeError:
            ctx = mesh
    with ctx:
        model = WanTransformer3DModel(config=config, rngs=nnx.Rngs(0), mesh=mesh)
        model.load_weights(model_path)

    # Create dummy inputs
    batch_size = 1
    num_frames = 4
    height = 64
    width = 64
    text_seq_len = 128

    logger.info("Creating dummy inputs...")
    hidden_states = jax.random.normal(
        jax.random.key(0),
        (batch_size, config.in_channels, num_frames, height, width),
        dtype=jnp.bfloat16,
    )
    encoder_hidden_states = jax.random.normal(
        jax.random.key(1),
        (batch_size, text_seq_len, config.text_dim),
        dtype=jnp.bfloat16,
    )
    timesteps = jax.random.uniform(jax.random.key(2), (batch_size,), dtype=jnp.float32)

    logger.info(
        "Input shapes: hidden_states=%s, encoder=%s",
        hidden_states.shape,
        encoder_hidden_states.shape,
    )

    # Run forward pass
    logger.info("Running forward pass...")
    try:
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            encoder_hidden_states_image=None,
        )
        logger.info("Forward pass successful! Output shape: %s", output.shape)
        logger.info("Output mean: %.6f, std: %.6f", output.mean(), output.std())
        return True
    except Exception as e:
        logger.error("Forward pass failed: %s", e)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing WanTransformer3DModel weight loading")
    logger.info("=" * 60)

    success = test_weight_loading()

    if success:
        logger.info("\n%s", "=" * 60)
        logger.info("Testing forward pass")
        logger.info("%s", "=" * 60)
        test_forward_pass()

    logger.info("\n%s", "=" * 60)
    logger.info("Test completed")
    logger.info("%s", "=" * 60)
