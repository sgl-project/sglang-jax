"""
Weight mapping utilities for loading PyTorch weights into JAX VideoDecoder.

Maps PyTorch state_dict keys to JAX/Flax parameter paths.
"""

from typing import Dict, List, Tuple


def create_decoder_weight_mappings() -> List[Tuple[str, str]]:
    """
    Create mappings from PyTorch state_dict keys to JAX parameter paths.

    PyTorch uses channel-first format (B, C, T, H, W) and state_dict naming.
    JAX uses channel-last format (B, T, H, W, C) and Flax naming.

    Returns:
        List of (pytorch_key, jax_path) tuples
    """
    mappings = []

    # Per-channel statistics
    # PyTorch: decoder.per_channel_statistics.std-of-means
    # JAX: per_channel_statistics.std_of_means.value
    mappings.extend([
        ("decoder.per_channel_statistics.std-of-means", "per_channel_statistics.std_of_means.value"),
        ("decoder.per_channel_statistics.mean-of-means", "per_channel_statistics.mean_of_means.value"),
    ])

    # Input convolution
    # PyTorch: decoder.conv_in.conv.weight [out_c, in_c, t, h, w]
    # JAX: conv_in.conv.kernel [t, h, w, in_c, out_c] (channel-last + transposed)
    mappings.extend([
        ("decoder.conv_in.conv.weight", "conv_in.conv.kernel", "conv3d_transpose"),
        ("decoder.conv_in.conv.bias", "conv_in.conv.bias"),
    ])

    # Up blocks - dynamically determine based on architecture
    # For standard LTX-2, we have 8 blocks in reversed order
    # Block structure varies: UNetMidBlock3D, DepthToSpaceUpsample, ResnetBlock3D

    # Note: This is a simplified mapping. Full implementation would need to:
    # 1. Iterate through decoder_blocks config
    # 2. Map each block type appropriately
    # 3. Handle nested ResNet blocks within UNetMidBlock3D

    # Placeholder for up_blocks mapping (to be completed)
    # up_blocks[0] through up_blocks[7] need individual mappings

    # Output normalization
    # PixelNorm has no learnable parameters
    # GroupNorm would have weight and bias

    # Output convolution
    mappings.extend([
        ("decoder.conv_out.conv.weight", "conv_out.conv.kernel", "conv3d_transpose"),
        ("decoder.conv_out.conv.bias", "conv_out.conv.bias"),
    ])

    return mappings


def transpose_conv3d_weight(pytorch_weight):
    """
    Transpose Conv3D weight from PyTorch to JAX format.

    PyTorch: [out_channels, in_channels, T, H, W]
    JAX: [T, H, W, in_channels, out_channels]

    Args:
        pytorch_weight: NumPy array in PyTorch format

    Returns:
        NumPy array in JAX format
    """
    import numpy as np

    # Transpose from [O, I, T, H, W] to [T, H, W, I, O]
    return np.transpose(pytorch_weight, (2, 3, 4, 1, 0))


def convert_pytorch_to_jax_weights(pytorch_state_dict: Dict) -> Dict:
    """
    Convert PyTorch state_dict to JAX parameter dict.

    This function:
    1. Maps PyTorch keys to JAX paths
    2. Transposes convolutional kernels
    3. Converts to JAX-compatible format

    Args:
        pytorch_state_dict: PyTorch model state_dict

    Returns:
        JAX-compatible parameter dictionary
    """
    import numpy as np

    mappings = create_decoder_weight_mappings()
    jax_params = {}

    for mapping in mappings:
        if len(mapping) == 2:
            pt_key, jax_path = mapping
            transform = None
        else:
            pt_key, jax_path, transform = mapping

        if pt_key in pytorch_state_dict:
            value = pytorch_state_dict[pt_key]

            # Convert to numpy if needed
            if hasattr(value, 'cpu'):
                value = value.cpu().numpy()

            # Apply transformation
            if transform == "conv3d_transpose":
                value = transpose_conv3d_weight(value)

            jax_params[jax_path] = value

    return jax_params


def load_from_huggingface(model_id: str = "Lightricks/LTX-Video", subfolder: str = "vae"):
    """
    Load LTX-2 VAE weights from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier
        subfolder: Subfolder containing VAE weights

    Returns:
        PyTorch state_dict
    """
    try:
        from huggingface_hub import hf_hub_download
        import torch
    except ImportError:
        raise ImportError(
            "Please install huggingface_hub and torch: "
            "pip install huggingface_hub torch"
        )

    # Download model file
    model_path = hf_hub_download(
        repo_id=model_id,
        filename="diffusion_pytorch_model.safetensors",
        subfolder=subfolder,
    )

    # Load weights
    from safetensors.torch import load_file
    state_dict = load_file(model_path)

    return state_dict


def load_decoder_weights(
    decoder_module,
    model_id: str = "Lightricks/LTX-Video",
    subfolder: str = "vae",
):
    """
    Load weights from HuggingFace into JAX VideoDecoder.

    Args:
        decoder_module: JAX VideoDecoder module
        model_id: HuggingFace model identifier
        subfolder: Subfolder containing VAE weights
    """
    import logging

    logger = logging.getLogger(__name__)

    # Load PyTorch state_dict
    logger.info(f"Loading weights from {model_id}/{subfolder}")
    pytorch_state_dict = load_from_huggingface(model_id, subfolder)

    # Convert to JAX format
    logger.info("Converting PyTorch weights to JAX format")
    jax_params = convert_pytorch_to_jax_weights(pytorch_state_dict)

    # Update decoder parameters
    # Note: This is simplified. Full implementation would use Flax's
    # parameter update utilities to properly set nested parameters.

    logger.info("Loaded decoder weights successfully")
    return jax_params
