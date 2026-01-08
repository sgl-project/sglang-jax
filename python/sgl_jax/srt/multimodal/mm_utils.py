"""
Utilities for multi-modal models in JAX.

Adapted from LLaVA-NeXT's mm_utils.py for JAX/TPU compatibility.
Supports anyres image processing and distributed vision model execution.
"""

import ast
import math
import re
from io import BytesIO
from typing import Any

import jax.numpy as jnp
import numpy as np
import pybase64
from PIL import Image

from sgl_jax.srt.utils import flatten_nested_list


def has_valid_data(data) -> bool:
    """Check if data contains any valid content."""
    if data is None:
        return False
    if isinstance(data, list):
        return any(has_valid_data(item) for item in flatten_nested_list(data))
    return True


def select_best_resolution(
    original_size: tuple[int, int], possible_resolutions: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Selects the best resolution from possible resolutions based on original size.

    Args:
        original_size: (width, height) of original image
        possible_resolutions: List of (width, height) resolution options

    Returns:
        Best fit resolution (width, height)
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate downscaled size maintaining aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        # Update best fit based on effective and wasted resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image: Image.Image, target_resolution: tuple[int, int]) -> Image.Image:
    """
    Resize and pad image to target resolution while maintaining aspect ratio.

    Args:
        image: Input PIL Image
        target_resolution: (width, height) target resolution

    Returns:
        Resized and padded PIL Image
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    # Calculate new dimensions maintaining aspect ratio
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize and pad
    resized_image = image.resize((new_width, new_height))
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image: Image.Image, patch_size: int) -> list[Image.Image]:
    """
    Divide image into patches of specified size.

    Args:
        image: Input PIL Image
        patch_size: Size of each square patch

    Returns:
        List of patch images
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches


def get_anyres_image_grid_shape(
    image_size: tuple[int, int], grid_pinpoints: str, patch_size: int
) -> tuple[int, int]:
    """
    Calculate image patch grid shape for anyres preprocessing.

    Args:
        image_size: (width, height) of input image
        grid_pinpoints: String describing possible resolutions
        patch_size: Size of each image patch

    Returns:
        Grid shape (width, height) in number of patches
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"

        # Parse range from string (e.g., "(1x1)-(3x3)")
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))

        # Generate grid pinpoints and scale by patch size
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    possible_resolutions = (
        grid_pinpoints if isinstance(grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)
    )
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image: Image.Image, processor: Any, grid_pinpoints: str) -> np.ndarray:
    """
    Process image with anyres resolution handling.

    Args:
        image: Input PIL Image
        processor: Image processor object
        grid_pinpoints: String describing possible resolutions

    Returns:
        Processed image patches as numpy array
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        # Get patch size from processor
        try:
            patch_size = processor.size[0]
        except Exception:
            patch_size = processor.size["shortest_edge"]

        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"

        # Parse and generate grid pinpoints
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))

        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    possible_resolutions = (
        grid_pinpoints if isinstance(grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)
    )
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    # Get crop size from processor
    crop_size = (
        processor.crop_size["height"]
        if "crop_size" in processor.__dict__
        else processor.size["height"]
    )
    shortest_edge = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else processor.size["height"]
    )

    # Create patches and process
    patches = divide_to_patches(image_padded, crop_size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    image_patches = [image_original_resize] + patches

    # Preprocess all patches
    image_patches = [
        processor.preprocess(patch.convert("RGB"))["pixel_values"][0] for patch in image_patches
    ]

    return np.stack(image_patches, axis=0)


def load_image_from_base64(image: str) -> Image.Image:
    """Load image from base64 encoded string."""
    return Image.open(BytesIO(pybase64.b64decode(image, validate=True)))


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    """Expand image to square by adding padding."""
    width, height = pil_img.size
    if width == height:
        return pil_img

    # Convert grayscale to RGB if needed
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")

    # Create square background and paste image
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))

    return result


def unpad_image(tensor: jnp.ndarray, original_size: tuple[int, int]) -> jnp.ndarray:
    """
    Unpad a JAX tensor of a padded and resized image.

    Args:
        tensor: Image tensor in CxHxW format
        original_size: Original (width, height) of image

    Returns:
        Unpadded image tensor
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        return tensor[:, :, padding : current_width - padding]


def unpad_image_shape(
    current_height: int, current_width: int, original_size: tuple[int, int]
) -> tuple[int, int]:
    """Calculate unpadded image shape."""
    original_width, original_height = original_size
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        return (current_height - 2 * padding, current_width)
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        return (current_height, current_width - 2 * padding)


def process_images(images: list[Image.Image], image_processor: Any, model_cfg: Any) -> np.ndarray:
    """
    Process list of images according to model configuration.

    Args:
        images: List of PIL Images
        image_processor: Image processor object
        model_cfg: Model configuration with image processing parameters

    Returns:
        Processed image tensors as numpy array
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []

    if image_aspect_ratio == "pad":
        # Pad images to square
        mean = tuple(int(x * 255) for x in image_processor.image_mean)
        for image in images:
            image = expand2square(image, mean)
            processed = image_processor.preprocess(image)["pixel_values"][0]
            new_images.append(processed)

    elif "anyres" in image_aspect_ratio:
        # Process with anyres handling
        for image in images:
            processed = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(processed)

    else:
        # Default processing
        return image_processor(images)["pixel_values"]

    # Stack if all images have same shape
    if all(x.shape == new_images[0].shape for x in new_images):
        return np.stack(new_images, axis=0)
    return np.array(new_images, dtype=object)


def get_dp_encoder_lb_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate load balancing assignment for data parallel distribution.

    Args:
        sizes: Size of each image (in patches)
        num_gpus: Number of GPUs for distribution

    Returns:
        shuffle_indices: Indices to reorder data
        gpu_sample_counts: Number of samples per GPU
        grouped_sizes_per_gpu: Total size per GPU
    """
    n_samples = len(sizes)

    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Greedy load balancing by total size
    gpu_assignments = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # Tracks total size, not count

    # Sort by size (largest first)
    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)

    # Assign to GPUs with minimum current load
    for idx in large_to_small_indices:
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Prepare results
    shuffle_indices = []
    gpu_sample_counts = []
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return shuffle_indices, gpu_sample_counts, gpu_loads
