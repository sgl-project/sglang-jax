import hashlib
import pickle
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.utils import flatten_nested_list

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import MultimodalDataItem


# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def hash_feature(f: Any) -> int:
    """Hash multimodal features"""
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], (jnp.ndarray, np.ndarray)):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        return data_hash(arr.tobytes())
    elif isinstance(f, jnp.ndarray):
        return tensor_hash([f])
    return data_hash(pickle.dumps(f))


def data_hash(data: Any) -> int:
    """Hash raw data bytes"""
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list: Any) -> int:
    """Hash JAX tensors or tensor lists using CPU-based hashing"""
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.reshape(-1) for x in tensor_list if isinstance(x, (jnp.ndarray, np.ndarray))
        ]
        if not tensor_list:
            return 0
        # Convert to numpy arrays for consistent CPU-based hashing
        numpy_arrays = []
        for x in tensor_list:
            if isinstance(x, jnp.ndarray):
                numpy_arrays.append(np.asarray(jax.device_get(x)))
            else:
                numpy_arrays.append(np.asarray(x))
        tensor = np.concatenate(numpy_arrays)
    else:
        tensor = tensor_list

    # Handle JAX arrays - convert to numpy
    if isinstance(tensor, jnp.ndarray):
        tensor = np.asarray(jax.device_get(tensor))

    # Handle numpy arrays with CPU-based hashing
    if isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor.astype(np.float32))
        return data_hash(arr.tobytes())

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def pad_input_tokens(
    input_ids: list[int],
    mm_items: list["MultimodalDataItem"],
    im_token_id: int = None,
    video_token_id: int = None,
    audio_token_id: int = None,
) -> list[int]:
    """
    Replace multimodal placeholder tokens in input_ids with corresponding pad_values from mm_items.

    This is critical for radix cache to differentiate between different images/videos.
    Different images/videos will have different pad_values (hash-based), so the cache
    will correctly identify them as different prefixes.

    Args:
        input_ids: The input token IDs containing placeholder tokens
        mm_items: List of multimodal data items with pad_value set
        im_token_id: Token ID used for image placeholders
        video_token_id: Token ID used for video placeholders
        audio_token_id: Token ID used for audio placeholders

    Returns:
        Modified input_ids with placeholder tokens replaced by pad_values
    """
    if not input_ids or not mm_items:
        return input_ids

    # Build mapping from token_id to list of pad_values for each modality
    # We need to handle multiple items of the same modality
    image_pad_values = []
    video_pad_values = []
    audio_pad_values = []

    for item in mm_items:
        if item.pad_value is None:
            item.set_pad_value()

        if item.is_image() and im_token_id is not None:
            image_pad_values.append(item.pad_value)
        elif item.is_video() and video_token_id is not None:
            video_pad_values.append(item.pad_value)
        elif item.is_audio() and audio_token_id is not None:
            audio_pad_values.append(item.pad_value)

    # Create a mutable copy of input_ids
    padded_ids = list(input_ids)

    # Replace image tokens
    if im_token_id is not None and image_pad_values:
        image_idx = 0
        for i, token_id in enumerate(padded_ids):
            if token_id == im_token_id:
                # Use the pad_value for current image, cycling through if needed
                pad_value = image_pad_values[min(image_idx, len(image_pad_values) - 1)]
                padded_ids[i] = pad_value
                # Don't increment image_idx for each token, only when we hit a boundary
                # Actually, for simple replacement, use same pad_value for all tokens of an image

    # Replace video tokens
    if video_token_id is not None and video_pad_values:
        video_idx = 0
        for i, token_id in enumerate(padded_ids):
            if token_id == video_token_id:
                # Use the pad_value for current video
                pad_value = video_pad_values[min(video_idx, len(video_pad_values) - 1)]
                padded_ids[i] = pad_value

    # Replace audio tokens
    if audio_token_id is not None and audio_pad_values:
        audio_idx = 0
        for i, token_id in enumerate(padded_ids):
            if token_id == audio_token_id:
                pad_value = audio_pad_values[min(audio_idx, len(audio_pad_values) - 1)]
                padded_ids[i] = pad_value

    return padded_ids
