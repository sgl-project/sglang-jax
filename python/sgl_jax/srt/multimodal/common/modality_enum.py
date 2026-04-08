import dataclasses
import hashlib
import pickle
from enum import Enum, auto
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [item for sublist in nested_list for item in flatten_nested_list(sublist)]
    else:
        return [nested_list]


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


class Modality(Enum):
    IMAGE = auto()
    MULTI_IMAGES = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError as err:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            ) from err

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem contains all inputs of one modality.
    For example, if there are 3 images and 1 audio input, there will be 2 MultimodalDataItems: one for images, one for audio.
    Common fields are placed at the front, model-specific fields are in model_specific_data.
    """

    modality: Modality
    hash: int | None = None
    pad_value: int | None = None
    offsets: list | None = None

    # Raw features returned by processor, e.g. pixel_values or audio_features
    feature: jax.Array | np.ndarray | None = None
    # Precomputed embeddings passed as final encoder embeddings
    # Only one of feature and precomputed_embeddings is non-empty
    precomputed_embeddings: jax.Array | np.ndarray | None = None

    # Model-specific data stored in dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if "model_specific_data" in self.__dict__ and name in self.__dict__["model_specific_data"]:
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(lst):
        if lst is None:
            return True
        return len([item for item in flatten_nested_list(lst) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set padding value after hashing the data first
        """
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        # Use a smaller modulo to keep pad_value in a reasonable range
        # The pad_value is used for radix cache differentiation, not for embedding lookup
        # We use a 24-bit range which gives ~16M unique values, sufficient for cache keys
        self.pad_value = self.hash % (1 << 24)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        # TODO: Implement validation logic
        pass

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def merge(self, other):
        # Merge features (handle JAX arrays and NumPy arrays)
        if self.feature is not None and other.feature is not None:
            if isinstance(self.feature, jax.Array) and isinstance(other.feature, jax.Array):
                self.feature = jnp.concatenate([self.feature, other.feature], axis=0)
            elif isinstance(self.feature, np.ndarray) and isinstance(other.feature, np.ndarray):
                self.feature = np.concatenate([self.feature, other.feature], axis=0)
            else:
                # Convert to JAX arrays for mixed types
                self.feature = jnp.concatenate(
                    [jax.device_put(self.feature), jax.device_put(other.feature)], axis=0
                )

        # Merge offsets
        if self.offsets is not None and other.offsets is not None:
            self.offsets += other.offsets

        # Update hash
        self.hash = hash((self.hash, other.hash))
        self.set_pad_value()


@dataclasses.dataclass
class MultimodalInputs:
    """Inputs related to multimodal data"""

    # List of data items
    mm_items: list[MultimodalDataItem]
    image_pad_len: list | None = None
    num_image_tokens: int | None = None

    # Image-related
    im_token_id: int | None = None
    im_start_id: int | None = None
    im_end_id: int | None = None
    slice_start_id: int | None = None
    slice_end_id: int | None = None

    # Video-related
    video_token_id: int | None = None

    # Audio-related
    audio_token_id: int | None = None
    audio_start_id: int | None = None
    audio_end_id: int | None = None

    # QWen2-VL related
    mrope_positions: jax.Array | None = None
    mrope_position_delta: jax.Array | None = None

    @staticmethod
    def from_dict(obj: dict):
        mm_items = []
        for item_data in obj.get("mm_items", []):
            if isinstance(item_data, dict):
                mm_items.append(MultimodalDataItem.from_dict(item_data))
            elif isinstance(item_data, MultimodalDataItem):
                mm_items.append(item_data)

        ret = MultimodalInputs(
            mm_items=mm_items,
        )

        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]
        for item in ret.mm_items:
            item.set_pad_value()

        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "image_pad_len",
            "num_image_tokens",
        ]
        for arg in optional_args:
            if arg in obj:
                value = obj[arg]
                if isinstance(value, (np.ndarray, jax.Array)):
                    setattr(ret, arg, jax.device_put(value))
                else:
                    setattr(ret, arg, value)

        return ret

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def merge(self, other):
        """
        Merge multimodal inputs when merging requests
        """
        # Parameters to merge
        if self.image_pad_len is not None and other.image_pad_len is not None:
            self.image_pad_len += other.image_pad_len

        # Merge mm_items
        self.mm_items += other.mm_items

        # Merge mrope_positions (JAX array handling)
        if self.mrope_positions is not None:
            if other.mrope_positions is not None:
                self.mrope_positions = jnp.concatenate(
                    [self.mrope_positions, other.mrope_positions], axis=1
                )
        else:
            self.mrope_positions = other.mrope_positions

        # Merge mrope_position_delta (JAX array handling)
        if self.mrope_position_delta is not None:
            if other.mrope_position_delta is not None:
                self.mrope_position_delta = jnp.concatenate(
                    [self.mrope_position_delta, other.mrope_position_delta], axis=0
                )
        else:
            self.mrope_position_delta = other.mrope_position_delta

        # Merge token id related parameters (keep non-None values)
        for key in dir(self):
            if "_id" in key and not key.startswith("__"):
                self_val = getattr(self, key, None)
                other_val = getattr(other, key, None)
                if self_val is None and other_val is not None:
                    setattr(self, key, other_val)

        # Merge other numeric parameters
        if self.num_image_tokens is not None and other.num_image_tokens is not None:
            self.num_image_tokens += other.num_image_tokens
        elif self.num_image_tokens is None:
            self.num_image_tokens = other.num_image_tokens
