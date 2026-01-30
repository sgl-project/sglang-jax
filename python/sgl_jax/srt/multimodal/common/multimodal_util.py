import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Literal
from urllib.parse import urlparse

import numpy as np
import pybase64
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def decode_video_base64(video_base64):
    from PIL import Image

    # Decode the base64 string
    video_bytes = pybase64.b64decode(video_base64, validate=True)

    # Placeholder for the start indices of each PNG image
    img_starts = []

    frame_format = "PNG"  # str(os.getenv('FRAME_FORMAT', "JPEG"))

    assert frame_format in [
        "PNG",
        "JPEG",
    ], "FRAME_FORMAT must be either 'PNG' or 'JPEG'"

    if frame_format == "PNG":
        # Find each PNG start signature to isolate images
        i = 0
        while i < len(video_bytes) - 7:  # Adjusted for the length of the PNG signature
            # Check if we found the start of a PNG file
            if (
                video_bytes[i] == 0x89
                and video_bytes[i + 1] == 0x50
                and video_bytes[i + 2] == 0x4E
                and video_bytes[i + 3] == 0x47
                and video_bytes[i + 4] == 0x0D
                and video_bytes[i + 5] == 0x0A
                and video_bytes[i + 6] == 0x1A
                and video_bytes[i + 7] == 0x0A
            ):
                img_starts.append(i)
                i += 8  # Skip the PNG signature
            else:
                i += 1
    else:
        # Find each JPEG start (0xFFD8) to isolate images
        i = 0
        while i < len(video_bytes) - 1:  # Adjusted for the length of the JPEG SOI signature
            # Check if we found the start of a JPEG file
            if video_bytes[i] == 0xFF and video_bytes[i + 1] == 0xD8:
                img_starts.append(i)
                # Move to the next byte to continue searching for the next image start
                i += 2
            else:
                i += 1

    frames = []
    for start_idx in img_starts:
        # Assuming each image is back-to-back, the end of one image is the start of another
        # The last image goes until the end of the byte string
        end_idx = (
            img_starts[img_starts.index(start_idx) + 1]
            if img_starts.index(start_idx) + 1 < len(img_starts)
            else len(video_bytes)
        )
        img_bytes = video_bytes[start_idx:end_idx]

        # Convert bytes to a PIL Image
        img = Image.open(BytesIO(img_bytes))

        # Convert PIL Image to a NumPy array
        frame = np.array(img)

        # Append the frame to the list of frames
        frames.append(frame)

    # Ensure there's at least one frame to avoid errors with np.stack
    if frames:
        return np.stack(frames, axis=0), img.size
    else:
        return np.array([]), (
            0,
            0,
        )  # Return an empty array and size tuple if no frames were found


def load_audio(audio_file: str, sr: int | None = None, mono: bool = True) -> np.ndarray:
    # Use soundfile here, since librosa use it under the hood,
    # and librosa will not support audio loading in the future
    import soundfile as sf
    from scipy.signal import resample

    if sr is None:
        sr = 16000

    # Load audio data
    if isinstance(audio_file, bytes):
        audio, original_sr = sf.read(BytesIO(audio_file))
    elif audio_file.startswith("data:"):
        audio_file = audio_file.split(",")[1]
        audio, original_sr = sf.read(BytesIO(pybase64.b64decode(audio_file, validate=True)))
    elif audio_file.startswith("http://") or audio_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
        response = requests.get(audio_file, stream=True, timeout=timeout)
        audio_file = BytesIO(response.content)
        response.close()
        audio, original_sr = sf.read(audio_file)
    elif isinstance(audio_file, str):
        audio, original_sr = sf.read(audio_file)
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    # Resample audio if the original sample rate is different from the desired sample rate
    if original_sr != sr:
        num_samples = int(len(audio) * float(sr) / original_sr)
        audio = resample(audio, num_samples)

    # Convert to mono if requested and audio is stereo
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio


@dataclass
class ImageData:
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"


def load_image(
    image_file: Image.Image | str | ImageData | bytes,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = image_size = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, stream=True, timeout=timeout)
        try:
            response.raise_for_status()
            image = Image.open(response.raw)
            image.load()  # Force loading to avoid issues after closing the stream
        finally:
            response.close()
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image_file}")

    return image, image_size


def get_image_bytes(image_file: str | bytes):
    if isinstance(image_file, bytes):
        return image_file
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        return response.content
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        with open(image_file, "rb") as f:
            return f.read()
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        return pybase64.b64decode(image_file, validate=True)
    elif isinstance(image_file, str):
        return pybase64.b64decode(image_file, validate=True)
    else:
        raise NotImplementedError(f"Invalid image: {image_file}")


def load_video(video_file: str | bytes, use_gpu: bool = True):
    # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
    from decord import VideoReader, cpu, gpu

    try:
        from decord.bridge import decord_bridge

        ctx = gpu(0)
        _ = decord_bridge.get_ctx_device(ctx)
    except Exception:
        ctx = cpu(0)

    tmp_file = None
    vr = None
    try:
        if isinstance(video_file, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file)
                tmp_file_name = tmp_file.name
            vr = VideoReader(tmp_file_name, ctx=ctx)
        elif isinstance(video_file, str):
            if video_file.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                response = requests.get(video_file, stream=True, timeout=timeout)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    tmp_file_name = tmp_file.name
                vr = VideoReader(tmp_file_name, ctx=ctx)
            elif video_file.startswith("data:"):
                _, encoded = video_file.split(",", 1)
                video_bytes = pybase64.b64decode(encoded, validate=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_bytes)
                    tmp_file_name = tmp_file.name
                vr = VideoReader(tmp_file_name, ctx=ctx)
            # `urlparse` supports file:// paths, and so does VideoReader
            elif os.path.isfile(urlparse(video_file).path):
                vr = VideoReader(video_file, ctx=ctx)
            else:
                video_bytes = pybase64.b64decode(video_file, validate=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_bytes)
                    tmp_file_name = tmp_file.name
                vr = VideoReader(tmp_file_name, ctx=ctx)
        else:
            raise ValueError(f"Unsupported video input type: {type(video_file)}")

        return vr

    finally:
        if tmp_file and hasattr(tmp_file, "name") and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


def encode_video(video_path, frame_count_limit=None):
    # Lazy import because decord is not available on some arm platforms.
    from decord import VideoReader, cpu

    if not os.path.exists(video_path):
        logger.error("Video %s does not exist", video_path)
        return []

    if frame_count_limit == 0:
        return []

    def uniform_sample(lst, n):
        gap = len(lst) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [lst[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_indices = [i for i in range(0, len(vr), sample_fps)]
    if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
        frame_indices = uniform_sample(frame_indices, frame_count_limit)

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames
