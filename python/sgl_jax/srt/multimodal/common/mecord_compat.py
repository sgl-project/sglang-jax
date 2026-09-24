"""Compatibility shim for Moonshot's internal mecord video reader.

The Kimi-K2.5 HuggingFace remote code (media_utils.py /
kimi_k25_vision_processing.py) decodes video exclusively through:

    from mecord import VideoReader

https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/kimi_k25_vision_processing.py
https://huggingface.co/moonshotai/Kimi-K2.5/discussions/25

mecord is Moonshot-internal and is not published on PyPI, so the remote
code falls back to VideoReader = None and every video request dies with
`TypeError: 'NoneType' object is not callable`.

Only the interface is missing, not the capability: sglang-jax repo already depends
on decord and already uses it for the Qwen-VL video path. So this module is a thin
adapter that maps Moonshot's API onto decord rather than a second decoder
implementation.

The adapter exists because decord is not API-compatible with mecord:

Surface required by the Kimi remote code:

    video = VideoReader(src, auto_init=..., num_threads=...,
                        frame_time_info=..., key_indices=...)
    video.num_frames        # int  > 0
    video.original_width    # int  > 0
    video.original_height   # int  > 0
    video.avg_fps           # float > 0
    video.key_indices       # list[int]  (pydantic-validated, must not be None)
    video.frame_time_info   # dict       (pydantic-validated, must not be None)
    frames = video[[i0, i1, ...]]   # -> list[np.ndarray] (H, W, 3) uint8 RGB
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import logging
import os
import sys
import tempfile
import types
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DATA_URI_MARKER = "base64,"


def _write_temp_video(payload: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as stream:
        stream.write(payload)
        return stream.name


@contextlib.contextmanager
def materialize_video(video_src: Any):
    """Keep one local source alive for both metadata probing and frame decoding."""
    path, temporary = _materialize_to_path(video_src)
    try:
        yield path
    finally:
        if temporary:
            os.unlink(path)


def _materialize_to_path(video_src: Any) -> tuple[str, bool]:
    """Resolve an arbitrary video reference to a filesystem path.

    decord decodes from a path, so in-memory sources are spilled to a temp
    file. Returns (path, is_temporary); temporary files are removed when the
    reader is collected.
    """
    if isinstance(video_src, os.PathLike):
        video_src = str(video_src)

    if isinstance(video_src, (bytes, bytearray, memoryview)):
        return _write_temp_video(bytes(video_src)), True

    if hasattr(video_src, "read"):
        data = video_src.read()
        return _write_temp_video(data if isinstance(data, bytes) else bytes(data)), True

    if isinstance(video_src, str):
        if os.path.exists(video_src):
            return video_src, False
        if video_src.startswith("file://"):
            return video_src[len("file://") :], False
        if video_src.startswith(("http://", "https://")):
            from sgl_jax.srt.multimodal.processors.base_processor import (
                fetch_remote_bytes,
            )

            return _write_temp_video(fetch_remote_bytes(video_src)), True
        if video_src.startswith("data:") and _DATA_URI_MARKER in video_src:
            payload = video_src.split(_DATA_URI_MARKER, 1)[1]
            return _write_temp_video(base64.b64decode(payload)), True
        try:
            return _write_temp_video(base64.b64decode(video_src, validate=True)), True
        except Exception as exc:
            raise ValueError(
                "Unsupported video source string: not a path, URL, data URI or base64 payload"
            ) from exc

    raise ValueError(f"Unsupported video source type: {type(video_src)}")


class VideoReader:
    """Minimal mecord.VideoReader work-alike backed by decord."""

    def __init__(
        self,
        video_src: Any,
        auto_init: bool = True,
        num_threads: int = 1,
        frame_time_info: dict | None = None,
        key_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        # auto_init and the hint kwargs below are accelerators in the real
        # mecord. They are accepted and ignored, but non-None values must be
        # echoed back: get_video_meta() feeds them into a pydantic model whose
        # list[int] / dict fields are not Optional.
        self.frame_time_info: dict = dict(frame_time_info or {})
        self.key_indices: list[int] = list(key_indices) if key_indices else [0]

        self._path, self._is_temp = _materialize_to_path(video_src)

        from decord import VideoReader as _DecordVideoReader
        from decord import cpu

        self._reader = _DecordVideoReader(
            self._path, ctx=cpu(0), num_threads=max(1, int(num_threads or 1))
        )

        self.num_frames = int(len(self._reader))
        if self.num_frames <= 0:
            raise ValueError("Video must have at least one frame.")

        self.avg_fps = float(self._reader.get_avg_fps() or 0.0)
        if self.avg_fps <= 0:
            self.avg_fps = 1.0
            logger.warning("Could not determine video fps; defaulting to 1.0")

        # decord exposes no dimension metadata, so take it from a frame.
        first = self._reader[0].asnumpy()
        self.original_height, self.original_width = int(first.shape[0]), int(first.shape[1])

    def __del__(self) -> None:
        if getattr(self, "_is_temp", False):
            with contextlib.suppress(OSError):
                os.unlink(self._path)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int | slice | Iterable[int]) -> Any:
        if isinstance(index, (int, np.integer)):
            return self._read_indices([int(index)])[0]
        if isinstance(index, slice):
            return self._read_indices(list(range(*index.indices(self.num_frames))))
        return self._read_indices([int(i) for i in index])

    def get_batch(self, indices: Iterable[int]) -> np.ndarray:
        """decord-style helper; returns a stacked (T, H, W, 3) array."""
        return np.stack(self._read_indices([int(i) for i in indices]))

    def _read_indices(self, indices: list[int]) -> list[np.ndarray]:
        if not indices:
            return []
        if min(indices) < 0:
            raise IndexError(f"Negative frame index requested: {min(indices)}")

        # Clamp out-of-range requests to the final frame rather than crashing:
        # containers routinely over-report frame count, and Kimi's sampler
        # derives its indices from that count via np.linspace.
        last = self.num_frames - 1
        clamped = []
        for i in indices:
            if i > last:
                logger.warning(
                    "Frame %d unavailable (video ends at %d); reusing last frame.", i, last
                )
                clamped.append(last)
            else:
                clamped.append(i)

        batch = self._reader.get_batch(clamped).asnumpy()
        return [np.asarray(frame) for frame in batch]


def install_mecord_shim() -> bool:
    """Register this module's VideoReader as mecord when needed.

    Returns True if the shim was installed, False if a real mecord is
    already importable (which always takes precedence) or one is already
    registered.

    Moonshot's import is wrapped in try/except and only dereferenced when a
    video is actually decoded, so this does not need to run before the processor
    is constructed -- only before the first video request.
    """
    if "mecord" in sys.modules:
        return False
    if importlib.util.find_spec("mecord") is not None:
        return False

    module = types.ModuleType("mecord")
    module.VideoReader = VideoReader
    module.__doc__ = "sgl-jax compatibility shim for Moonshot's mecord video reader."
    module.__all__ = ["VideoReader"]
    sys.modules["mecord"] = module
    logger.info(
        "Installed sgl-jax 'mecord' compatibility shim (real mecord not available); "
        "Kimi video decoding will use decord."
    )
    return True
