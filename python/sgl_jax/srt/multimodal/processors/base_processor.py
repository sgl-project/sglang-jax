import base64
import io
import os
from abc import ABC, abstractmethod
from urllib.parse import unquote, urlparse

import numpy as np
import requests
from PIL import Image

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalInputs

# Stage 1 safety limits for fetching remote multimodal payloads. These are
# intentionally conservative; future stages should make them configurable via
# ServerArgs and add proper async fetching.
DEFAULT_HTTP_TIMEOUT_SECS = 30
MAX_REMOTE_BYTES = 64 * 1024 * 1024  # 64 MiB hard cap per asset


def _fetch_url(url: str) -> bytes:
    with requests.get(url, timeout=DEFAULT_HTTP_TIMEOUT_SECS, stream=True) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_REMOTE_BYTES:
            raise ValueError(
                f"Remote asset at {url} reports {content_length} bytes, "
                f"exceeds limit of {MAX_REMOTE_BYTES} bytes."
            )
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=1 << 20):
            buffer.extend(chunk)
            if len(buffer) > MAX_REMOTE_BYTES:
                raise ValueError(
                    f"Remote asset at {url} exceeds limit of {MAX_REMOTE_BYTES} bytes."
                )
        return bytes(buffer)


def _normalize_image_source(source) -> bytes | str:
    """Normalize an image source into raw bytes or a local file path.

    Accepts: bytes, http(s) URL, file:// URI, data: URI, local file path,
    or a bare base64 string.
    """
    if isinstance(source, bytes):
        return source
    if not isinstance(source, str):
        raise ValueError(f"Unsupported image source: {type(source)}")
    if source.startswith(("http://", "https://")):
        return _fetch_url(source)
    if source.startswith("file://"):
        return unquote(urlparse(source).path)
    if source.startswith("data:"):
        return base64.b64decode(source.split(",", 1)[1], validate=True)
    if os.path.isfile(source):
        return source
    return base64.b64decode(source, validate=True)


class BaseMultimodalProcessor(ABC):
    models: tuple[str, ...] = ()

    def __init__(self, hf_config, server_args, processor):
        self.hf_config = hf_config
        self.server_args = server_args
        self.processor = processor

    def apply_chat_template(self, *args, **kwargs):
        return self.processor.apply_chat_template(*args, **kwargs)

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> MultimodalInputs:
        """Process multimodal payload and return a ``MultimodalInputs``."""
        pass

    @staticmethod
    def normalize_data(data) -> list:
        if data is None:
            return []
        return data if isinstance(data, list) else [data]

    @staticmethod
    def unwrap_source(source):
        if isinstance(source, dict) and "url" in source:
            return source["url"]
        if hasattr(source, "url"):
            return source.url
        return source

    @classmethod
    def load_image(cls, source) -> Image.Image:
        source = cls.unwrap_source(source)
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")

        payload = _normalize_image_source(source)
        if isinstance(payload, bytes):
            return Image.open(io.BytesIO(payload)).convert("RGB")
        return Image.open(payload).convert("RGB")
