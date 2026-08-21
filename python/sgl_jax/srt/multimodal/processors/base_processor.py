import asyncio
import base64
import concurrent.futures
import io
import logging
import os
from abc import ABC, abstractmethod
from urllib.parse import unquote, urlparse

import numpy as np
import requests
from PIL import Image

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalInputs
from sgl_jax.srt.multimodal.processors.executor import MultimodalProcessorExecutor

logger = logging.getLogger(__name__)

# Safety limits for fetching remote multimodal payloads. These are intentionally
# conservative and should become configurable via ServerArgs.
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
    auto_mm_io_worker_num = 4
    auto_mm_processor_worker_num = 1
    supports_mm_processor_concurrency = False

    def __init__(self, hf_config, server_args, processor):
        self.hf_config = hf_config
        self.server_args = server_args
        self.processor = processor
        self._shutdown = False

        requested_io_workers = getattr(server_args, "mm_io_worker_num", 0)
        env_io_workers = os.environ.get("SGLANG_IO_WORKERS")
        self.mm_io_worker_num = (
            requested_io_workers
            or (int(env_io_workers) if env_io_workers is not None else 0)
            or self.auto_mm_io_worker_num
        )
        if self.mm_io_worker_num <= 0:
            raise ValueError("Multimodal I/O worker count must be positive.")
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.mm_io_worker_num,
            thread_name_prefix="sgl-jax-mm-io",
        )

        self.mm_processor_worker_num = (
            getattr(server_args, "mm_processor_worker_num", 0) or self.auto_mm_processor_worker_num
        )
        if self.mm_processor_worker_num <= 0:
            raise ValueError("Multimodal processor worker count must be positive.")
        if self.mm_processor_worker_num > 1 and not self.supports_mm_processor_concurrency:
            logger.warning(
                "%s does not support concurrent multimodal processing; using one worker.",
                type(self).__name__,
            )
            self.mm_processor_worker_num = 1
        try:
            self.mm_processor_executor = MultimodalProcessorExecutor(
                processor, self.mm_processor_worker_num
            )
        except Exception:
            logger.warning(
                "Unable to clone %s processor; using one worker.",
                type(self).__name__,
                exc_info=True,
            )
            self.mm_processor_worker_num = 1
            self.mm_processor_executor = MultimodalProcessorExecutor(processor, 1)

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

    async def _run_io_async(self, function, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.io_executor, function, *args)

    async def load_image_async(self, source) -> Image.Image:
        return await self._run_io_async(self.load_image, source)

    async def _run_hf_processor_async(
        self,
        input_text: str,
        image_sources: list,
        videos: list | None,
        processor_kwargs: dict,
    ):
        images = await asyncio.gather(*(self.load_image_async(source) for source in image_sources))

        def run_hf_processor(*, processor):
            kwargs = {
                "text": [input_text],
                "images": images or None,
                "padding": True,
                "return_tensors": "pt",
                **processor_kwargs,
            }
            if videos is not None:
                kwargs["videos"] = videos or None
            return processor(**kwargs)

        return await self.mm_processor_executor.run(run_hf_processor)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        self.mm_processor_executor.shutdown()
