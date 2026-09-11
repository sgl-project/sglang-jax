from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import numpy as np
import pybase64
import requests
from PIL import Image

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalInputs
from sgl_jax.srt.multimodal.processors.executor import MultimodalProcessorExecutor

if TYPE_CHECKING:
    from transformers.image_utils import ImageInput

logger = logging.getLogger(__name__)

# Safety limits for fetching remote multimodal payloads. These are intentionally
# conservative and should become configurable via ServerArgs.
DEFAULT_HTTP_TIMEOUT_SECS = 30
MAX_REMOTE_BYTES = 64 * 1024 * 1024  # 64 MiB hard cap per asset


def fetch_remote_bytes(url: str) -> bytes:
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
        return fetch_remote_bytes(source)
    if source.startswith("file://"):
        return unquote(urlparse(source).path)
    if source.startswith("data:"):
        return pybase64.b64decode(source.split(",", 1)[1], validate=True)
    if os.path.isfile(source):
        return source
    return pybase64.b64decode(source, validate=True)


class BaseMultimodalProcessor(ABC):
    models: tuple[str, ...] = ()
    auto_mm_processor_worker_num = 1
    supports_mm_processor_concurrency = False
    use_torchcodec_image_decode = False

    def __init__(self, hf_config, server_args, processor):
        self.hf_config = hf_config
        self.server_args = server_args
        self.processor = processor
        self._shutdown = False

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
    def load_image(cls, source) -> ImageInput:
        source = cls.unwrap_source(source)
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")

        payload = _normalize_image_source(source)
        if cls.use_torchcodec_image_decode:
            from torchcodec.decoders import decode_image

            try:
                image = decode_image(payload, mode="RGB")
                if image.ndim == 3:
                    return image
            except (RuntimeError, ValueError):
                logger.debug("Falling back to Pillow image decode", exc_info=True)
        if isinstance(payload, bytes):
            return Image.open(io.BytesIO(payload)).convert("RGB")
        return Image.open(payload).convert("RGB")

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            # NumPy has no portable bfloat16 representation. Multimodal
            # features are host-side inputs, so use float32 for interchange.
            if str(getattr(value, "dtype", "")) == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _to_grid_list(cls, value) -> list[tuple[int, int, int]]:
        if value is None:
            return []
        return [tuple(map(int, row)) for row in cls._to_numpy(value).reshape(-1, 3)]

    def process_mm_data(
        self,
        input_text: str,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        *,
        processor,
        **kwargs,
    ):
        """Run the Hugging Face processor synchronously.

        Call this from a multimodal processor worker, after loading its inputs.
        """
        processor_inputs = {
            "text": [input_text],
            "images": images or None,
            "padding": True,
            # Preserve float32 video timing metadata from the HF processor.
            "return_tensors": "pt" if videos else None,
            **kwargs,
        }
        if videos is not None:
            processor_inputs["videos"] = videos or None
        if audios is not None:
            processor_inputs["audios"] = audios or None
        return processor(**processor_inputs)

    def collect_mm_items_from_processor_output(
        self,
        processor_output,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        **kwargs,
    ) -> MultimodalInputs:
        """Convert one HF processor output into the runtime MM contract.

        Model adapters override this hook when their feature layout or token
        metadata is model-specific. The default handles text-only output.
        """
        del images, videos, audios, kwargs
        input_ids = self._to_numpy(processor_output.get("input_ids"))
        if input_ids is None:
            raise ValueError("HF multimodal processor did not return input_ids.")
        return MultimodalInputs(mm_items=[], input_ids=input_ids.reshape(-1).tolist())

    def process_and_combine_mm_data(
        self,
        input_text: str,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        *,
        processor,
        **processor_kwargs,
    ) -> MultimodalInputs:
        processor_output = self.process_mm_data(
            input_text,
            images=images,
            videos=videos,
            audios=audios,
            processor=processor,
            **processor_kwargs,
        )
        return self.collect_mm_items_from_processor_output(
            processor_output,
            images=images,
            videos=videos,
            audios=audios,
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.mm_processor_executor.shutdown()
