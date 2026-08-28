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

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.executor import MultimodalProcessorExecutor

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

    # EPD input reconstruction adapted from SGLang:
    # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/processors/base_processor.py
    @property
    def spatial_merge_size(self) -> int:
        return self.hf_config.vision_config.spatial_merge_size

    def build_input_ids(
        self,
        prompt,
        img_grid_thw=None,
        video_grid_thw=None,
        audio_seq_lens=None,
    ):
        """Expand one placeholder per multimodal item into its encoded length."""
        if not isinstance(prompt, list):
            prompt = self.processor.tokenizer(prompt)["input_ids"]

        grids = {
            Modality.IMAGE: self._to_grid_list(img_grid_thw),
            Modality.VIDEO: self._to_grid_list(video_grid_thw),
        }
        audio_lengths = (
            []
            if audio_seq_lens is None
            else self._to_numpy(audio_seq_lens).reshape(-1).astype(int).tolist()
        )
        token_ids = {
            Modality.IMAGE: getattr(self.hf_config, "image_token_id", None),
            Modality.VIDEO: getattr(self.hf_config, "video_token_id", None),
            Modality.AUDIO: getattr(self.hf_config, "audio_token_id", None),
        }
        modality_by_token = {
            token_id: modality for modality, token_id in token_ids.items() if token_id is not None
        }
        item_sizes = {
            modality: [int(np.prod(grid) // (self.spatial_merge_size**2)) for grid in values]
            for modality, values in grids.items()
        }
        item_sizes[Modality.AUDIO] = audio_lengths
        consumed = {modality: 0 for modality in item_sizes}
        input_ids, ranges, modalities = [], [], []

        for token_id in prompt:
            modality = modality_by_token.get(token_id)
            if modality is None:
                input_ids.append(token_id)
                continue

            item_index = consumed[modality]
            if item_index >= len(item_sizes[modality]):
                raise ValueError(f"missing {modality.name} encoder metadata")
            token_count = item_sizes[modality][item_index]
            start = len(input_ids)
            input_ids.extend([token_id] * token_count)
            ranges.append((start, len(input_ids)))
            modalities.append(modality)
            consumed[modality] += 1

        for modality, sizes in item_sizes.items():
            if consumed[modality] != len(sizes):
                raise ValueError(f"unused {modality.name} encoder metadata")
        return input_ids, ranges, modalities

    def get_mm_data(self, prompt, embeddings, **metadata) -> MultimodalInputs:
        """Rebuild native multimodal inputs from encoder-disaggregated outputs."""
        input_ids, ranges, modalities = self.build_input_ids(
            prompt,
            img_grid_thw=metadata.get("img_grid_thw", metadata.get("image_grid_thw")),
            video_grid_thw=metadata.get("video_grid_thw"),
            audio_seq_lens=metadata.get("audio_feature_lens"),
        )
        consumed = {modality: 0 for modality in Modality.all()}
        mm_items = []
        for modality, placeholder_range in zip(modalities, ranges):
            modality_embeddings = embeddings.get(modality)
            if modality_embeddings is None:
                raise ValueError(f"missing {modality.name} encoder embeddings")
            start = consumed[modality]
            end = start + placeholder_range[1] - placeholder_range[0]
            embedding = modality_embeddings[start:end]
            if len(embedding) != end - start:
                raise ValueError(f"incomplete {modality.name} encoder embeddings")

            item = MultimodalDataItem(
                modality=modality,
                placeholder_ranges=[placeholder_range],
                precomputed_embeddings=embedding,
            )
            item.set_pad_value()
            mm_items.append(item)
            consumed[modality] = end

        for modality, modality_embeddings in embeddings.items():
            if modality in consumed and consumed[modality] != len(modality_embeddings):
                raise ValueError(f"unused {modality.name} encoder embeddings")

        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_start_id=getattr(self.hf_config, "vision_start_token_id", None),
            im_end_id=getattr(self.hf_config, "vision_end_token_id", None),
            im_token_id=getattr(self.hf_config, "image_token_id", None),
            video_token_id=getattr(self.hf_config, "video_token_id", None),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_id=getattr(self.hf_config, "audio_start_token_id", None),
            audio_end_id=getattr(self.hf_config, "audio_end_token_id", None),
        )

    @classmethod
    def _to_grid_list(cls, value) -> list[tuple[int, int, int]]:
        if value is None:
            return []
        return [tuple(map(int, row)) for row in cls._to_numpy(value).reshape(-1, 3)]

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

    async def load_images_async(self, image_sources: list) -> list[Image.Image]:
        return await asyncio.gather(*(self.load_image_async(source) for source in image_sources))

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

        This mirrors upstream SGLang's processor layering. Callers should use
        ``process_and_combine_mm_data_async`` so this CPU work runs in the
        isolated multimodal processor executor.
        """
        processor_inputs = {
            "text": [input_text],
            "images": images or None,
            "padding": True,
            "return_tensors": "pt",
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

    async def process_and_combine_mm_data_async(
        self,
        input_text: str,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        **processor_kwargs,
    ) -> MultimodalInputs:
        """Run HF processing and output collection outside the event loop."""
        return await self.mm_processor_executor.run(
            self.process_and_combine_mm_data,
            input_text,
            images,
            videos,
            audios,
            **processor_kwargs,
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        self.mm_processor_executor.shutdown()
