"""Assign multimodal parts to encoders and dispatch them concurrently."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
import orjson

from sgl_jax.srt.managers.io_struct import GenerateReqInput, ImageData
from sgl_jax.srt.multimodal.common.modality_enum import Modality, flatten_nested_list

def create_part_req_id(req_id: str, part_idx: int) -> str:
    return f"{req_id}_local_part_{part_idx}"


class EncoderRequestDispatcher:
    """Dispatch encoder requests through a reusable HTTP client."""

    def __init__(self, timeout: float | None) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def dispatch(
        self,
        request: GenerateReqInput,
        encoder_urls: list[str],
    ) -> tuple[dict[Modality, list[int]], asyncio.Task[None]]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        client = self._client

        items_by_modality = {}
        for name, modality in (
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ):
            data = getattr(request, name, None)
            if data is None:
                continue
            items = []
            for item in flatten_nested_list(data):
                if item is None:
                    continue
                if isinstance(item, ImageData):
                    item = item.url
                elif isinstance(item, dict) and "url" in item:
                    item = item["url"]
                items.append(item)
            if items:
                items_by_modality[modality] = items

        assignments = {}
        encoder_indices = list(range(len(encoder_urls)))
        random.shuffle(encoder_indices)
        offset = 0
        for modality, items in items_by_modality.items():
            base, remainder = divmod(len(items), len(encoder_urls))
            counts = [base] * len(encoder_urls)
            for index in range(remainder):
                counts[encoder_indices[(offset + index) % len(encoder_urls)]] += 1
            assignments[modality] = counts
            offset = (offset + remainder) % len(encoder_urls)

        num_parts = sum(count > 0 for counts in assignments.values() for count in counts)
        encode_requests = []
        for modality, counts in assignments.items():
            items = items_by_modality[modality]
            item_offset = 0
            for encoder_idx, count in enumerate(counts):
                if count == 0:
                    continue
                part_idx = len(encode_requests)
                encode_requests.append(
                    (
                        encoder_urls[encoder_idx],
                        {
                            "req_id": create_part_req_id(request.rid, part_idx),
                            "mm_items": items[item_offset : item_offset + count],
                            "num_parts": num_parts,
                            "part_idx": part_idx,
                            "modality": modality.name,
                        },
                    )
                )
                item_offset += count

        async def send_encode_requests() -> None:
            async def send_one(encoder_url: str, payload: dict[str, Any]) -> None:
                url = f"{encoder_url.rstrip('/')}/encode"
                response = await client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"content-type": "application/json"},
                )
                response.raise_for_status()

            results = await asyncio.gather(
                *(send_one(*encode_request) for encode_request in encode_requests),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result

        task = asyncio.create_task(
            send_encode_requests(),
            name=f"encoder-dispatch-{request.rid}",
        )

        return assignments, task

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
