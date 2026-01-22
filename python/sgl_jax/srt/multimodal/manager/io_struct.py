import dataclasses
import uuid
from enum import Enum, auto

import numpy as np
from pydantic import BaseModel


class ImageGenerationsRequest(BaseModel):
    prompt: str
    neg_prompt: str = (
        "blurry, low quality, inconsistent lighting, floating, disconnected from scene"
    )
    model: str | None = None
    n: int | None = 1
    response_format: str | None = "url"  # url | b64_json
    size: str | None = "1024x1024"  # e.g., 1024x1024
    background: str | None = "auto"  # transparent | opaque | auto
    output_format: str | None = None  # png | jpeg | webp
    save_output: bool = True
    num_inference_steps: int | None = None
    # guidance_scale: float | None = 7.5


class ImageResponse(BaseModel):
    id: str
    b64_json: str | None = None
    url: str | None = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    neg_prompt: str = (
        "blurry, low quality, inconsistent lighting, floating, disconnected from scene"
    )
    input_reference: str | None = None
    model: str | None = None
    seconds: int | None = 4
    size: str | None = "720x1280"
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    save_output: bool = True
    # guidance_scale: float | None = 5.0
    # text_embeds: np.ndarray | None = None
    # latents: np.ndarray | None = None


class VideoResponse(BaseModel):
    id: str
    path: str | None = None


class DataType(Enum):
    IMAGE = auto()
    VIDEO = auto()

    def get_default_extension(self) -> str:
        if self == DataType.IMAGE:
            return "jpg"
        else:
            return "mp4"


@dataclasses.dataclass
class GenerateMMReqInput:
    rid: str | None = None
    data_type: DataType | None = None
    prompt: str | None = None
    neg_prompt: str | None = (
        "blurry, low quality, inconsistent lighting, floating, disconnected from scene"
    )
    input_ids: list[int] | None = None
    n: int | None = 1
    input_reference: str | None = None
    size: str = None
    seconds: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    save_output: bool = True
    output_format: str | None = None
    background: str | None = None
    response_format: str | None = None

    def __post_init__(self):
        self.rid = uuid.uuid4().hex


@dataclasses.dataclass
class TokenizedGenerateMMReqInput:
    rid: str | None = None
    data_type: DataType | None = None
    prompt: str | None = None
    input_ids: list[int] | None = None
    negative_prompt: str | None = None
    negative_input_ids: list[int] | None = None
    n: int | None = 1
    input_reference: str | None = None
    preprocessed_image: np.ndarray | None = None
    size: str = None
    seconds: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    output_format: str | None = None
    save_output: bool = True
    background: str | None = None
    response_format: str | None = None
