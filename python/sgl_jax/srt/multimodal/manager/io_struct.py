import dataclasses
import uuid
from enum import Enum, auto

import numpy as np
from pydantic import BaseModel, Field


class ImageGenerationsRequest(BaseModel):
    prompt: str
    neg_prompt: str = Field(
        default=(
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
            "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
            "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
            "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
            "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
            "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
            "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
            "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
            "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
            "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
            "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
        ),
        alias="negative_prompt",
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
    neg_prompt: str = Field(
        default=(
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
            "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
            "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
            "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
            "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
            "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
            "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
            "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
            "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
            "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
            "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
        ),
        alias="negative_prompt",
    )
    input_reference: str | None = None
    model: str | None = None
    seconds: int | None = 4
    size: str | None = "720x1280"
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    save_output: bool = True
    guidance_scale: float | None = None
    stg_scale: float | None = None
    rescale_scale: float | None = None
    # text_embeds: np.ndarray | None = None
    # latents: np.ndarray | None = None


class VideoResponse(BaseModel):
    id: str
    path: str | None = None


class AudioSpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request."""

    input: str
    model: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str | None = None
    stream_format: str | None = None


class AudioTranscriptionRequest(BaseModel):
    file: bytes | None = None
    url: str | None = None
    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float | None = None
    timestamp_granularities: list[str] | None = None
    chunking_strategy: dict | None = None
    known_speaker_names: list[str] | None = None
    known_speaker_references: list[str] | None = None
    include: list[str] | None = None
    stream: bool = False


class AudioTranscriptionResponse(BaseModel):
    """OpenAI transcription response (json format)."""

    text: str
    task: str | None = None
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None
    words: list[dict] | None = None

    usage: dict | None = None


class DataType(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    def get_default_extension(self) -> str:
        if self == DataType.IMAGE:
            return "jpg"
        elif self == DataType.VIDEO:
            return "mp4"
        else:
            return "wav"


@dataclasses.dataclass
class GenerateMMReqInput:
    rid: str | None = None
    data_type: DataType | None = None
    prompt: str | None = None
    neg_prompt: str | None = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )
    input_ids: list[int] | None = None
    stream: bool = False
    n: int | None = 1
    input_reference: str | None = None
    image_data: list[str] | str | None = None
    video_data: list[str] | str | None = None
    size: str = None
    seconds: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    save_output: bool = True
    output_format: str | None = None
    background: str | None = None
    response_format: str | None = None
    guidance_scale: float | None = None
    stg_scale: float | None = None
    rescale_scale: float | None = None

    def __post_init__(self):
        self.rid = uuid.uuid4().hex


@dataclasses.dataclass
class GenerateOmniReqInput:
    """Input request for Omni/VLM chat/completions."""

    rid: str | None = None
    prompt: str | None = None
    input_ids: list[int] | None = None
    image_data: list[str] | str | None = None
    video_data: list[str] | str | None = None
    audio_data: list[str] | str | None = None
    stream: bool = False
    n: int | None = 1
    sampling_params: dict | None = None
    stop: str | list[str] | None = None

    def __post_init__(self):
        if self.rid is None:
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
    mm_inputs: "OmniInputs | None" = None
    size: str = None
    seconds: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    output_format: str | None = None
    save_output: bool = True
    background: str | None = None
    response_format: str | None = None
    guidance_scale: float | None = None
    stg_scale: float | None = None
    rescale_scale: float | None = None


@dataclasses.dataclass
class TokenizedGenerateOmniReqInput:
    rid: str | None = None
    prompt: str | None = None
    input_ids: list[int] | None = None
    mm_inputs: "OmniInputs | None" = None
    stream: bool = False
    n: int | None = 1
    sampling_params: dict | None = None
    stop: str | list[str] | None = None


@dataclasses.dataclass
class TokenizedGenerateAudioReqInput:
    rid: str | None = None
    audio_mode: str | None = None
    data_type: DataType | None = None
    text: str | None = None
    text_input_ids: list[int] | None = None
    prompt: str | None = None
    prompt_input_ids: list[int] | None = None
    mel_input: np.ndarray | None = None
    mel_input_lens: np.ndarray | None = None

    # Audio configuration
    sample_rate: int = 24000
    n_q: int | None = None


@dataclasses.dataclass
class OmniInputs:
    audio_features: np.ndarray | None = None
    audio_feature_lengths: list[int] | None = None
    pixel_values: np.ndarray | None = None
    pixel_values_videos: np.ndarray | None = None
    image_grid_thw: list[tuple[int, int, int]] | None = None
    video_grid_thw: list[tuple[int, int, int]] | None = None
    second_per_grid_ts: list[float] | None = None
    mrope_positions: np.ndarray | None = None
    mrope_position_delta: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    pad_values: list[int] | None = None
    multimodal_embeddings: np.ndarray | None = None
    deepstack_visual_embedding: np.ndarray | None = None
    deepstack_visual_pos_mask: np.ndarray | None = None
