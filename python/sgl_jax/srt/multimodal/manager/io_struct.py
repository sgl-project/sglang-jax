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


# === OpenAI Audio API Compatibility ===

class AudioSpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request."""
    input: str                                    # 必需，最大 4096 字符
    model: str                                    # tts-1, tts-1-hd, gpt-4o-mini-tts, etc.
    voice: str                                    # alloy, echo, fable, onyx, nova, shimmer, etc.
    response_format: str = "mp3"                 # mp3, opus, aac, flac, wav, pcm
    speed: float = 1.0                           # 0.25-4.0
    instructions: str | None = None              # 仅 gpt-4o 模型支持
    stream_format: str | None = None             # sse 或 audio（仅 gpt-4o 模型）


class AudioTranscriptionRequest(BaseModel):
    """OpenAI /v1/audio/transcriptions request.

    Note: 这是内部表示，实际 HTTP 端点支持 multipart/form-data 或 url
    """
    file: bytes | None = None                    # 音频文件字节（上传方式）
    url: str | None = None                       # 音频文件 URL（URL 方式）
    model: str                                   # gpt-4o-transcribe, whisper-1, etc.
    language: str | None = None                  # ISO-639-1 代码
    prompt: str | None = None                    # 上下文提示
    response_format: str = "json"                # json, text, srt, verbose_json, vtt, diarized_json
    temperature: float | None = None             # 0-1
    timestamp_granularities: list[str] | None = None  # ["word", "segment"]
    chunking_strategy: dict | None = None        # auto 或 server_vad
    known_speaker_names: list[str] | None = None # 最多 4 个
    known_speaker_references: list[str] | None = None  # data URLs
    include: list[str] | None = None             # ["logprobs"]
    stream: bool = False                         # SSE 流式传输


class AudioTranscriptionResponse(BaseModel):
    """OpenAI transcription response (json format)."""
    text: str                                    # 转录文本
    # verbose_json 额外字段:
    task: str | None = None                      # "transcribe"
    language: str | None = None                  # 检测到的语言
    duration: float | None = None                # 音频时长（秒）
    segments: list[dict] | None = None           # 时间戳片段
    words: list[dict] | None = None              # 词级时间戳

    # usage 统计
    usage: dict | None = None                    # token 或 duration 统计


# === OpenAI Audio Chat Compatibility ===

class InputAudio(BaseModel):
    data: str | None = None       # Base64 encoded audio bytes
    url: str | None = None        # URL to audio file
    format: str                   # wav, mp3

class ContentPart(BaseModel):
    type: str        # "text" | "input_audio"
    text: str | None = None
    input_audio: InputAudio | None = None

class ChatMessage(BaseModel):
    role: str        # "user", "assistant", "system"
    content: str | list[ContentPart]

class AudioOutputConfig(BaseModel):
    voice: str       # e.g. "alloy"
    format: str      # "wav", "mp3"

class GenerateOpenAIAudioInput(BaseModel):
    """
    OpenAI-compatible request body for multimodal audio chat.
    Maps to /api/v1/chat/completions
    """
    model: str
    messages: list[ChatMessage]
    modalities: list[str] | None = ["text", "audio"]
    audio: AudioOutputConfig | None = None
    max_tokens: int | None = 2048
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    stream: bool = False


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
        "blurry, low quality, inconsistent lighting, floating, disconnected from scene"
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

    def __post_init__(self):
        self.rid = uuid.uuid4().hex


@dataclasses.dataclass
class GenerateVLMReqInput:
    """Input request for VLM chat/completions."""

    rid: str | None = None
    prompt: str | None = None
    input_ids: list[int] | None = None
    image_data: list[str] | str | None = None
    video_data: list[str] | str | None = None
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
    mm_inputs: "VLMMInputs | None" = None
    size: str = None
    seconds: int | None = None
    fps: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    output_format: str | None = None
    save_output: bool = True
    background: str | None = None
    response_format: str | None = None


@dataclasses.dataclass
class TokenizedGenerateVLMReqInput:
    rid: str | None = None
    prompt: str | None = None
    input_ids: list[int] | None = None
    mm_inputs: "VLMMInputs | None" = None
    stream: bool = False
    n: int | None = 1
    sampling_params: dict | None = None
    stop: str | list[str] | None = None


@dataclasses.dataclass
class VLMMInputs:
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
