from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image

from sgl_jax.srt.managers.io_struct import BatchTokenIDOut, TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.manager.io_struct import DataType, VLMMInputs
from sgl_jax.srt.sampling.sampling_params import SamplingParams

NegativePromptSuffix = "_negative"

# MiMo Audio aggregation constants
MIMO_AUDIO_GROUP_SIZE = 4
MIMO_AUDIO_CHANNELS = 8
MIMO_SPEECH_EMPTY_IDS = (1024, 1024, 128, 128, 128, 128, 128, 128)
MIMO_EMPTY_IDX = 151667       # <|empty|> - text placeholder in audio patches
MIMO_SOSP_IDX = 151665        # <|sosp|> - start of speech
MIMO_EOSP_IDX = 151666        # <|eosp|> - end of speech
MIMO_SOSTM_IDX = 151670       # <|sostm|> - start of streaming
MIMO_EOSTM_IDX = 151671       # <|eostm|> - end of streaming
MIMO_EOT_IDX = 151672         # <|eot|> - end of turn
MIMO_TEXT_PADDING = -100      # padding for text patches (timesteps 1-3)


@dataclass
class Req:
    """
    Complete state passed through the diffusion execution.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """

    # TODO(will): double check that args are separate from server_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    data_type: DataType | None = None

    rid: str | None = None
    # generator: jax.random.PRNGKey | list[jax.random.PRNGKey] | None = None

    # Image inputs
    image_path: str | None = None
    # Image encoder hidden states
    image_embeds: list[jax.Array] = field(default_factory=list)
    pil_image: jax.Array | PIL.Image.Image | None = None
    pixel_values: jax.Array | PIL.Image.Image | None = None
    preprocessed_image: jax.Array | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    input_ids: list[int] | None = None
    origin_input_text: str | list[str] | None = None
    origin_input_ids: list[int] | None = None
    negative_prompt: str | list[str] | None = None
    negative_input_ids: list[int] | None = None
    prompt_path: str | None = None
    output_path: str = "outputs/"
    # without extension
    output_file_name: str | None = None
    output_file_ext: str | None = None
    # Primary encoder embeddings
    prompt_embeds: list[jax.Array] | jax.Array = field(default_factory=list)
    negative_prompt_embeds: list[jax.Array] | None = None
    prompt_attention_mask: list[jax.Array] | None = None
    negative_attention_mask: list[jax.Array] | None = None
    clip_embedding_pos: list[jax.Array] | None = None
    clip_embedding_neg: list[jax.Array] | None = None

    pooled_embeds: list[jax.Array] = field(default_factory=list)
    neg_pooled_embeds: list[jax.Array] = field(default_factory=list)

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

    # VLM inputs/outputs
    vlm_inputs: VLMMInputs | None = None
    vision_embeds: jax.Array | None = None
    input_embeds: jax.Array | None = None
    image_grid_thw: tuple | None = None
    video_grid_thw: tuple | None = None
    cache_input_ids: list[int] | None = None

    # Batch info
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    seeds: list[int] | None = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: jax.Array | None = None
    raw_latent_shape: jax.Array | None = None
    noise_pred: jax.Array | None = None
    image_latent: jax.Array | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: list[int] | int = 1  # Default for image models
    num_frames_round_down: bool = (
        False  # Whether to round down num_frames if it's not divisible by num_gpus
    )

    # Original dimensions (before VAE scaling)
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    fps: list[int] | int | None = None
    height_not_provided: bool = False
    width_not_provided: bool = False

    # Timesteps
    timesteps: jax.Array | None = None
    timestep: jax.Array | float | int | None = None
    step_index: int | None = None
    boundary_ratio: float | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    true_cfg_scale: float | None = None  # qwen-image specific now

    n_tokens: int | None = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False
    trajectory_timesteps: list[jax.Array] | None = None
    trajectory_latents: jax.Array | None = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_output: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    # teacache_params: TeaCacheParams | WanTeaCacheParams | None = None

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0
    # perf_logger: PerformanceLogger | None = None

    # stage logging
    # logging_info: PipelineLoggingInfo = field(default_factory=PipelineLoggingInfo)

    # profile
    profile: bool = False
    num_profiled_timesteps: int = 8

    # debugging
    debug: bool = False

    # results
    output: jax.Array | None = None

    # Audio inputs
    audio_input: jax.Array | None = None
    mel_input: jax.Array | None = None  # Preprocessed mel spectrogram [B, T, n_mels]
    mel_input_lens: jax.Array | None = None  # Lengths for mel spectrogram
    codes: jax.Array | None = None
    audio_codes: jax.Array | None = None
    use_quantizer: bool = True
    n_q: int | None = None
    sample_rate: int = 24000
    audio_mode: str | None = None  # "encode", "decode", "generation", "tts"

    # TTS inputs
    text: str | None = None                     # TTS text to synthesize
    text_input_ids: list[int] | None = None     # Tokenized TTS text
    prompt_input_ids: list[int] | None = None   # Tokenized voice style prompt

    # Audio backbone outputs
    backbone_cache: Any = None
    generated_text_tokens: jax.Array | None = None
    generated_audio_tokens: jax.Array | None = None
    text_logits: jax.Array | None = None

    def to_stage_reqs(self, scheduler: str):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "to_stage_reqs: scheduler=%s, rid=%s, output=%s, audio_codes=%s, input_ids=%s",
            scheduler,
            self.rid,
            self.output is not None,
            self.audio_codes is not None,
            self.input_ids is not None,
        )
        if scheduler == "auto_regressive":
            is_vlm_request = (
                self.vlm_inputs is not None or self.extra.get("sampling_params") is not None
            )
            if is_vlm_request:
                params = self.extra.get("sampling_params")
                if isinstance(params, SamplingParams):
                    sampling_params = params
                elif isinstance(params, dict):
                    params = dict(params)
                    if self.extra.get("stop") and "stop" not in params:
                        params["stop"] = self.extra["stop"]
                    sampling_params = SamplingParams(**params)
                else:
                    sampling_params = SamplingParams(
                        max_new_tokens=1,
                        stop=self.extra.get("stop"),
                    )
                tokenized_req = TokenizedGenerateReqInput(
                    rid=self.rid,
                    input_ids=self.input_ids or self.origin_input_ids,
                    sampling_params=sampling_params,
                    stream=bool(self.extra.get("stream", False)),
                )
                tokenized_req.mm_inputs = self.vlm_inputs
                return [tokenized_req]
            return [
                TokenizedGenerateReqInput(
                    rid=self.rid,
                    input_ids=self.input_ids,
                    sampling_params=SamplingParams(max_new_tokens=1),
                    return_hidden_states=True,
                ),
                TokenizedGenerateReqInput(
                    rid=self.rid + NegativePromptSuffix,
                    input_ids=self.negative_input_ids,
                    sampling_params=SamplingParams(max_new_tokens=1),
                    return_hidden_states=True,
                ),
            ]
        elif scheduler == "audio_backbone":
            if self.output is not None and self.audio_codes is None:
                self.audio_codes = self.output
                self.output = None
            logger.info(
                "to_stage_reqs audio_backbone: audio_codes=%s, text_input_ids=%s",
                self.audio_codes.shape if self.audio_codes is not None else None,
                len(self.text_input_ids) if self.text_input_ids else None,
            )
            # Aggregate text and audio into backbone input format [1, 9, seq_len]
            backbone_input = self._build_backbone_input()
            if backbone_input is not None:
                self.input_ids = backbone_input
                logger.info(
                    "to_stage_reqs audio_backbone: built input_ids=%s",
                    self.input_ids.shape if self.input_ids is not None else None,
                )
            return [self]
        elif scheduler == "audio_decoder":
            if self.generated_audio_tokens is not None and self.codes is None:
                # generated_audio_tokens shape: [B, group_size, audio_channels]
                # decoder expects codes shape: [n_q, seq_len] = [audio_channels, group_size]
                tokens = self.generated_audio_tokens
                if tokens.ndim == 3:
                    # Take first batch item and transpose
                    tokens = tokens[0].T  # [audio_channels, group_size]
                self.codes = tokens
                self.generated_audio_tokens = None
            self.audio_mode = "decode"
            return [self]
        else:
            return [self]

    @staticmethod
    def from_stage(stage_result: Any, req_store: dict):
        """Convert stage output to a Req object.

        Args:
            stage_result: Output from a stage (BatchTokenIDOut or Req).
            req_store: Dictionary mapping rid to ReqTrackingState.

        Returns:
            The Req object, or None if not ready or if the request was aborted
            (rid no longer in req_store).
        """
        if type(stage_result) is BatchTokenIDOut:
            if not getattr(stage_result, "output_hidden_states_for_mm", None):
                return stage_result
            req = None
            for i, rid in enumerate(stage_result.rids):
                if rid.endswith(NegativePromptSuffix):
                    rid = rid[: -len(NegativePromptSuffix)]
                    if rid not in req_store:
                        # Request was aborted, skip it
                        return None
                    # req_store now contains ReqTrackingState, access .req
                    tracking_state = req_store[rid]
                    req = tracking_state.req if hasattr(tracking_state, "req") else tracking_state
                    req.negative_prompt_embeds = stage_result.output_hidden_states_for_mm[i][0]
                else:
                    if rid not in req_store:
                        # Request was aborted, skip it
                        return None
                    # req_store now contains ReqTrackingState, access .req
                    tracking_state = req_store[rid]
                    req = tracking_state.req if hasattr(tracking_state, "req") else tracking_state
                    req.prompt_embeds = stage_result.output_hidden_states_for_mm[i][0]
            if req is None or req.prompt_embeds is None or req.negative_prompt_embeds is None:
                return None
            return req
        else:
            return stage_result

    # =========================================================================
    # MiMo Audio Aggregation Methods
    # =========================================================================

    def _build_backbone_input(self) -> jax.Array:
        """Build backbone input by aggregating text and audio into [1, 9, seq_len] format.

        The backbone expects input_ids with shape [B, 1+audio_channels, seq_len] where:
        - Channel 0: text tokens (with -100 padding for non-first timesteps in each group)
        - Channels 1-8: audio codes (or speech_empty_ids for text patches)

        Returns:
            input_ids: [1, 9, seq_len] tensor ready for backbone forward pass
        """
        import logging

        logger = logging.getLogger(__name__)
        segments = []

        if self.audio_mode == "tts":
            # TTS: text patches + <|sosp|> marker (backbone will generate audio)
            if self.text_input_ids:
                segments.append(self._build_text_segment(self.text_input_ids))
            # Add start-of-speech marker to signal audio generation
            segments.append(self._build_sosp_segment())

        elif self.audio_mode == "generation":
            # Audio generation/continuation: audio patches + optional text
            if self.audio_codes is not None:
                segments.append(self._build_audio_segment(self.audio_codes))
            if self.text_input_ids:
                segments.append(self._build_text_segment(self.text_input_ids))

        elif self.audio_mode == "audio_understanding":
            # Audio understanding: audio patches + question text
            if self.audio_codes is not None:
                segments.append(self._build_audio_segment(self.audio_codes))
            if self.text_input_ids:
                segments.append(self._build_text_segment(self.text_input_ids))

        else:
            # Default: just use whatever is available
            if self.audio_codes is not None:
                segments.append(self._build_audio_segment(self.audio_codes))
            if self.text_input_ids:
                segments.append(self._build_text_segment(self.text_input_ids))

        if not segments:
            logger.warning("No segments to build for backbone input, rid=%s", self.rid)
            return None

        # Concatenate all segments along sequence dimension
        input_ids = jnp.concatenate(segments, axis=1)  # [9, total_seq_len]
        logger.info(
            "Built backbone input: audio_mode=%s, shape=%s, rid=%s",
            self.audio_mode,
            input_ids.shape,
            self.rid,
        )
        return input_ids[None, :, :]  # [1, 9, total_seq_len]

    def _build_text_segment(self, text_ids: list) -> jax.Array:
        """Build a text-only segment where text tokens are active.

        For each text token, creates a group of (group_size) timesteps:
        - Timestep 0: actual text token
        - Timesteps 1 to group_size-1: -100 (padding)

        Audio channels are filled with speech_empty_ids.

        Args:
            text_ids: List of text token IDs

        Returns:
            Tensor of shape [9, len(text_ids) * group_size]
        """
        if not text_ids:
            return jnp.zeros((1 + MIMO_AUDIO_CHANNELS, 0), dtype=jnp.int32)

        # Expand each text token with padding: [t1] -> [t1, -100, -100, -100]
        text_expanded = []
        for t in text_ids:
            text_expanded.append(t)
            text_expanded.extend([MIMO_TEXT_PADDING] * (MIMO_AUDIO_GROUP_SIZE - 1))

        seq_len = len(text_expanded)
        text_row = jnp.array(text_expanded, dtype=jnp.int32)

        # Build audio channel rows with speech_empty_ids
        rows = [text_row]
        for ch in range(MIMO_AUDIO_CHANNELS):
            row = jnp.full((seq_len,), MIMO_SPEECH_EMPTY_IDS[ch], dtype=jnp.int32)
            rows.append(row)

        return jnp.stack(rows, axis=0)  # [9, seq_len]

    def _build_audio_segment(self, audio_codes: jax.Array) -> jax.Array:
        """Build an audio-only segment where audio codes are active.

        Text channel is filled with <|empty|> token (151667).
        Audio codes are placed directly in channels 1-8.

        Args:
            audio_codes: Audio codes tensor of shape [8, T_audio]

        Returns:
            Tensor of shape [9, T_audio]
        """
        if audio_codes is None:
            return jnp.zeros((1 + MIMO_AUDIO_CHANNELS, 0), dtype=jnp.int32)

        # Ensure audio_codes is a jax array
        if isinstance(audio_codes, np.ndarray):
            audio_codes = jnp.array(audio_codes)

        T_audio = audio_codes.shape[1]

        # Text channel: all timesteps filled with <|empty|> token
        text_row = jnp.full((T_audio,), MIMO_EMPTY_IDX, dtype=jnp.int32)

        # Concatenate text row with audio codes
        return jnp.concatenate(
            [text_row[None, :], audio_codes.astype(jnp.int32)], axis=0
        )  # [9, T_audio]

    def _build_sosp_segment(self) -> jax.Array:
        """Build a <|sosp|> (start of speech) segment.

        This marks the beginning of audio generation.
        Creates one group with:
        - Text channel: [<|sosp|>, -100, -100, -100]
        - Audio channels: speech_empty_ids

        Returns:
            Tensor of shape [9, group_size]
        """
        # Text row: sosp token followed by padding
        text_row = jnp.array(
            [MIMO_SOSP_IDX] + [MIMO_TEXT_PADDING] * (MIMO_AUDIO_GROUP_SIZE - 1),
            dtype=jnp.int32,
        )

        rows = [text_row]
        for ch in range(MIMO_AUDIO_CHANNELS):
            row = jnp.full((MIMO_AUDIO_GROUP_SIZE,), MIMO_SPEECH_EMPTY_IDS[ch], dtype=jnp.int32)
            rows.append(row)

        return jnp.stack(rows, axis=0)  # [9, group_size]

    def _build_eosp_segment(self) -> jax.Array:
        """Build a <|eosp|> (end of speech) segment.

        This marks the end of audio.
        Creates one group with:
        - Text channel: [<|eosp|>, -100, -100, -100]
        - Audio channels: speech_empty_ids

        Returns:
            Tensor of shape [9, group_size]
        """
        text_row = jnp.array(
            [MIMO_EOSP_IDX] + [MIMO_TEXT_PADDING] * (MIMO_AUDIO_GROUP_SIZE - 1),
            dtype=jnp.int32,
        )

        rows = [text_row]
        for ch in range(MIMO_AUDIO_CHANNELS):
            row = jnp.full((MIMO_AUDIO_GROUP_SIZE,), MIMO_SPEECH_EMPTY_IDS[ch], dtype=jnp.int32)
            rows.append(row)

        return jnp.stack(rows, axis=0)  # [9, group_size]
