from dataclasses import dataclass, field
from typing import Any

import jax
import PIL.Image

from sgl_jax.srt.managers.io_struct import BatchTokenIDOut, TokenizedGenerateReqInput
from sgl_jax.srt.sampling.sampling_params import SamplingParams


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
    # data_type: DataType

    rid: str | None = None
    # 随机数生成器
    # generator: jax.random.PRNGKey | list[jax.random.PRNGKey] | None = None

    # Image inputs
    image_path: str | None = None
    # Image encoder hidden states
    # 图像或者视频过vit之后的tensor
    image_embeds: list[jax.Array] = field(default_factory=list)
    pil_image: jax.Array | PIL.Image.Image | None = None
    pixel_values: jax.Array | PIL.Image.Image | None = None
    preprocessed_image: jax.Array | None = None

    # Text inputs
    prompt: str | list[str] | None = None
    input_ids: list[int] | None = None
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
    # clip 插入模型的位置
    clip_embedding_pos: list[jax.Array] | None = None
    clip_embedding_neg: list[jax.Array] | None = None

    # 文本模型的输出
    pooled_embeds: list[jax.Array] = field(default_factory=list)
    neg_pooled_embeds: list[jax.Array] = field(default_factory=list)

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

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
    ## guidance_scale 越大，图像或者视频更贴近prompt
    guidance_scale: float = 1.0
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

    def to_stage_req(self, scheduler: str):
        if scheduler == "auto_regressive":
            return TokenizedGenerateReqInput(
                rid=self.rid,
                input_ids=self.input_ids,
                sampling_params=SamplingParams(max_new_tokens=1),
                return_hidden_states=True,
            )
        else:
            return self

    @staticmethod
    def from_stage(stage_result: Any):
        if type(stage_result) is BatchTokenIDOut:
            return Req(rid=stage_result.rids[0], prompt_embeds=stage_result.output_hidden_states)
        else:
            return stage_result
