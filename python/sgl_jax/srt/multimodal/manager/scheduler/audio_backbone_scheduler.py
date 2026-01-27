"""Audio Backbone Scheduler for MiMo Audio LLM inference."""

import logging

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import MIMO_EMPTY_IDX, Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_worker import (
    AudioBackboneModelWorker,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)

# Deprecated: Use model_config.eos_token_id instead
# Kept for backward compatibility if config doesn't have eos_token_id
MIMO_EOS_TOKENS_FALLBACK = {151672, 151643, 151645, 151671}  # EOT, EOS, IM_END, EOSTM
MIMO_GROUP_SIZE = 4

DEFAULT_AUDIO_INPUT_GROUP_PADDINGS = [64, 128, 256, 512, 1024]
DEFAULT_AUDIO_GENERATION_GROUP_PADDINGS = [64, 128, 256, 512, 1024]


class AudioBackboneScheduler:
    """Scheduler for MiMo Audio backbone (LLM with audio token generation).

    Responsibilities:
    - Receive requests containing text + audio tokens from previous stage.
    - Run main transformer forward pass to generate text logits and local hidden states.
    - Run patch decoder to generate audio tokens for each group.
    - Send generated tokens to next stage for audio decoding.
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend,
        mesh: jax.sharding.Mesh,
        model_class,
        **kwargs,
    ):
        """Initialize the AudioBackboneScheduler.

        Args:
            server_args: Multimodal server configuration.
            communication_backend: Backend used to receive/send requests.
            mesh: JAX device mesh used for sharding inputs/outputs.
            model_class: The backbone model class (MiMoAudioForCausalLM).
        """
        self._comm_backend = communication_backend
        self.mesh = mesh
        self.server_args = server_args
        self.aborted_rids: set[str] = set()

        self._cache_locations: dict[str, np.ndarray] = {}
        self._total_seq_lens: dict[str, int] = {}

        self.rng_key = jax.random.PRNGKey(42)

        self.input_group_paddings = sorted(DEFAULT_AUDIO_INPUT_GROUP_PADDINGS)
        self.generation_group_paddings = sorted(DEFAULT_AUDIO_GENERATION_GROUP_PADDINGS)

        with self.mesh:
            self.backbone_worker = AudioBackboneModelWorker(
                model_class=model_class, mesh=self.mesh, server_args=server_args
            )

        # Initialize EOS token IDs from model config
        self._init_eos_token_ids()

    def _init_eos_token_ids(self):
        """Initialize EOS token IDs from model config or fallback to hardcoded values."""
        model_config = self.backbone_worker.model_runner.model_config
        eos_token_id = getattr(model_config, "eos_token_id", None)

        if eos_token_id is not None:
            # Convert to set for efficient membership testing
            if isinstance(eos_token_id, int):
                self.eos_token_ids = {eos_token_id}
            else:
                # Assume it's a list or set
                self.eos_token_ids = set(eos_token_id)
            logger.info("Loaded EOS token IDs from model config: %s", self.eos_token_ids)
        else:
            # Fallback to hardcoded values for backward compatibility
            self.eos_token_ids = MIMO_EOS_TOKENS_FALLBACK
            logger.warning(
                "Model config does not have eos_token_id, using fallback: %s", self.eos_token_ids
            )

    def _find_padding_size(self, actual_size: int, padding_list: list[int]) -> int:
        for size in padding_list:
            if size >= actual_size:
                return size
        logger.warning(
            "Actual size %d exceeds all padding sizes %s, will trigger recompilation",
            actual_size, padding_list
        )
        return actual_size

    def event_loop_normal(self):
        """Main blocking loop for processing requests.

        Repeatedly polls the communication_backend for requests, processes them
        through the backbone model, and sends results to the next stage.
        """
        # Set mesh context for all JAX operations (required for shard_map, KV cache updates, etc.)
        with self.mesh:
            while True:
                reqs = self._comm_backend.recv_requests()
                if reqs is not None and len(reqs) > 0:
                    valid_reqs = []
                    for req in reqs:
                        if isinstance(req, AbortReq):
                            logger.info("AudioBackboneScheduler received abort for rid=%s", req.rid)
                            self.aborted_rids.add(req.rid)
                        elif isinstance(req, Req):
                            if req.rid in self.aborted_rids:
                                logger.info(
                                    "AudioBackboneScheduler skipping aborted request rid=%s", req.rid
                                )
                                self.aborted_rids.discard(req.rid)
                                continue
                            self.preprocess(req)
                            valid_reqs.append(req)
                        else:
                            logger.warning(
                                "AudioBackboneScheduler received unknown request type: %s", type(req)
                            )

                    if valid_reqs:
                        self.run_backbone_batch(valid_reqs)
                else:
                    self._comm_backend.wait_for_new_requests(0.001)


    def preprocess(self, req: Req):
        """Apply preprocessing to a single Req.

        Moves input_ids to device with proper sharding.
        """
        sharding = NamedSharding(self.mesh, PartitionSpec())
        if req.input_ids is not None:
            req.input_ids = device_array(req.input_ids, sharding=sharding)
        if req.audio_codes is not None:
            req.audio_codes = device_array(req.audio_codes, sharding=sharding)

    def _reset_cache_state(self, rid: str = None):
        """Reset cache state for a specific request or all requests."""
        if rid is not None:
            self._cache_locations.pop(rid, None)
            self._total_seq_lens.pop(rid, None)
        else:
            self._cache_locations.clear()
            self._total_seq_lens.clear()

    def _create_forward_batch(
        self,
        input_ids: jax.Array,
        rid: str,
        T_groups: int,
        actual_T_groups: int,
        is_prefill: bool,
    ) -> ForwardBatch:
        B = 1

        if is_prefill:
            padded_input_groups = self._find_padding_size(actual_T_groups, self.input_group_paddings)

            max_gen_groups = self.generation_group_paddings[-1]
            total_cache_groups = padded_input_groups + max_gen_groups

            all_cache_loc = self.backbone_worker.model_runner.token_to_kv_pool_allocator.alloc(
                total_cache_groups * B
            )
            if all_cache_loc is None:
                raise RuntimeError(f"Failed to allocate {total_cache_groups} KV cache slots")

            self._cache_locations[rid] = all_cache_loc
            self._total_seq_lens[rid] = actual_T_groups

            out_cache_loc = all_cache_loc[:actual_T_groups]
            current_seq_len = actual_T_groups

            positions = jnp.arange(T_groups, dtype=jnp.int32)
            forward_mode = ForwardMode.EXTEND
            extend_seq_lens = jnp.array([actual_T_groups] * B, dtype=jnp.int32)
            extend_prefix_lens = jnp.zeros(B, dtype=jnp.int32)

        else:
            all_cache_loc = self._cache_locations[rid]
            existing_seq_len = self._total_seq_lens[rid]

            out_cache_loc = all_cache_loc[existing_seq_len:existing_seq_len + 1]
            current_seq_len = existing_seq_len + 1

            self._total_seq_lens[rid] = current_seq_len

            positions = jnp.array([existing_seq_len], dtype=jnp.int32)
            forward_mode = ForwardMode.DECODE
            extend_seq_lens = None
            extend_prefix_lens = None

        req_pool_indices = np.arange(B, dtype=np.int32)
        seq_lens_arr = np.array([current_seq_len] * B, dtype=np.int32)

        attention_mask = None
        if is_prefill and actual_T_groups < T_groups:
            attention_mask = jnp.zeros(T_groups, dtype=jnp.bool_)
            attention_mask = attention_mask.at[:actual_T_groups].set(True)
            logger.debug(f"Created attention mask: {actual_T_groups}/{T_groups} valid groups")

        return ForwardBatch(
            bid=0,
            forward_mode=forward_mode,
            batch_size=B,
            input_ids=None,
            positions=positions,
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array(seq_lens_arr),
            out_cache_loc=jnp.array(out_cache_loc),
            cache_loc=jnp.array(all_cache_loc),
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            attn_backend=self.backbone_worker.model_runner.attn_backend,
            attention_mask=attention_mask,
        )

    def _create_logits_metadata(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool = True,
    ) -> LogitsMetadata:
        if is_prefill:
            return LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                extend_seq_lens=jnp.array([seq_len] * batch_size, dtype=jnp.int32),
                extend_seq_lens_cpu=[seq_len] * batch_size,
                top_logprobs_nums=[0] * batch_size,
            )
        else:
            return LogitsMetadata(
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                top_logprobs_nums=[0] * batch_size,
            )

    def _prepare_batch(self, req: Req) -> tuple[jax.Array, ForwardBatch, LogitsMetadata]:
        input_ids = req.input_ids
        rid = req.rid

        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if not jnp.issubdtype(input_ids.dtype, jnp.integer):
            input_ids = input_ids.astype(jnp.int32)

        if input_ids.ndim == 2:
            input_ids = input_ids[None, :, :]

        B, _, seq_len = input_ids.shape
        T_groups = seq_len // MIMO_GROUP_SIZE
        actual_T_groups = T_groups

        existing_seq_len = self._total_seq_lens.get(rid, 0)
        is_prefill = existing_seq_len == 0

        if is_prefill:
            padded_groups = self._find_padding_size(T_groups, self.input_group_paddings)

            if padded_groups > T_groups:
                pad_len = (padded_groups - T_groups) * MIMO_GROUP_SIZE
                input_ids = jnp.pad(input_ids, ((0, 0), (0, 0), (0, pad_len)), constant_values=0)
                T_groups = padded_groups
                logger.debug(f"Padded from {actual_T_groups} to {padded_groups} groups")

        forward_batch = self._create_forward_batch(input_ids, rid, T_groups, actual_T_groups, is_prefill)

        logits_seq_len = actual_T_groups if is_prefill else 1
        logits_metadata = self._create_logits_metadata(B, logits_seq_len, is_prefill)

        return input_ids, forward_batch, logits_metadata

    def run_backbone_batch(self, batch: list[Req]):
        """Run the backbone forward pass for a batch of requests.

        Internal loop generates all tokens until EOS, avoiding cross-process overhead.
        Only sends result once when generation is complete.

        Steps per iteration:
        1. Forward through main transformer to get text logits and local hidden states.
        2. Sample text token from model output.
        3. If model outputs <|empty|>, generate audio tokens via patch decoder.
        4. Check if generation is finished (hit EOS token).
        5. If not finished, build next step input and continue loop.
        """
        for req in batch:
            sampler_config = self._get_sampler_config(req)
            accumulated_tokens = []
            step_count = 0

            if getattr(req, "is_prefill", True):
                self._reset_cache_state(req.rid)

            logger.info("Starting generation loop for rid=%s", req.rid)

            while True:
                input_ids, forward_batch, logits_metadata = self._prepare_batch(req)

                (text_logits_output, local_hidden_states, _), _ = self.backbone_worker.forward(
                    input_ids, forward_batch, logits_metadata
                )

                next_token_logits = text_logits_output.next_token_logits
                logits_np = np.array(jax.device_get(next_token_logits))

                if logits_np.ndim == 3:
                    logits_np = logits_np[:, -1, :]

                if sampler_config.do_sample:
                    logits = logits_np / max(sampler_config.temperature, 1e-5)
                    logits = logits - np.max(logits, axis=-1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= np.sum(probs, axis=-1, keepdims=True)
                    text_token_id = int(np.random.choice(probs.shape[-1], p=probs[0]))
                else:
                    text_token_id = int(np.argmax(logits_np, axis=-1)[0])

                accumulated_tokens.append(text_token_id)
                logger.debug("Step %d: generated token %d", step_count, text_token_id)

                if text_token_id == MIMO_EMPTY_IDX:
                    self.rng_key, subkey = jax.random.split(self.rng_key)
                    audio_tokens, _ = self.backbone_worker.patch_decode(
                        local_hidden_states, subkey, sampler_config
                    )
                    req.generated_audio_tokens = jax.device_get(audio_tokens)
                else:
                    req.generated_audio_tokens = None

                if text_token_id in self.eos_token_ids:
                    logger.info("Generation finished for rid=%s at step %d (EOS)", req.rid, step_count)
                    req.is_finished = True
                    break

                current_seq_len = self._total_seq_lens.get(req.rid, 0)
                allocated_cache_size = len(self._cache_locations.get(req.rid, []))
                if current_seq_len + 1 >= allocated_cache_size:
                    logger.warning(
                        "Generation reached cache limit for rid=%s (used %d/%d slots). "
                        "Consider increasing generation_group_paddings.",
                        req.rid, current_seq_len + 1, allocated_cache_size
                    )
                    req.is_finished = True
                    break

                req.generated_text_tokens = np.array([text_token_id], dtype=np.int32)
                try:
                    req.input_ids = req.build_next_step_input()
                    req.is_prefill = False
                    step_count += 1
                except Exception as e:
                    logger.error("Failed to build next step input for rid=%s: %s", req.rid, e)
                    req.is_finished = True
                    break

            req.generated_text_tokens = np.array(accumulated_tokens, dtype=np.int32)

            req.input_ids = None
            req.audio_codes = None
            req.backbone_cache = None

            logger.info(
                "Completed generation for rid=%s: %d tokens, finished=%s",
                req.rid, len(accumulated_tokens), req.is_finished
            )

            self._comm_backend.send_pyobj(req)

    def _get_sampler_config(self, req: Req) -> MiMoSamplerConfig:
        """Get sampler configuration from request."""
        return MiMoSamplerConfig(
            do_sample=getattr(req, "do_sample", False),
            temperature=getattr(req, "temperature", 1.0),
            top_k=getattr(req, "top_k", 0),
            top_p=getattr(req, "top_p", 1.0),
        )
