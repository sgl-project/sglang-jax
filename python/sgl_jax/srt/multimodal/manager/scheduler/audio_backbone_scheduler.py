"""Audio Backbone Scheduler for MiMo Audio LLM inference."""

import logging

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import (
    MIMO_EMPTY_IDX,
    Req,
)
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_worker import (
    AudioBackboneModelWorker,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)

# EOS token IDs for stopping generation
MIMO_EOS_TOKENS = {151672, 151643, 151645, 151671}  # EOT, EOS, IM_END, EOSTM


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

        # Random key for sampling
        self.rng_key = jax.random.PRNGKey(42)

        # Create worker within mesh context so JIT functions are traced with mesh available
        with self.mesh:
            self.backbone_worker = AudioBackboneModelWorker(
                model_class=model_class, mesh=self.mesh, server_args=server_args
            )

    def event_loop_normal(self):
        """Main blocking loop for processing requests.

        Repeatedly polls the communication_backend for requests, processes them
        through the backbone model, and sends results to the next stage.
        """
        # Set mesh context for all JAX operations (required for shard_map, KV cache updates, etc.)
        with self.mesh:
            while True:
                reqs = self._comm_backend.recv_requests()
                if len(reqs) > 0:
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

    def preprocess(self, req: Req):
        """Apply preprocessing to a single Req.

        Moves input_ids to device with proper sharding.
        """
        sharding = NamedSharding(self.mesh, PartitionSpec())
        if req.input_ids is not None:
            req.input_ids = device_array(req.input_ids, sharding=sharding)
        if req.audio_codes is not None:
            req.audio_codes = device_array(req.audio_codes, sharding=sharding)

    def run_backbone_batch(self, batch: list[Req]):
        """Run the backbone forward pass for a batch of requests.

        Unified logic for all audio modes:
        1. Forward through main transformer to get text logits and local hidden states.
        2. Sample text token from model output (let model decide).
        3. If model outputs <|empty|>, generate audio tokens via patch decoder.
        4. Otherwise, no audio tokens needed.
        5. Check if generation is finished (hit EOS token).
        """
        for req in batch:
            sampler_config = self._get_sampler_config(req)

            # Check if this is start of a new request (need to reset cache)
            if getattr(req, "is_prefill", True):
                self.backbone_worker.reset_cache_state(req.rid)

                # Dump first entry to stage1 for debugging
                self._dump_stage1_input(req)

            # Forward through main transformer
            (text_logits_output, local_hidden_states, _), _ = self.backbone_worker.forward(
                req
            )

            # Extract logits and sample text token
            next_token_logits = text_logits_output.next_token_logits
            logits_np = np.array(jax.device_get(next_token_logits))

            # Remove seq_len dim if present [B, 1, V] -> [B, V]
            if logits_np.ndim == 3:
                logits_np = logits_np[:, -1, :]

            # === DEBUG: Check for NaN/Inf ===
            step_num = self.backbone_worker._total_seq_lens.get(req.rid, 0)
            nan_count = np.sum(np.isnan(logits_np))
            inf_count = np.sum(np.isinf(logits_np))
            if nan_count > 0 or inf_count > 0:
                logger.error(
                    "!!! NaN/Inf detected at step=%d, rid=%s: nan_count=%d, inf_count=%d",
                    step_num, req.rid, nan_count, inf_count
                )
                # Check local_hidden_states
                local_hs_np = np.array(jax.device_get(local_hidden_states))
                logger.error(
                    "  local_hidden_states: nan=%d, inf=%d, min=%.4f, max=%.4f",
                    np.sum(np.isnan(local_hs_np)),
                    np.sum(np.isinf(local_hs_np)),
                    np.nanmin(local_hs_np),
                    np.nanmax(local_hs_np),
                )
                # Check input_ids
                if req.input_ids is not None:
                    input_np = np.array(jax.device_get(req.input_ids))
                    logger.error("  input_ids shape=%s, min=%d, max=%d", input_np.shape, input_np.min(), input_np.max())
            else:
                logger.info(
                    "Step=%d: logits OK, min=%.4f, max=%.4f, mean=%.4f",
                    step_num, logits_np.min(), logits_np.max(), logits_np.mean()
                )
            # === END DEBUG ===

            # Debug: show top-10 predicted tokens
            top_k = 10
            top_indices = np.argsort(logits_np[0])[-top_k:][::-1]
            top_logits = logits_np[0][top_indices]
            logger.info("Top-%d logits: %s", top_k, list(zip(top_indices.tolist(), top_logits.tolist())))



            # Sample token - let the model decide what to generate
            if sampler_config.do_sample:
                logits = logits_np / max(sampler_config.temperature, 1e-5)
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                probs = np.exp(logits)
                probs /= np.sum(probs, axis=-1, keepdims=True)
                text_token_id = int(np.random.choice(probs.shape[-1], p=probs[0]))
            else:
                text_token_id = int(np.argmax(logits_np, axis=-1)[0])

            logger.info(
                "Backbone generated token: %d for rid=%s, audio_mode=%s",
                text_token_id,
                req.rid,
                req.audio_mode,
            )

            # Store generated text token
            req.generated_text_tokens = np.array([text_token_id], dtype=np.int32)

            # Check if model wants to generate audio (output <|empty|>)
            if text_token_id == MIMO_EMPTY_IDX:
                # Check for NaN in local_hidden_states before patch_decode
                local_hs_np = np.array(jax.device_get(local_hidden_states))
                if np.any(np.isnan(local_hs_np)):
                    logger.error(
                        "NaN in local_hidden_states BEFORE patch_decode at step=%d, rid=%s. "
                        "Skipping audio generation.",
                        step_num, req.rid
                    )
                    req.generated_audio_tokens = None
                else:
                    # Model decided to generate audio
                    self.rng_key, subkey = jax.random.split(self.rng_key)
                    audio_tokens, _ = self.backbone_worker.patch_decode(
                        local_hidden_states, subkey, sampler_config
                    )
                    audio_tokens_np = jax.device_get(audio_tokens)

                    # Validate generated audio tokens
                    max_token = np.max(audio_tokens_np)
                    if max_token > 1024:  # Max possible valid token
                        logger.error(
                            "patch_decode produced invalid tokens: max=%d at step=%d, rid=%s",
                            max_token, step_num, req.rid
                        )
                    req.generated_audio_tokens = audio_tokens_np
            else:
                # Model generated text token, no audio needed
                req.generated_audio_tokens = None

            # Check if generation is finished (hit EOS)
            req.is_finished = text_token_id in MIMO_EOS_TOKENS

            # Mark as decode phase for next iteration
            req.is_prefill = False

            # Clear inputs to free memory and avoid JAX arrays in pickle
            req.input_ids = None
            req.audio_codes = None
            req.backbone_cache = None

            self._comm_backend.send_pyobj(req)

    def _get_sampler_config(self, req: Req) -> MiMoSamplerConfig:
        """Get sampler configuration from request."""
        return MiMoSamplerConfig(
            do_sample=getattr(req, "do_sample", False),
            temperature=getattr(req, "temperature", 1.0),
            top_k=getattr(req, "top_k", 0),
            top_p=getattr(req, "top_p", 1.0),
        )

    def _dump_stage1_input(self, req: Req):
        """Dump stage1 input data for debugging."""
        import os
        import pickle
        from datetime import datetime

        dump_dir = "/tmp/stage1_dumps"
        os.makedirs(dump_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = os.path.join(dump_dir, f"stage1_input_{req.rid}_{timestamp}.pkl")

        dump_data = {
            "rid": req.rid,
            "audio_mode": req.audio_mode,
            "input_ids": jax.device_get(req.input_ids) if req.input_ids is not None else None,
            "audio_codes": jax.device_get(req.audio_codes) if req.audio_codes is not None else None,
            "prompt_input_ids": getattr(req, "prompt_input_ids", None),
            "text_input_ids": getattr(req, "text_input_ids", None),
            "is_prefill": getattr(req, "is_prefill", None),
        }

        with open(dump_file, "wb") as f:
            pickle.dump(dump_data, f)

        logger.info("=" * 60)
        logger.info("Stage1 Input Dump saved to: %s", dump_file)
        logger.info("  rid: %s", req.rid)
        logger.info("  audio_mode: %s", req.audio_mode)
        if dump_data["input_ids"] is not None:
            logger.info("  input_ids shape: %s", dump_data["input_ids"].shape)
            logger.info("  input_ids dtype: %s", dump_data["input_ids"].dtype)
        if dump_data["audio_codes"] is not None:
            logger.info("  audio_codes shape: %s", dump_data["audio_codes"].shape)
            logger.info("  audio_codes dtype: %s", dump_data["audio_codes"].dtype)
        logger.info("=" * 60)
