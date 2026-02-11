"""Audio Backbone Scheduler for MiMo Audio LLM inference."""

import logging

import jax
import jax.sharding
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import MIMO_EMPTY_IDX, Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_worker import (
    AudioBackboneModelWorker,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)

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

        1. Forward through main transformer to get text logits and local hidden states.
        2. Sample text token from model output.
        3. If model outputs <|empty|>, generate audio tokens via patch decoder.
        4. Check if generation is finished (hit EOS token).
        """
        for req in batch:
            sampler_config = self._get_sampler_config(req)

            if getattr(req, "is_prefill", True):
                self.backbone_worker.reset_cache_state(req.rid)

            # Forward through main transformer
            (text_logits_output, local_hidden_states, _), _ = self.backbone_worker.forward(req)

            # Extract logits and sample text token
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

            req.generated_text_tokens = np.array([text_token_id], dtype=np.int32)

            # Generate audio tokens if model outputs <|empty|>
            if text_token_id == MIMO_EMPTY_IDX:
                self.rng_key, subkey = jax.random.split(self.rng_key)
                audio_tokens, _ = self.backbone_worker.patch_decode(
                    local_hidden_states, subkey, sampler_config
                )
                req.generated_audio_tokens = jax.device_get(audio_tokens)
            else:
                req.generated_audio_tokens = None

            req.is_finished = text_token_id in MIMO_EOS_TOKENS
            req.is_prefill = False

            # Clear inputs to free memory
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
