import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req

logger = logging.getLogger(__name__)


class TextEncoderScheduler:
    """Scheduler for text encoder stages that run without KV cache.

    This scheduler loads a text encoder model (e.g. LTX2GemmaTextEncoder),
    runs it directly using simple_attention (no RadixAttention/KV cache),
    and forwards the encoded embeddings to the next pipeline stage.

    This is appropriate for models like the LTX-2 text encoder which only
    need a single prefill pass per request (no autoregressive decoding).
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh,
        communication_backend: CommunicationBackend,
        model_class,
        stage_sub_dir: str | None = None,
        precompile_params: dict | None = None,
    ):
        self.communication_backend = communication_backend
        self.mesh = mesh
        self.server_args = server_args
        self.model_class = model_class
        self.aborted_rids: set[str] = set()

        # Load the text encoder model
        self.model = self._load_model(server_args)
        logger.info("TextEncoderScheduler initialized with model %s", model_class.__name__)

    def _load_model(self, server_args):
        """Load and initialize the text encoder model.

        Uses nnx.eval_shape to create abstract parameters first (avoiding
        device placement issues), then loads real weights from checkpoints.
        """
        # Load HF config from the tokenizer path (Gemma) for model architecture
        tokenizer_path = server_args.tokenizer_path or server_args.model_path
        hf_config = AutoConfig.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        # For Gemma3 multimodal, the text config is under text_config
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        # Create model with abstract params (no actual tensor allocation)
        with jax.set_mesh(self.mesh):
            model = nnx.eval_shape(
                lambda: self.model_class(
                    config=hf_config,
                    mesh=self.mesh,
                    dtype=jnp.bfloat16,
                )
            )

        # Load weights from checkpoints (Gemma + LTX-2 connector)
        model_config = ModelConfig(model_path=tokenizer_path)
        model.load_weights(model_config)

        logger.info("Text encoder model loaded and weights initialized")
        return model

    def event_loop_normal(self):
        while True:
            reqs = self.communication_backend.recv_requests()
            if reqs is not None and len(reqs) > 0:
                for req in reqs:
                    if isinstance(req, AbortReq):
                        logger.info(
                            "TextEncoderScheduler received abort for rid=%s",
                            req.rid,
                        )
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, Req):
                        if req.rid in self.aborted_rids:
                            logger.info(
                                "TextEncoderScheduler skipping aborted rid=%s",
                                req.rid,
                            )
                            self.aborted_rids.discard(req.rid)
                            continue
                        self._run_text_encoder(req)
                    else:
                        logger.warning(
                            "TextEncoderScheduler received unknown type: %s",
                            type(req),
                        )
            else:
                self.communication_backend.wait_for_new_requests(0.001)

    def _run_text_encoder(self, req: Req):
        """Run the text encoder on positive and negative prompts, store results in req."""
        input_ids_pos = req.input_ids
        input_ids_neg = req.negative_input_ids

        if input_ids_pos is None:
            logger.warning("No input_ids for rid=%s, skipping text encoding", req.rid)
            self.communication_backend.send_pyobj(req)
            return

        if input_ids_neg is None:
            # Use empty sequence as negative if not provided
            input_ids_neg = []

        result = self.model.forward_no_cache(input_ids_pos, input_ids_neg)

        if len(result) == 4:
            # Audio-video text encoder returns (video_pos, video_neg, audio_pos, audio_neg)
            prompt_embeds, negative_prompt_embeds, audio_prompt_embeds, audio_negative_prompt_embeds = result
            req.audio_prompt_embeds = audio_prompt_embeds
            req.audio_negative_prompt_embeds = audio_negative_prompt_embeds
        else:
            prompt_embeds, negative_prompt_embeds = result

        req.prompt_embeds = prompt_embeds
        req.negative_prompt_embeds = negative_prompt_embeds

        logger.info(
            "Text encoder done for rid=%s: pos=%s, neg=%s, audio=%s",
            req.rid,
            prompt_embeds.shape,
            negative_prompt_embeds.shape,
            req.audio_prompt_embeds.shape if req.audio_prompt_embeds is not None else None,
        )
        self.communication_backend.send_pyobj(req)
