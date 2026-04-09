import logging

import jax.numpy as jnp

from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.encoder.encoder_model_runner import EncoderModelRunner
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EncoderModelWorker:
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh=None,
        model_class: str | list[str] = None,
        stage_sub_dir: str = None,
        tokenizer: str = None,
    ):
        self.mesh = mesh
        self.model_runner = EncoderModelRunner(
            server_args, mesh, model_class=model_class, stage_sub_dir=stage_sub_dir, tokenizer=tokenizer
        )

    def run_precompile(self):
        """Precompile encoder JIT by running a dummy forward pass."""
        import time
        start_time = time.perf_counter()
        for i, spec in enumerate(self.model_runner.encoder_specs):
            max_length = spec.max_length or 512
            logger.info(
                '[ENCODER] Precompiling encoder %d/%d (%s), max_length=%d',
                i + 1, len(self.model_runner.encoder_specs),
                spec.model_class.__name__, max_length,
            )
            dummy_input_ids = jnp.ones((1, max_length), dtype=jnp.int32)
            dummy_attention_mask = jnp.ones((1, max_length), dtype=jnp.int32)
            spec.jitted_forward(dummy_input_ids, dummy_attention_mask)
        elapsed = time.perf_counter() - start_time
        logger.info('[ENCODER] Precompile finished in %.0f secs', elapsed)

    def forward(self, batch: Req):
        # Implement the encoder model inference logic here
        # return batch
        return self.model_runner.forward(batch)