import logging

import jax
import jax.sharding
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_model_worker import AudioModelWorker
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


class AudioScheduler:
    """Scheduler for audio tokenizer model inference within the multimodal pipeline.

    Responsibilities:
    - Receive batched requests via a communication backend and prepare inputs.
    - Move input arrays onto JAX devices using the provided `mesh` and a
      `NamedSharding`/`PartitionSpec` before forwarding to the audio worker.
    - Run the audio encode/decode forward pass and return/send outputs via the
      communication backend.
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend,
        mesh: jax.sharding.Mesh,
        model_class,
        **kwargs,
    ):
        """Initialize the AudioScheduler.

        Args:
            server_args: Multimodal server configuration.
            communication_backend: Backend used to receive/send requests.
            mesh: JAX device mesh used for sharding inputs/outputs.
            model_class: The audio model class; used to build `AudioModelWorker`.
        """

        self._comm_backend = communication_backend
        self.mesh = mesh
        self.audio_worker = AudioModelWorker(
            model_class=model_class, mesh=self.mesh, server_args=server_args
        )
        self.server_args = server_args
        self.aborted_rids: set[str] = set()

    def event_loop_normal(self):
        """Main blocking loop used in non-async environments.

        Repeatedly polls the `communication_backend` for requests, shards inputs
        onto `self.mesh` with a `NamedSharding(PartitionSpec())`, and then
        processes the batch via `run_audio_batch`. AbortReq messages are processed
        to track aborted request IDs, and any Req whose rid matches an aborted ID
        is skipped.
        """

        while True:
            reqs = self._comm_backend.recv_requests()
            if len(reqs) > 0:
                valid_reqs = []
                for req in reqs:
                    if isinstance(req, AbortReq):
                        logger.info("AudioScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, Req):
                        if req.rid in self.aborted_rids:
                            logger.info("AudioScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        self.preprocess(req)
                        valid_reqs.append(req)
                    else:
                        logger.warning(
                            "AudioScheduler received unknown request type: %s", type(req)
                        )

                if valid_reqs:
                    self.run_audio_batch(valid_reqs)

    def preprocess(self, req: Req):
        sharding = NamedSharding(self.mesh, PartitionSpec())
        if req.audio_mode in ("tts", "asr", "audio_understanding"):
            # Use mel_input (preprocessed in tokenizer) instead of audio_input
            if req.mel_input is not None:
                req.mel_input = device_array(req.mel_input, sharding=sharding)
            if req.mel_input_lens is not None:
                req.mel_input_lens = device_array(req.mel_input_lens, sharding=sharding)

    def run_audio_batch(self, batch: list[Req]):
        for req in batch:
            mode = req.audio_mode
            logger.info("AudioScheduler.run_audio_batch: mode=%s, rid=%s", mode, req.rid)

            if mode in ("tts", "asr", "audio_understanding"):
                if req.mel_input is not None:
                    output, _ = self.audio_worker.forward(
                        req, mode="encode", use_quantizer=req.use_quantizer, n_q=req.n_q
                    )
                    codes = jax.device_get(output.codes)
                    output_lens = jax.device_get(output.output_lengths)

                    # DEBUG: Check audio codes range and valid length
                    if codes is not None:
                        logger.info(
                            "AudioScheduler encode output: codes shape=%s, output_lens=%s",
                            codes.shape,
                            output_lens,
                        )

                        # IMPORTANT: Replace codes beyond valid length with empty IDs
                        # The encoder produces random codes for masked positions, we need to fix them
                        if output_lens is not None and len(output_lens) > 0:
                            valid_len = int(output_lens[0])
                            total_len = codes.shape[1]

                            if valid_len < total_len:
                                logger.info(
                                    "Fixing audio codes: valid_len=%d, total_len=%d, "
                                    "replacing positions %d-%d with empty IDs",
                                    valid_len, total_len, valid_len, total_len - 1
                                )
                                # Import empty IDs
                                from sgl_jax.srt.multimodal.manager.schedule_batch import MIMO_SPEECH_EMPTY_IDS

                                # Replace invalid positions with empty IDs for each channel
                                import numpy as np
                                codes = np.array(codes)  # Convert to numpy for easier manipulation
                                for ch in range(min(codes.shape[0], 8)):
                                    empty_id = MIMO_SPEECH_EMPTY_IDS[ch]
                                    codes[ch, valid_len:] = empty_id
                                    logger.info(
                                        "  Channel %d: replaced %d invalid codes with empty_id=%d",
                                        ch, total_len - valid_len, empty_id
                                    )

                    req.output = codes
                else:
                    logger.info(
                        "AudioScheduler skipping encode for mode=%s, rid=%s (no mel_input)",
                        mode,
                        req.rid,
                    )

            elif mode == "tts":
                # Pass-through for TTS at encoder stage (text input only)
                pass

            else:
                logger.warning("AudioScheduler received unknown or unhandled mode: %s", mode)

            # Clear inputs to free memory
            req.mel_input = None
            req.mel_input_lens = None
            req.codes = None
            self._comm_backend.send_pyobj(req)
