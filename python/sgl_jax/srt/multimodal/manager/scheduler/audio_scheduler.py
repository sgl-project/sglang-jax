import logging

import jax
import jax.sharding
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import MIMO_SPEECH_EMPTY_IDS, Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_model_worker import AudioModelWorker
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


class AudioScheduler:
    """Scheduler for audio tokenizer encode/decode within multimodal pipeline."""

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend,
        mesh: jax.sharding.Mesh,
        model_class,
        **kwargs,
    ):
        self._comm_backend = communication_backend
        self.mesh = mesh
        self.audio_worker = AudioModelWorker(
            model_class=model_class, mesh=self.mesh, server_args=server_args
        )
        self.server_args = server_args
        self.aborted_rids: set[str] = set()

    def event_loop_normal(self):
        while True:
            reqs = self._comm_backend.recv_requests()
            if reqs is not None and len(reqs) > 0:
                valid_reqs = []
                for req in reqs:
                    if isinstance(req, AbortReq):
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, Req):
                        if req.rid in self.aborted_rids:
                            self.aborted_rids.discard(req.rid)
                            continue
                        self.preprocess(req)
                        valid_reqs.append(req)
                    else:
                        logger.warning("Unknown request type: %s", type(req))

                if valid_reqs:
                    self.run_audio_batch(valid_reqs)
            else:
                self._comm_backend.wait_for_new_requests(0.001)

    def preprocess(self, req: Req):
        sharding = NamedSharding(self.mesh, PartitionSpec())
        if req.mel_input is not None:
            req.mel_input = device_array(req.mel_input, sharding=sharding)
        if req.mel_input_lens is not None:
            req.mel_input_lens = device_array(req.mel_input_lens, sharding=sharding)

    def run_audio_batch(self, batch: list[Req]):
        for req in batch:
            if req.audio_mode not in ("tts", "asr", "audio_understanding"):
                logger.warning("Unknown audio mode: %s", req.audio_mode)
                self._comm_backend.send_pyobj(req)
                continue

            if req.mel_input is None:
                self._comm_backend.send_pyobj(req)
                continue

            output, _ = self.audio_worker.forward(
                req, mode="encode", use_quantizer=req.use_quantizer, n_q=req.n_q
            )
            codes = jax.device_get(output.codes)
            output_lens = jax.device_get(output.output_lengths)

            if codes is not None and output_lens is not None and len(output_lens) > 0:
                valid_len = int(output_lens[0])
                total_len = codes.shape[1]

                if valid_len < total_len:
                    codes = np.array(codes)
                    for ch in range(min(codes.shape[0], 8)):
                        codes[ch, valid_len:] = MIMO_SPEECH_EMPTY_IDS[ch]

            req.output = codes
            req.mel_input = None
            req.mel_input_lens = None
            req.codes = None
            self._comm_backend.send_pyobj(req)
