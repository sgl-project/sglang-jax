import logging

import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.embed.embed_model_worker import (
    EmbedModelWorker,
)

logger = logging.getLogger(__name__)


class EmbedScheduler:
    """Scheduler shell for embed-stage feature extraction."""

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh,
        communication_backend: CommunicationBackend,
        model_class,
        stage_sub_dir: str | None = None,
    ):
        self.communication_backend = communication_backend
        self.mesh = mesh
        self.embed_worker = EmbedModelWorker(server_args, mesh=mesh, model_class=model_class)
        self.aborted_rids: set[str] = set()

    def event_loop_normal(self):
        while True:
            reqs = self.communication_backend.recv_requests()
            if reqs is None:
                continue
            for req in reqs:
                if isinstance(req, AbortReq):
                    logger.info("EmbedScheduler received abort for rid=%s", req.rid)
                    self.aborted_rids.add(req.rid)
                elif isinstance(req, Req):
                    if req.rid in self.aborted_rids:
                        logger.info("EmbedScheduler skipping aborted request rid=%s", req.rid)
                        self.aborted_rids.discard(req.rid)
                        continue
                    self.run_embed_step(req)
                else:
                    logger.warning("EmbedScheduler received unknown request type: %s", type(req))

    def run_embed_step(self, req: Req):
        """Placeholder: run embed encoder and forward request to next stage."""
        self.embed_worker.forward(req, self.mesh)
        self.communication_backend.send_pyobj(req)
