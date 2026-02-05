import logging

import jax.sharding

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq, ProfileReq
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vit.vit_model_worker import VitModelWorker

logger = logging.getLogger(__name__)


class VitScheduler(SchedulerProfilerMixin):
    """Scheduler shell for ViT-stage feature extraction."""

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
        self.vit_worker = VitModelWorker(server_args, mesh=mesh, model_class=model_class)
        self.forward_ct = 0
        self.init_profier()
        self.aborted_rids: set[str] = set()

    def event_loop_normal(self):
        while True:
            reqs = self.communication_backend.recv_requests()
            if reqs is not None and len(reqs) > 0:
                for req in reqs:
                    if isinstance(req, AbortReq):
                        logger.info("VitScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, ProfileReq):
                        result = self.profile(req)
                        self.communication_backend.send_pyobj(result)
                    elif isinstance(req, Req):
                        if req.rid in self.aborted_rids:
                            logger.info("VitScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        self.run_vit_step(req)
                    else:
                        logger.warning("VitScheduler received unknown request type: %s", type(req))
            else:
                self.communication_backend.wait_for_new_requests(0.001)

    def run_vit_step(self, req: Req):
        """Placeholder: run ViT encoder and forward request to next stage."""
        self.vit_worker.forward(req, self.mesh)
        self.forward_ct += 1
        self._profile_batch_predicate(None)
        self.communication_backend.send_pyobj(req)
