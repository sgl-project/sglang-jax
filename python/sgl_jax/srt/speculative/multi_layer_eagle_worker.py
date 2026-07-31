"""NEXTN/MTP orchestrator using one fused topk=1 runner per prediction layer."""

from __future__ import annotations

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker


class MultiLayerEAGLEWorker(EAGLEWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        super().__init__(
            server_args,
            target_worker,
            draft_worker=MultiLayerDraftWorker(server_args, target_worker),
        )
