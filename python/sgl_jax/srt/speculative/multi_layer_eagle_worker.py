"""MiMo-V2.5-Pro multi-layer MTP spec orchestrator (#1053 P1-4).

Thin wrapper over ``EAGLEWorker`` that swaps the draft worker for
``MultiLayerDraftWorker``. Orchestration (prefill/decode dispatch,
``verify()``, precompile) is identical to standard EAGLE — only the draft
side differs (N runners, layer→layer hidden chaining).
"""

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
