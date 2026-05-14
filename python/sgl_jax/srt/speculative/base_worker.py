from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.model_executor.model_runner import ModelRunner


class BaseDraftWorker(ABC):
    """Draft model worker interface for speculative decoding.

    Concrete implementations hold the draft model runner and own all
    draft-specific logic (multi-step decode, tree building, extend).
    Standard EAGLE uses ``EagleDraftWorker``; MTP will use
    ``MultiLayerDraftWorker``.
    """

    @property
    @abstractmethod
    def draft_model_runner(self) -> ModelRunner: ...

    @abstractmethod
    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        """Run multi-step draft decode.

        Mutates ``model_worker_batch.spec_info`` from an
        ``EagleDraftInput`` to an ``EagleVerifyInput``.
        """

    @abstractmethod
    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        """Run draft model extend after target prefill.

        Populates ``model_worker_batch.spec_info`` with the draft state
        needed for the next decode round.
        """

    @abstractmethod
    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        """Run draft extend after target verify.

        Updates ``batch_output.next_draft_input`` with hidden_states,
        topk_p, topk_index for the next decode iteration.
        """


class BaseSpecWorker(ABC):
    """Speculative decode orchestrator.

    Owns a ``target_worker`` (the full model) and a ``draft_worker``
    (the draft/MTP model).  Subclasses implement the top-level
    ``forward_batch_speculative_generation`` entry point; shared logic
    such as ``verify`` lives here.
    """

    def __init__(self, server_args, target_worker, draft_worker: BaseDraftWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.draft_worker = draft_worker

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.mesh = target_worker.mesh

        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

    @abstractmethod
    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        """Main entry point called by the scheduler."""
