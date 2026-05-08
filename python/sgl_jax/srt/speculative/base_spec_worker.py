from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.managers.tp_worker import ModelWorker
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft(self, model_worker_batch: ModelWorkerBatch) -> EagleVerifyInput:
        raise NotImplementedError

    @abstractmethod
    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden_states,
        next_token_ids,
    ):
        raise NotImplementedError

    @abstractmethod
    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        raise NotImplementedError


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> ModelWorker:
        raise NotImplementedError

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        raise NotImplementedError

    @abstractmethod
    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        cur_allocate_lens=None,
    ) -> GenerationBatchResult:
        raise NotImplementedError
