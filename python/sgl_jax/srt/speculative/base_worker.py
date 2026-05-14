from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.managers.tp_worker import ModelWorker


class BaseDraftWorker(ABC):
    """Draft model worker interface for speculative decoding.

    Concrete implementations hold the draft model runner and own all
    draft-specific logic (multi-step decode, tree building, extend).
    Standard EAGLE uses ``EagleDraftWorker``; MTP will use
    ``MultiLayerDraftWorker``.
    """

    @abstractmethod
    def draft(self):
        pass


class BaseSpecWorker(ABC):
    """Speculative decode orchestrator.

    Owns a ``target_worker`` (the full model) and a ``draft_worker``
    (the draft/MTP model).  Concrete subclasses implement the main
    entry point and connect their specific draft/verify logic.
    """

    @property
    @abstractmethod
    def target_worker(self) -> ModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass
