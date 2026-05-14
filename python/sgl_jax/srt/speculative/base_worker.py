from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from sgl_jax.srt.managers.tp_worker import ModelWorker


def replicate_to_mesh(
    mesh: jax.sharding.Mesh, *arrs: jax.Array
) -> tuple[jax.Array, ...] | jax.Array:
    """Replicate arrays across a mesh under explicit sharding.

    JIT outputs are typically vocab/data-sharded; spec-decode host orchestration
    (top_k, gather, build_tree) needs replicated arrays.
    """
    out = jax.device_put(arrs, NamedSharding(mesh, P()))
    return out[0] if len(out) == 1 else out


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
