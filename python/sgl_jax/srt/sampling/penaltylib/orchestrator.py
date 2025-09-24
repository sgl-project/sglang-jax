from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import numpy as np

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type["_BatchedPenalizer"]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        # No longer need internal penalty array management -
        # work directly on the provided linear_penalty array

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(self, output_ids: np.ndarray):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (jax.Array): The output tokens.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self) -> np.ndarray | None:
        """
        Apply the penalizers and return the penalty array.
        Optimized to avoid np.zeros() allocation by computing penalties on-demand.

        Returns:
            np.ndarray | None: The computed penalty array, or None if no penalties
        """
        # Import penalty classes for ordering
        from sgl_jax.srt.sampling.penaltylib.frequency_penalty import (
            BatchedFrequencyPenalizer,
        )
        from sgl_jax.srt.sampling.penaltylib.min_new_tokens import (
            BatchedMinNewTokensPenalizer,
        )
        from sgl_jax.srt.sampling.penaltylib.presence_penalty import (
            BatchedPresencePenalizer,
        )

        # Get active penalizers
        active_penalizers = [
            (type(p), p) for p in self.penalizers.values() if p.is_prepared()
        ]

        if len(active_penalizers) == 0:
            return None
        elif len(active_penalizers) == 1:
            # Single penalty optimization - return directly to avoid allocation
            _, penalizer = active_penalizers[0]
            return penalizer.compute_penalty()
        else:
            # Multiple penalties - compute in fixed order: presence -> frequency -> min_new_tokens
            penalty_order = [
                BatchedPresencePenalizer,
                BatchedFrequencyPenalizer,
                BatchedMinNewTokensPenalizer,
            ]
            result = None

            for penalty_type in penalty_order:
                if (
                    penalty_type in self.penalizers
                    and self.penalizers[penalty_type].is_prepared()
                ):
                    penalty_values = self.penalizers[penalty_type].compute_penalty()
                    if result is None:
                        result = penalty_values.copy()
                    else:
                        result += penalty_values

            return result

    def filter(self, keep_indices: np.ndarray):
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (np.ndarray): Array of indices to keep in the batch.
        """
        if not self.is_required:
            return

        if len(keep_indices) == 0:
            self.is_required = False
            for penalizer in self.penalizers.values():
                penalizer.teardown()
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    def merge(self, their: "BatchedPenalizerOrchestrator"):
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """
        if not self.is_required and not their.is_required:
            return

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):
    """
    An abstract class for a batched penalizer.
    """

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._is_prepared = False

    def cumulate_output_tokens(self, output_ids: np.ndarray):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids)

    def filter(self, keep_indices: np.ndarray):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: "_BatchedPenalizer"):
        if not self._is_prepared and not their._is_prepared:
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """
        Check if the penalizer is required to be prepared.
        """
        pass

    @abc.abstractmethod
    def _prepare(self):
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        """
        Cumulate the output tokens.
        Orchestrator will call this function to feed the output tokens to the penalizer.
        """
        pass

    @abc.abstractmethod
    def _filter(self, keep_indices: np.ndarray):
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """
        pass

    @abc.abstractmethod
    def _merge(self, their: "_BatchedPenalizer"):
        """
        Merge the penalizer with another penalizer.
        """
        pass
