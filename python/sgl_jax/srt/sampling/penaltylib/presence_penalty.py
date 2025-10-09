import numpy as np

from sgl_jax.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedPresencePenalizer(_BatchedPenalizer):
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_presence_penalties = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=np.float32,
        )

        presence_penalties = np.array(
            [req.sampling_params.presence_penalty for req in self.orchestrator.reqs()],
            dtype=np.float32,
        )
        self.presence_penalties = np.expand_dims(presence_penalties, axis=1)

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        self.cumulated_presence_penalties[np.arange(len(output_ids)), output_ids] = (
            self.presence_penalties.squeeze(axis=1)
        )

    def _apply(self, logits: np.ndarray) -> np.ndarray:
        return logits - self.cumulated_presence_penalties

    def _filter(self, keep_indices: np.ndarray):
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):
        self.presence_penalties = np.concatenate(
            [self.presence_penalties, their.presence_penalties], axis=0
        )
        self.cumulated_presence_penalties = np.concatenate(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            axis=0,
        )
