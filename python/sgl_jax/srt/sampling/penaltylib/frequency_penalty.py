import numpy as np

from sgl_jax.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_frequency_penalties = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=np.float32,
        )

        frequency_penalties = np.array(
            [req.sampling_params.frequency_penalty for req in self.orchestrator.reqs()],
            dtype=np.float32,
        )
        self.frequency_penalties = np.expand_dims(frequency_penalties, axis=1)

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        # Convert JAX array to numpy for CPU operations
        # Use atleast_1d to ensure it's always at least 1-dimensional
        output_ids_np = np.atleast_1d(np.array(output_ids).squeeze())

        # Use numpy operations for CPU-based accumulation
        batch_indices = np.arange(len(output_ids_np))
        self.cumulated_frequency_penalties[
            batch_indices, output_ids_np
        ] += self.frequency_penalties.squeeze(axis=1)

    def _apply(self, logits: np.ndarray) -> np.ndarray:
        # Convert numpy to JAX array for GPU operations
        cumulated_penalties_jax = np.array(self.cumulated_frequency_penalties)
        return logits - cumulated_penalties_jax

    def _filter(self, keep_indices: np.ndarray):
        # Convert JAX array to numpy for CPU operations
        keep_indices_np = np.array(keep_indices)
        self.frequency_penalties = self.frequency_penalties[keep_indices_np]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices_np
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.frequency_penalties = np.concatenate(
            [self.frequency_penalties, their.frequency_penalties], axis=0
        )
        self.cumulated_frequency_penalties = np.concatenate(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            axis=0,
        )
