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
        # Only keep the frequency penalty values, not the large penalty array
        frequency_penalties = np.array(
            [req.sampling_params.frequency_penalty for req in self.orchestrator.reqs()],
            dtype=np.float32,
        )
        self.frequency_penalties = np.expand_dims(frequency_penalties, axis=1)

        # Track token frequencies with a lightweight structure
        self.token_frequencies = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=np.int32,
        )

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        batch_indices = np.arange(len(output_ids))
        self.token_frequencies[batch_indices, output_ids] += 1

    def compute_penalty(self) -> np.ndarray:
        """
        Compute and return the frequency penalty array.

        Returns:
            np.ndarray: The frequency penalty values for all tokens
        """
        return self.token_frequencies.astype(np.float32) * (-self.frequency_penalties)

    def _filter(self, keep_indices: np.ndarray):
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.token_frequencies = self.token_frequencies[keep_indices]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.frequency_penalties = np.concatenate(
            [self.frequency_penalties, their.frequency_penalties], axis=0
        )
        self.token_frequencies = np.concatenate(
            [self.token_frequencies, their.token_frequencies], axis=0
        )
