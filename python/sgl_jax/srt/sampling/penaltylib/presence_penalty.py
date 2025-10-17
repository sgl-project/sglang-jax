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
        return any(req.sampling_params.presence_penalty != 0.0 for req in self.orchestrator.reqs())

    def _prepare(self):
        # Only keep the presence penalty values, not the large penalty array
        presence_penalties = np.array(
            [req.sampling_params.presence_penalty for req in self.orchestrator.reqs()],
            dtype=np.float32,
        )
        self.presence_penalties = np.expand_dims(presence_penalties, axis=1)

        # Track token presence with a lightweight boolean array
        self.token_presence = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=bool,
        )

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        batch_indices = np.arange(len(output_ids))
        self.token_presence[batch_indices, output_ids] = True

    def compute_penalty(self) -> np.ndarray:
        """
        Compute and return the presence penalty array.

        Returns:
            np.ndarray: The presence penalty values for all tokens
        """
        return self.token_presence.astype(np.float32) * (-self.presence_penalties)

    def _filter(self, keep_indices: np.ndarray):
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.token_presence = self.token_presence[keep_indices]

    def _merge(self, their: "BatchedPresencePenalizer"):
        self.presence_penalties = np.concatenate(
            [self.presence_penalties, their.presence_penalties], axis=0
        )
        self.token_presence = np.concatenate([self.token_presence, their.token_presence], axis=0)
