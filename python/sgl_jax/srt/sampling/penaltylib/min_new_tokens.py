import numpy as np

from sgl_jax.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
)


def pad_sequence(sequences, batch_first=True, padding_value=0):
    """
    Numpy equivalent of torch.nn.utils.rnn.pad_sequence
    """
    max_len = max(len(seq) for seq in sequences)
    if batch_first:
        padded = np.full(
            (len(sequences), max_len), padding_value, dtype=sequences[0].dtype
        )
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
    else:
        padded = np.full(
            (max_len, len(sequences)), padding_value, dtype=sequences[0].dtype
        )
        for i, seq in enumerate(sequences):
            padded[: len(seq), i] = seq
    return padded


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self.orchestrator = orchestrator
        self._is_prepared = False

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        min_new_tokens_list = [
            req.sampling_params.min_new_tokens for req in self.orchestrator.reqs()
        ]
        self.min_new_tokens = np.expand_dims(
            np.array(min_new_tokens_list, dtype=np.int32), axis=1
        )

        # Prepare stop token sequences
        stop_token_sequences = []
        for req in self.orchestrator.reqs():
            stop_tokens = set()
            if req.sampling_params.stop_token_ids:
                stop_tokens.update(req.sampling_params.stop_token_ids)
            if req.tokenizer.additional_stop_token_ids:
                stop_tokens.update(req.tokenizer.additional_stop_token_ids)
            if req.tokenizer.eos_token_id is not None:
                stop_tokens.add(req.tokenizer.eos_token_id)

            stop_token_sequences.append(np.array(list(stop_tokens), dtype=np.int64))

        # Pad sequences
        padded_stop_token_ids = pad_sequence(
            stop_token_sequences,
            batch_first=True,
            padding_value=self.orchestrator.vocab_size,
        )

        # Create stop token penalties
        self.stop_token_penalties = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size + 1),
            dtype=np.float32,
        )

        # Use numpy operations to set penalties for stop tokens
        batch_size, seq_len = padded_stop_token_ids.shape
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = padded_stop_token_ids[i, j]
                if token_id < self.orchestrator.vocab_size:  # Valid token, not padding
                    self.stop_token_penalties[i, token_id] = float("-inf")

        # Remove the extra dimension used for padding
        self.stop_token_penalties = self.stop_token_penalties[
            :, : self.orchestrator.vocab_size
        ]

        self.len_output_tokens = np.zeros(
            (len(self.orchestrator.reqs()), 1),
            dtype=np.int32,
        )

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        # Simple numpy increment for CPU operations
        self.len_output_tokens = self.len_output_tokens + 1

    def _apply(self, logits: np.ndarray) -> np.ndarray:
        # Create mask for requests that haven't reached min_new_tokens
        mask = self.len_output_tokens < self.min_new_tokens
        mask_expanded = np.broadcast_to(mask, logits.shape)
        penalty_to_add = np.where(mask_expanded, self.stop_token_penalties, 0.0)

        logits = logits + penalty_to_add

    def _filter(self, keep_indices: np.ndarray):
        self.min_new_tokens = self.min_new_tokens[keep_indices]
        self.stop_token_penalties = self.stop_token_penalties[keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]

    def _merge(self, their: "BatchedMinNewTokensPenalizer"):
        self.min_new_tokens = np.concatenate(
            [self.min_new_tokens, their.min_new_tokens], axis=0
        )
        self.stop_token_penalties = np.concatenate(
            [self.stop_token_penalties, their.stop_token_penalties], axis=0
        )
        self.len_output_tokens = np.concatenate(
            [self.len_output_tokens, their.len_output_tokens], axis=0
        )
