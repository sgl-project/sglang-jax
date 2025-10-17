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
        padded = np.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
    else:
        padded = np.full((max_len, len(sequences)), padding_value, dtype=sequences[0].dtype)
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
        return any(req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs())

    def _prepare(self):
        min_new_tokens_list = [
            req.sampling_params.min_new_tokens for req in self.orchestrator.reqs()
        ]
        self.min_new_tokens = np.expand_dims(np.array(min_new_tokens_list, dtype=np.int32), axis=1)

        # Store stop token sequences without creating large penalty array
        self.stop_token_sequences = []
        for req in self.orchestrator.reqs():
            stop_tokens = set()
            if req.sampling_params.stop_token_ids:
                stop_tokens.update(req.sampling_params.stop_token_ids)
            if req.tokenizer.additional_stop_token_ids:
                stop_tokens.update(req.tokenizer.additional_stop_token_ids)
            if req.tokenizer.eos_token_id is not None:
                stop_tokens.add(req.tokenizer.eos_token_id)

            self.stop_token_sequences.append(np.array(list(stop_tokens), dtype=np.int64))

        self.len_output_tokens = np.zeros(
            (len(self.orchestrator.reqs()), 1),
            dtype=np.int32,
        )

    def _cumulate_output_tokens(self, output_ids: np.ndarray):
        # Simple numpy increment for CPU operations
        self.len_output_tokens = self.len_output_tokens + 1

    def compute_penalty(self) -> np.ndarray:
        """
        Compute and return the min new tokens penalty array.
        Note: Returns negative values since min_new_tokens penalties are ADDED to logits,
        but orchestrator applies all penalties with subtraction.

        Returns:
            np.ndarray: The min new tokens penalty values (negative for orchestrator)
        """
        # Create mask for requests that haven't reached min_new_tokens
        mask = self.len_output_tokens < self.min_new_tokens

        # Create stop token penalties on-demand
        stop_token_penalties = self._create_stop_token_penalties()

        mask_expanded = np.broadcast_to(mask, stop_token_penalties.shape)
        penalty_values = np.where(mask_expanded, stop_token_penalties, 0.0)

        return penalty_values

    def _create_stop_token_penalties(self):
        # Create stop token penalties on-demand to avoid storing large arrays
        stop_token_penalties = np.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=np.float32,
        )

        for i, stop_tokens in enumerate(self.stop_token_sequences):
            if len(stop_tokens) > 0:
                valid_tokens = stop_tokens[stop_tokens < self.orchestrator.vocab_size]
                stop_token_penalties[i, valid_tokens] = float("-inf")

        return stop_token_penalties

    def _filter(self, keep_indices: np.ndarray):
        self.min_new_tokens = self.min_new_tokens[keep_indices]
        self.stop_token_sequences = [self.stop_token_sequences[i] for i in keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]

    def _merge(self, their: "BatchedMinNewTokensPenalizer"):
        self.min_new_tokens = np.concatenate([self.min_new_tokens, their.min_new_tokens], axis=0)
        self.stop_token_sequences.extend(their.stop_token_sequences)
        self.len_output_tokens = np.concatenate(
            [self.len_output_tokens, their.len_output_tokens], axis=0
        )
