"""Detached speculative submissions for the single-owner overlap scheduler."""

import dataclasses
from concurrent.futures import Future
from copy import copy

import numpy as np

from sgl_jax.srt.speculative.overlap_utils import publish_spec_decode_new_seq_lens


@dataclasses.dataclass(frozen=True)
class SpeculativePlan:
    decode_relay: bool
    prefill_relay: bool
    legacy_eagle3_decode: bool = False


@dataclasses.dataclass(frozen=True)
class SpeculativeSubmission:
    batch: object
    plan: SpeculativePlan
    future: Future

    def wait(self):
        self.future.result()


def _snapshot(value):
    """Copy host containers; immutable JAX arrays keep their device references."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        snapshot = copy(value)
        # Some speculative metadata is attached dynamically between stages.
        for name, item in vars(value).items():
            object.__setattr__(snapshot, name, _snapshot(item))
        return snapshot
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


def snapshot_speculative_batch(batch):
    """Called on the owner, after any preceding stateful output is retired."""
    sampling = batch.sampling_info
    sampling.update_penalties()
    if sampling.grammars:
        sampling.update_grammar_vocab_mask()
    # Neither grammar objects nor the Req-backed penalizer can cross threads.
    sampling = dataclasses.replace(
        sampling, grammars=None, penalizer_orchestrator=None, sampling_info_done=None
    )
    batch = copy(batch)
    batch.sampling_info = sampling
    batch.launch_done = None
    # Allocation and prefix-cache operations remain on the owner thread.
    batch.tree_cache = None
    batch = _snapshot(batch)
    batch.spec_sampling_prepared = True
    return batch


def execute_speculative_batch(worker, batch, plan):
    """Device submission only: no ScheduleBatch or request mutation."""
    if plan.decode_relay:
        return worker.forward_batch_speculative_decode_overlap(batch)
    if plan.prefill_relay:
        return worker.forward_batch_speculative_prefill_overlap(batch), None
    # Generic verify mutates forward_mode to TARGET_VERIFY/DRAFT_EXTEND.
    is_decode = batch.forward_mode.is_decode()
    output = worker.forward_batch_speculative_generation(batch)
    new_seq_lens = (
        publish_spec_decode_new_seq_lens(output)
        if is_decode and not plan.legacy_eagle3_decode
        else None
    )
    return output, new_seq_lens


def pack_verified_tokens(verified_id, accept_lens, draft_token_num):
    """Convert accepted tree paths to the padded row layout used by retirement."""
    paths = np.asarray(verified_id).reshape(len(accept_lens), -1)
    tokens = np.zeros((len(accept_lens), draft_token_num), dtype=paths.dtype)
    width = min(paths.shape[1], draft_token_num)
    tokens[:, :width] = paths[:, :width]
    return tokens.reshape(-1)


def compact_speculative_state(state, batch):
    """Generic verify lengths use padded DP slots; persisted state uses real rows."""
    if state is None or batch.real_bs == batch.dp_size * batch.per_dp_bs_size:
        return state
    updates = {}
    for name in ("new_seq_lens", "accept_length", "accept_length_cpu"):
        value = getattr(state, name, None)
        if value is not None and value.shape[0] == batch.dp_size * batch.per_dp_bs_size:
            updates[name] = np.asarray(value)[batch.logits_indices_selector]
    return dataclasses.replace(state, **updates) if updates else state
