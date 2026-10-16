"""Output-token logprobs under speculative decoding (EAGLE / NEXTN).

The non-speculative path computes ``next_token_logprobs`` inside the jitted
sampler and materializes them in ``ModelWorker.forward_batch_generation``.
Speculative verify never samples through that path: the target forward runs
with ``skip_sample=True`` (recurrent verify) or inside the fused verify JIT, and
the accepted tokens are picked by tree/chain verification.  This module adds
the missing piece on both sides:

* worker side: ``attach_spec_output_logprobs`` turns the verify logits that
  were already gathered at the accepted positions (``bs * (steps + 1)`` rows,
  rejected slots redirected to each request's last slot) into per-row token
  logprobs plus optional top-k / token-id logprobs;
* scheduler side: ``append_spec_output_logprobs`` walks the ``accept_len``
  accepted rows of one request and appends them to the request's output
  logprob lists, mirroring GPU sglang ``compute_spec_v2_logprobs`` +
  ``_apply_decode_logprobs``.

Prefill under speculative decoding reuses the regular sampler (which already
computes prompt and first-token logprobs); ``spec_prefill_can_skip_sample``
decides when the greedy ``argmax`` shortcut is still allowed.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def batch_needs_output_logprob(batch) -> bool:
    return bool(
        getattr(batch, "return_logprob", False)
        or getattr(batch, "return_output_logprob_only", False)
    )


def spec_prefill_can_skip_sample(batch, legacy_non_overlap: bool) -> bool:
    """Greedy spec prefill may replace the sampler with ``argmax`` only when no
    request asked for logprobs: the sampler is the only path that computes and
    materializes ``next_token_logprobs`` / prompt logprobs."""
    if legacy_non_overlap:
        return False
    if not batch.sampling_info.is_all_greedy:
        return False
    return not batch_needs_output_logprob(batch)


def draft_extend_logits_metadata(batch, mesh):
    """LogitsMetadata for a draft-model extend built from a *target* extend batch.

    The target prefill batch carries the per-request extend-logprob lists, so
    ``LogitsMetadata.from_model_worker_batch`` would make the draft model emit
    full-vocab logits for every prompt token (``chunk x vocab`` fp32) and run
    log-softmax over them.  Draft-model logprobs are never returned, so build
    the metadata with every extend-logprob switch off; ``logits_indices`` keeps
    selecting the last token per request exactly as the no-logprob path does.
    """
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    md = LogitsMetadata.from_model_worker_batch(batch, mesh)
    return dataclasses.replace(
        md,
        extend_return_logprob=False,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_logprob_pruned_lens_cpu=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids_device=None,
        input_logprob_indices_device=None,
    )


@partial(jax.jit, static_argnames=("top_k", "need_full"))
def compute_spec_output_logprobs(
    logits, token_ids, temperatures=None, *, top_k: int, need_full: bool
):
    """Per-row logprob of ``token_ids`` under ``logits`` (``(N, vocab)``).

    Returns ``(token_logprobs (N,), top_vals (N, top_k) | None,
    top_idx (N, top_k) | None, full_logprobs (N, vocab) | None)``.  Matches the
    regular sampler's convention: greedy rows use the raw log-softmax, sampled
    rows the temperature-scaled one (``temperatures`` is ``(N,)`` or None).
    """
    logits = logits.astype(jnp.float32)
    if temperatures is not None:
        logits = logits / temperatures.astype(jnp.float32)[:, None]
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    logprobs = jnp.maximum(logprobs, jnp.finfo(jnp.float32).min)
    rows = jnp.arange(logprobs.shape[0], dtype=jnp.int32)
    token_logprobs = logprobs[rows, token_ids.astype(jnp.int32)]
    if top_k > 0:
        top_vals, top_idx = jax.lax.top_k(logprobs, top_k)
    else:
        top_vals = top_idx = None
    full = logprobs if need_full else None
    return token_logprobs, top_vals, top_idx, full


def _max_top_k(top_logprobs_nums) -> int:
    if not top_logprobs_nums:
        return 0
    return int(max(int(x) for x in top_logprobs_nums))


def _any_token_ids(token_ids_logprobs) -> bool:
    return bool(token_ids_logprobs) and any(ids for ids in token_ids_logprobs)


def attach_spec_output_logprobs(logits_output, token_ids, batch, mesh, *, width: int):
    """Fill ``logits_output.next_token_*`` for the ``bs * width`` verify rows.

    ``logits_output.next_token_logits`` must already be gathered at the accepted
    positions (replicated, ``(bs * width, vocab)``); ``token_ids`` are the
    matching accepted token ids (``verified_id``).  ``top_logprobs_nums`` /
    ``token_ids_logprobs`` come from the DP-padded ``ModelWorkerBatch`` lists.
    Only ``return_logprob`` requests get top-k / token-id columns;
    ``return_output_logprob_only`` needs the token logprob alone.
    """
    logits = logits_output.next_token_logits
    if logits is None:
        return logits_output
    n_rows = int(logits.shape[0])
    if getattr(batch, "return_logprob", False):
        top_k = _max_top_k(getattr(batch, "top_logprobs_nums", None))
        need_full = _any_token_ids(getattr(batch, "token_ids_logprobs", None))
    else:
        top_k = 0
        need_full = False
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if not isinstance(token_ids, jax.Array) or token_ids.sharding != rep:
        token_ids = jax.device_put(np.asarray(token_ids, dtype=np.int32), rep)
    temperatures = None
    sampling_info = getattr(batch, "sampling_info", None)
    if sampling_info is not None and not getattr(sampling_info, "is_all_greedy", True):
        temps = np.asarray(jax.device_get(sampling_info.temperatures), dtype=np.float32).reshape(-1)
        temps = np.repeat(temps, width)
        if temps.shape[0] < n_rows:
            temps = np.pad(temps, (0, n_rows - temps.shape[0]), constant_values=1.0)
        temperatures = jax.device_put(temps[:n_rows], rep)
    with jax.set_mesh(mesh):
        token_logprobs, top_vals, top_idx, full = compute_spec_output_logprobs(
            logits, token_ids, temperatures, top_k=top_k, need_full=need_full
        )
    logits_output.next_token_logprobs = token_logprobs
    logits_output.next_token_top_logprobs_val = top_vals
    logits_output.next_token_top_logprobs_idx = top_idx
    logits_output.next_token_token_ids_logprobs_val = full
    logits_output.next_token_token_ids_logprobs_idx = None
    return logits_output


@dataclasses.dataclass
class SpecOutputLogprobsHost:
    """Host copies of the per-row verify logprobs (``bs * width`` rows)."""

    token_logprobs: np.ndarray
    top_vals: np.ndarray | None
    top_idx: np.ndarray | None
    full: np.ndarray | None
    width: int


def materialize_spec_output_logprobs(logits_output, width: int) -> SpecOutputLogprobsHost | None:
    """Pull the verify-row logprobs to host once per batch; None when the
    worker did not attach them (caller decides whether that is an error)."""
    if logits_output is None or logits_output.next_token_logprobs is None:
        return None
    arrays = [
        logits_output.next_token_logprobs,
        logits_output.next_token_top_logprobs_val,
        logits_output.next_token_top_logprobs_idx,
        logits_output.next_token_token_ids_logprobs_val,
    ]
    for a in arrays:
        if a is not None and hasattr(a, "copy_to_host_async"):
            a.copy_to_host_async()
    token_logprobs, top_vals, top_idx, full = (
        None if a is None else np.asarray(jax.device_get(a)) for a in arrays
    )
    return SpecOutputLogprobsHost(
        token_logprobs=token_logprobs.astype(np.float32),
        top_vals=None if top_vals is None else top_vals.astype(np.float32),
        top_idx=top_idx,
        full=None if full is None else full.astype(np.float32),
        width=width,
    )


def append_spec_output_logprobs(req, host: SpecOutputLogprobsHost, slot: int, accepted_ids) -> None:
    """Append one verify round's accepted tokens to ``req``'s output logprobs.

    ``slot`` is the request's DP-padded batch index; its rows are
    ``slot * width + j`` for ``j < len(accepted_ids)``.
    """
    accepted_ids = [int(t) for t in accepted_ids]
    n = len(accepted_ids)
    if n == 0:
        return
    base = slot * host.width
    if n > host.width or base + n > host.token_logprobs.shape[0]:
        raise ValueError(
            f"spec logprob rows out of range: slot={slot} width={host.width} "
            f"accepted={n} rows={host.token_logprobs.shape[0]}"
        )
    vals = host.token_logprobs[base : base + n].astype(float).tolist()
    req.output_token_logprobs_val.extend(vals)
    req.output_token_logprobs_idx.extend(accepted_ids)
    if not req.return_logprob:
        return
    k = int(getattr(req, "top_logprobs_num", 0) or 0)
    if k > 0:
        if host.top_vals is None or host.top_idx is None:
            raise ValueError("request asked for top_logprobs but verify rows carry none")
        for j in range(n):
            req.output_top_logprobs_val.append(host.top_vals[base + j, :k].astype(float).tolist())
            req.output_top_logprobs_idx.append(host.top_idx[base + j, :k].tolist())
    ids = getattr(req, "token_ids_logprob", None)
    if ids is not None:
        if host.full is None:
            raise ValueError("request asked for token_ids_logprob but verify rows carry none")
        ids = list(ids)
        for j in range(n):
            req.output_token_ids_logprobs_val.append(
                host.full[base + j, ids].astype(float).tolist()
            )
            req.output_token_ids_logprobs_idx.append(list(ids))
