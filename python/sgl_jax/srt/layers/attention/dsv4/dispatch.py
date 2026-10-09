"""SWA/CSA/HCA dispatch over C-owned resource arrays.

Adapted from epic/dsv4 at ce1ebb637. Host read tables derive addresses from
C's request ledger. Device execution consumes M's projected Q/KV and compressor
weights, and returns functional per-family updates for B to package. The
model owns parameters and projections; R owns transport and update commits.
Production HCA uses the existing native-layout HCA mixin where supported;
the generic path also provides numerical coverage on CPU and small geometries.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.attention import (
    csa_fused_attention,
    csa_sparse_attention,
    dsv4_dense_attention,
    update_window_kv,
)
from sgl_jax.srt.layers.attention.dsv4.compressor import compress_chunk
from sgl_jax.srt.layers.attention.dsv4.indexer import (
    csa_indexer_topk_kernel,
    membership_from_scores,
    resolve_indexer_backend,
)
from sgl_jax.srt.layers.attention.dsv4.ref.indexer import csa_indexer_topk_ref

# Token count from which the fused CSA kernel replaces the XLA dense path.
_FUSED_MIN_TOKENS = int(os.environ.get("DSV4_CSA_FUSED_MIN_TOKENS", "64"))
# ``DSV4_CSA_TOPK_MASK=1``: on single-request steps the indexer returns the top-k
# membership mask (score >= k-th largest, bisection kernel) instead of an index list,
# skipping approx_max_k's per-row sort and the [T, K, E] membership reduction.
_TOPK_MASK = os.environ.get("DSV4_CSA_TOPK_MASK", "1") == "1"
# ``DSV4_INDEXER_ROW_SHARD=1``: inside the attention shard_map every device holds the
# full [T, H, D] indexer queries and scores all of them (8x redundant work on the
# exposed path, ~1.3 ms/layer at 8K). Score only this device's T/n row block and
# all-gather the [T/n, E] top-k membership back along "tensor" (2 MB at 8K).
# Single-request batches only; others keep the replicated path.
_INDEXER_ROW_SHARD = os.environ.get("DSV4_INDEXER_ROW_SHARD", "1") == "1"
# ``DSV4_COMPRESSOR_ROW_SHARD=1``: same idea for the two ratio-4 compressors, which
# every device ran over the full chunk (~1.6 ms/layer exposed at 8K). Each device
# compresses its own T/n rows (with a 7-row halo so every window is local), only
# its own N/n boundary slots, and all-gathers the records; the state ring is
# written from the request's last 8 tokens re-projected from the full input.
_COMPRESSOR_ROW_SHARD = os.environ.get("DSV4_COMPRESSOR_ROW_SHARD", "1") == "1"
_ROW_SHARD_AXIS = os.environ.get("DSV4_INDEXER_ROW_SHARD_AXIS", "tensor")


def resolve_csa_attention_backend() -> str:
    """``DSV4_CSA_ATTENTION=auto|sparse|dense|fused`` (auto == dense; fused = Pallas flash kernel)."""
    from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

    default = "fused" if is_tpu_runtime() else "dense"
    mode = os.environ.get("DSV4_CSA_ATTENTION", default).lower()
    if mode == "auto":
        # The gathered kernels only pay off when the per-block selection union is far
        # smaller than the candidate set; CSA's per-query top-512 over <=32K history
        # is not that case (measured ~45x slower than dense at 8K), so auto == dense.
        return "dense"
    if mode not in ("sparse", "dense", "fused"):
        raise ValueError(f"DSV4_CSA_ATTENTION must be auto|sparse|dense|fused, got {mode!r}")
    return mode


__all__ = [
    "ReadTables",
    "read_tables",
    "run_layer",
]


@jax.tree_util.register_pytree_node_class
class ReadTables:
    """Flat read addresses for one step, one DP rank.

    Attributes:
      window_rows: SWA physical rows every query in this step may need.
      window_positions / window_request_ids: what each row is.
      compressed_rows: flat compressed-entry addresses (C1's ``loc // ratio``).
      compressed_entry_ids / compressed_request_ids: what each entry is.

    Rows are a superset of any single query's window -- narrowing per query is the
    mask's job in M2.4, not this table's.
    """

    __slots__ = (
        "window_rows",
        "window_positions",
        "window_request_ids",
        "compressed_rows",
        "compressed_entry_ids",
        "compressed_request_ids",
        "decode_page_indices",
        "decode_window_rows",
        "decode_page_segments",
        "decode_page_segment_counts",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name) if name.startswith("decode_") else kw[name])

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls.__slots__, children, strict=True)))

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"ReadTables(window={len(self.window_rows)}, compressed={len(self.compressed_rows)})"


def read_tables(
    *,
    request_pool,
    allocator,
    slots,
    lengths,
    q_lens,
    ratio: int,
    window_size: int,
    page_size: int,
    rank: int = 0,
) -> ReadTables:
    """Build the step's read addresses from C1's ownership map.

    Args:
      request_pool: C1's `ReqToTokenPool`; `req_to_token[slot, p]` is the
        original-token location of position `p`.
      allocator: C1's V4 allocator, for `full_to_swa_index_mapping`.
      slots: [B] request slots. lengths: [B] seq_lens after this step.
      q_lens: [B] query tokens this step (0 for an inactive request).
      ratio: 0, 4 or 128.

    Raises rather than guessing when a required SWA page has been released -- a
    zero in the mapping means "not allocated", and reading row 0 would silently
    return another request's data.
    """
    mapping = allocator.full_to_swa_index_mapping
    mapping = mapping[rank] if isinstance(mapping, list) else mapping
    slots = np.asarray(slots, np.int64)
    lengths = np.asarray(lengths, np.int64)
    q_lens = np.asarray(q_lens, np.int64)

    w_rows, w_pos, w_req = [], [], []
    c_rows, c_ids, c_req = [], [], []
    for r, (slot, length, n) in enumerate(zip(slots, lengths, q_lens, strict=True)):
        if not n:
            continue
        last = int(length) - 1
        # Every position any query of this request may reach: its own window plus the
        # window of the oldest query in the chunk.
        first = max(0, int(length) - int(n) - window_size + 1)
        positions = np.arange(first, last + 1, dtype=np.int64)
        locations = np.asarray(request_pool.req_to_token[int(slot), positions], np.int64)
        if np.any(locations < page_size) or np.any(locations >= mapping.size):
            raise ValueError("V4 window positions must name allocated original-token slots")
        swa = mapping[locations]
        if np.any(swa == 0):
            raise ValueError(
                "V4 SWA rows required by this step were released; row 0 is the padding "
                "row and would return another request's data"
            )
        w_rows.append(swa)
        w_pos.append(positions)
        w_req.append(np.full(positions.shape, r, np.int64))

        if ratio > 0:
            complete = (last + 1) // ratio
            if complete:
                entries = np.arange(complete, dtype=np.int64)
                anchors = np.asarray(
                    request_pool.req_to_token[int(slot), entries * ratio], np.int64
                )
                if np.any(anchors < page_size):
                    raise ValueError("V4 compressed anchors must name allocated slots")
                c_rows.append(anchors // ratio)
                c_ids.append(entries)
                c_req.append(np.full(entries.shape, r, np.int64))

    def pack(parts, empty_fill):
        if not parts:
            return np.full((1,), empty_fill, np.int32)
        return np.concatenate(parts).astype(np.int32)

    return ReadTables(
        window_rows=pack(w_rows, 0),
        window_positions=pack(w_pos, -1),
        window_request_ids=pack(w_req, -1),
        compressed_rows=pack(c_rows, 0),
        compressed_entry_ids=pack(c_ids, -1),
        compressed_request_ids=pack(c_req, -1),
    )


def run_layer(
    *,
    q,
    new_kv,
    layer_id: int,
    ratio: int,
    metadata,
    tables: ReadTables,
    kv_buffers,
    state,
    compressor_weights=None,
    compressor_input=None,
    indexer=None,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    head_dim: int,
    index_topk: int | None = None,
    rope_head_dim: int = 64,
    norm_eps: float = 1e-6,
):
    """One V4 attention layer: compress, select, attend, and update both tiers.

    Args:
      q: [T, H, D] projected queries. new_kv: [T, D] this step's KV rows.
      metadata: `DeepseekV4AttentionMetadata` from M2.1.
      kv_buffers: dict with `"swa"` [W, D] and, for ratio > 0, `"compressed"` [E_all, D]
        and (ratio 4) `"indexer"` [E_all, Di]; C1 buffers flattened over their first
        two axes so a compressed address indexes them directly.
      state: [S, window, 2*coff*D] compressor state, or None for ratio 0.
      compressor_input: original [T, hidden] sublayer input, separate from projected KV.
      compressor_weights: `wkv`/`wgate`/`ape`/`norm_weight`/`cos_sin_cache` for the
        main compressor; required when ratio > 0.
      indexer: for ratio 4, a dict with the indexer's own `compressor_weights`,
        `q` [T, Hi, Di], `weights` [T, Hi] and `state`.

    Returns:
      `(out, updates)` where `updates` carries the new `swa`, `compressed`,
      `indexer` and `state` arrays that were actually touched. Nothing is written in
      place; the caller hands the results to `MemoryPools.replace_all`.
    """
    ratio_md = metadata.ratio(ratio) if ratio > 0 else None
    updates = {}

    compressed_kv = jnp.zeros((1, head_dim), jnp.float32)
    selected = None
    selected_mask = None

    if ratio > 0:
        if compressor_weights is None:
            raise ValueError(f"ratio {ratio} needs compressor weights")
        if compressor_input is None:
            raise ValueError("compressed layers require original hidden compressor_input")
        # Cache addresses and visibility use group ids; RoPE uses the group's
        # start in original-token coordinates (SGLang: seq_len - ratio).
        rope_positions = ratio_md.boundary_group_ids * ratio
        kv_call = functools.partial(
            compress_chunk,
            compressor_input,
            state=state,
            positions=metadata.query_positions,
            query_request_ids=metadata.query_request_ids,
            prefix_lens=metadata.prefix_lens,
            cu_q_lens=metadata.cu_q_lens,
            state_slots=metadata.request_slots,
            boundary_token_indices=ratio_md.boundary_token_indices,
            boundary_valid_mask=ratio_md.boundary_valid_mask,
            boundary_compressed_pos=rope_positions,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
            **compressor_weights,
        )
        dual = None
        row_shard_kw = _compressor_row_shard_plan(compressor_input, metadata, ratio_md, ratio)
        if row_shard_kw is not None:
            dual = _row_sharded_compress(
                compressor_input,
                state=state,
                weights=compressor_weights,
                rope_positions=rope_positions,
                metadata=metadata,
                ratio_md=ratio_md,
                ratio=ratio,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                norm_eps=norm_eps,
                **row_shard_kw,
            ) + _row_sharded_compress(
                indexer["compressor_input"],
                state=indexer["state"],
                weights=indexer["compressor_weights"],
                rope_positions=rope_positions,
                metadata=metadata,
                ratio_md=ratio_md,
                ratio=ratio,
                head_dim=indexer["head_dim"],
                rope_head_dim=indexer.get("rope_head_dim", rope_head_dim),
                norm_eps=norm_eps,
                **row_shard_kw,
            )
        if dual is None:
            records, record_valid, new_state = kv_call()
        else:
            records, record_valid, new_state, idx_records, idx_valid, idx_state = dual
        updates["state"] = new_state
        # Records land at the addresses M2.1 derived; invalid boundaries are dropped
        # rather than aimed at entry 0.
        compressed_buffer = _scatter_records(
            kv_buffers["compressed"],
            records,
            ratio_md.boundary_write_entries,
            record_valid,
            run=metadata.page_size // ratio,
        )
        updates["compressed"] = compressed_buffer
        if tables.decode_page_indices is None:
            compressed_kv = jnp.take(compressed_buffer, jnp.asarray(tables.compressed_rows), axis=0)

        if ratio == 4:
            if indexer is None or index_topk is None:
                raise ValueError("CSA layers need the indexer and index_topk")
            if dual is None:
                idx_records, idx_valid, idx_state = compress_chunk(
                    indexer["compressor_input"],
                    state=indexer["state"],
                    positions=metadata.query_positions,
                    query_request_ids=metadata.query_request_ids,
                    prefix_lens=metadata.prefix_lens,
                    cu_q_lens=metadata.cu_q_lens,
                    state_slots=metadata.request_slots,
                    boundary_token_indices=ratio_md.boundary_token_indices,
                    boundary_valid_mask=ratio_md.boundary_valid_mask,
                    boundary_compressed_pos=rope_positions,
                    ratio=ratio,
                    head_dim=indexer["head_dim"],
                    rope_head_dim=indexer.get("rope_head_dim", rope_head_dim),
                    norm_eps=norm_eps,
                    **indexer["compressor_weights"],
                )
            updates["indexer_state"] = idx_state
            indexer_buffer = _scatter_records(
                kv_buffers["indexer"],
                idx_records,
                ratio_md.boundary_write_entries,
                idx_valid,
                run=metadata.page_size // ratio,
            )
            updates["indexer"] = indexer_buffer
            if tables.decode_page_indices is None:
                if resolve_indexer_backend() == "kernel":
                    # Pallas scoring straight from the paged cache; same gathered-row
                    # coordinates as the reference, no [T, E] key gather.
                    use_mask = _TOPK_MASK and resolve_csa_attention_backend() != "sparse"
                    num_entries = int(np.shape(tables.compressed_entry_ids)[0])
                    row_shard = (
                        use_mask
                        and _INDEXER_ROW_SHARD
                        and int(metadata.cu_q_lens.shape[0]) == 2
                        and _row_shard_axis_size() is not None
                        and int(indexer["q"].shape[0]) % _row_shard_axis_size() == 0
                    )
                    if row_shard:
                        selected_mask = _row_sharded_membership(
                            indexer["q"],
                            indexer["weights"],
                            indexer_buffer,
                            metadata=metadata,
                            tables=tables,
                            k=index_topk,
                            ratio=ratio,
                            num_entries=num_entries,
                        )
                    else:
                        picked = csa_indexer_topk_kernel(
                            indexer["q"],
                            indexer["weights"],
                            indexer_buffer,
                            compressed_rows=jnp.asarray(tables.compressed_rows),
                            seq_lens=metadata.seq_lens,
                            q_lens=metadata.q_lens,
                            cu_q_lens=metadata.cu_q_lens,
                            query_request_ids=metadata.query_request_ids,
                            valid_token_mask=metadata.valid_token_mask,
                            k=index_topk,
                            ratio=ratio,
                            compressed_page_size=metadata.page_size // ratio,
                            return_scores=use_mask,
                        )
                        if use_mask:
                            scores, offsets = picked
                            selected_mask = membership_from_scores(
                                scores,
                                offsets,
                                q_lens=metadata.q_lens,
                                query_request_ids=metadata.query_request_ids,
                                valid_token_mask=metadata.valid_token_mask,
                                k=index_topk,
                                num_entries=num_entries,
                            )
                        else:
                            selected = picked
                else:
                    indexer_keys = jnp.take(
                        indexer_buffer, jnp.asarray(tables.compressed_rows), axis=0
                    )
                    selected = csa_indexer_topk_ref(
                        indexer["q"],
                        indexer["weights"],
                        indexer_keys,
                        metadata.query_positions,
                        metadata.query_request_ids,
                        jnp.asarray(tables.compressed_request_ids),
                        metadata.valid_token_mask,
                        entry_group_ids=jnp.asarray(tables.compressed_entry_ids),
                        k=index_topk,
                        ratio=ratio,
                    )

    # C1 retains all SWA pages read by this chunk until it completes. Publish
    # current-token KV before the read; causal positions, not write ordering,
    # prevent a query from attending to later tokens in the chunk.
    updates["swa"] = update_window_kv(
        kv_buffers["swa"], new_kv, metadata.swa_write_loc, metadata.valid_token_mask
    )
    if tables.decode_page_indices is not None:
        from sgl_jax.srt.layers.attention.dsv4.decode import csa_decode_attention

        out = csa_decode_attention(
            q,
            indexer["q"],
            indexer["weights"],
            updates["indexer"],
            updates["compressed"],
            updates["swa"],
            tables.decode_page_indices,
            tables.decode_window_rows,
            query_positions=metadata.query_positions,
            valid_token_mask=metadata.valid_token_mask,
            attention_sink=attention_sink,
            softmax_scale=softmax_scale,
            compressed_page_size=metadata.page_size // ratio,
            index_topk=index_topk,
            ratio=ratio,
            page_segments=tables.decode_page_segments,
            page_segment_counts=tables.decode_page_segment_counts,
        )
        return out, updates
    window_kv = jnp.take(updates["swa"], jnp.asarray(tables.window_rows), axis=0)
    backend = resolve_csa_attention_backend()
    if selected is not None and backend == "sparse":
        attend = csa_sparse_attention
    elif backend == "fused" and q.shape[0] >= _FUSED_MIN_TOKENS:
        # Measured on v7x: the flash kernel wins on prefill chunks (8K TTFT -64 ms)
        # but costs +0.3 ms/step on decode buckets (T <= 16), where the XLA dense
        # path over a few hundred keys is cheaper than a pallas_call per layer.
        attend = csa_fused_attention
    else:
        attend = dsv4_dense_attention
    mask_kwargs = {} if attend is csa_sparse_attention else {"selected_mask": selected_mask}
    out = attend(
        q,
        window_kv,
        compressed_kv,
        query_positions=metadata.query_positions,
        query_request_ids=metadata.query_request_ids,
        valid_token_mask=metadata.valid_token_mask,
        window_positions=jnp.asarray(tables.window_positions),
        window_request_ids=jnp.asarray(tables.window_request_ids),
        compressed_entry_ids=jnp.asarray(tables.compressed_entry_ids),
        compressed_request_ids=jnp.asarray(tables.compressed_request_ids),
        attention_sink=attention_sink,
        softmax_scale=softmax_scale,
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected,
        **mask_kwargs,
    )

    return out, updates


# ``DSV4_PAGED_RECORD_WRITE=1``: on prefill-sized boundary sets, place the compressed
# records with the page-run DMA writer instead of an XLA row scatter. A chunk's
# records land at consecutive entries inside each allocator page (``page_size //
# ratio`` of them), which the writer DMAs as one segment; segments that are not a
# tile-aligned contiguous run fall back to its row path, so any layout stays
# correct. The scatter costs ~0.28 ms per call (two per CSA layer) on an 8K chunk.
_PAGED_RECORD_WRITE = os.environ.get("DSV4_PAGED_RECORD_WRITE", "1") == "1"

_PAGED_RECORD_MIN = int(os.environ.get("DSV4_PAGED_RECORD_WRITE_MIN_RECORDS", "256"))


def _save_tail_single(state, rows, positions, slot, keep, window):
    """``compress_chunk._save_tail`` for one request: write its last ``window`` tokens."""
    flat = state.reshape(-1, state.shape[-1])
    index = jnp.where(keep, slot * window + jnp.mod(positions, window), flat.shape[0])
    return flat.at[index].set(rows, mode="drop").reshape(state.shape)


def _row_shard_axis_size():
    """Size of the row-shard mesh axis when tracing inside a shard_map, else None."""
    try:
        return int(jax.lax.axis_size(_ROW_SHARD_AXIS))
    except Exception:  # not under shard_map (CPU tests, single device)
        return None


def _row_sharded_membership(
    q, weights, indexer_buffer, *, metadata, tables, k, ratio, num_entries, scorer=None
):
    """Top-k membership [T, E] for one request, each device scoring its own T/n rows.

    Device ``i`` takes query rows ``[i*rows, (i+1)*rows)`` and presents them to the
    kernel as the last ``real`` queries of a sequence ending at ``prefix + i*rows +
    real`` (``real`` = rows of the block below the request's q_len, at least one so
    the kernel always has a query; surplus rows are invalidated). Masks are gathered
    along the axis back to full T.
    """
    n = _row_shard_axis_size()
    index = jax.lax.axis_index(_ROW_SHARD_AXIS)
    tokens = int(q.shape[0])
    rows = tokens // n
    start = index * rows
    q_len = jnp.asarray(metadata.cu_q_lens)[1] - jnp.asarray(metadata.cu_q_lens)[0]
    real = jnp.clip(q_len - start, 1, rows)
    prefix = jnp.asarray(metadata.prefix_lens)[0]
    seq_len = prefix + start + real
    q_i = jax.lax.dynamic_slice_in_dim(q, start, rows, axis=0)
    w_i = jax.lax.dynamic_slice_in_dim(jnp.asarray(weights), start, rows, axis=0)
    valid_i = (jnp.arange(rows, dtype=jnp.int32) < real) & (
        jax.lax.dynamic_slice_in_dim(
            jnp.asarray(metadata.valid_token_mask, bool), start, rows, axis=0
        )
    )
    q_lens_i = jnp.asarray([real], jnp.int32)
    scorer = csa_indexer_topk_kernel if scorer is None else scorer
    scores, offsets = scorer(
        q_i,
        w_i,
        indexer_buffer,
        compressed_rows=jnp.asarray(tables.compressed_rows),
        seq_lens=jnp.asarray([seq_len], jnp.int32),
        q_lens=q_lens_i,
        cu_q_lens=jnp.asarray([0, real], jnp.int32),
        query_request_ids=jnp.zeros((rows,), jnp.int32),
        valid_token_mask=valid_i,
        k=k,
        ratio=ratio,
        compressed_page_size=metadata.page_size // ratio,
        return_scores=True,
    )
    mask_i = membership_from_scores(
        scores,
        offsets,
        q_lens=q_lens_i,
        query_request_ids=jnp.zeros((rows,), jnp.int32),
        valid_token_mask=valid_i,
        k=k,
        num_entries=num_entries,
    )
    return jax.lax.all_gather(mask_i, _ROW_SHARD_AXIS, axis=0, tiled=True)


def _compressor_row_shard_plan(x, metadata, ratio_md, ratio):
    """Static gate for the row-sharded compressors: returns the block geometry kwargs
    or None (flag off, not under shard_map, ratio != 4, more than one request slot,
    or T / N not divisible by the axis size).

    ``x`` is either the full chunk ``[T, hidden]`` (every device slices its block) or,
    in local mode, this device's block ``[T/n, hidden]`` only (the caller all-gathers the low-rank projections itself:
    the previous block's last rows arrive by ppermute). Local mode has no fallback,
    so a failed gate there is a configuration error.
    """
    if ratio != 4 or ratio_md is None:
        return None
    n = _row_shard_axis_size()
    tokens = int(metadata.query_positions.shape[0])
    local = n is not None and int(x.shape[0]) != tokens
    if not _COMPRESSOR_ROW_SHARD:
        if local:
            raise ValueError("local-mode compression needs DSV4_COMPRESSOR_ROW_SHARD=1")
        return None
    if n is None or n < 2 or int(metadata.cu_q_lens.shape[0]) != 2:
        if local:
            raise ValueError("local compressor input needs a single-request batch under shard_map")
        return None
    if local and int(x.shape[0]) * n != tokens:
        raise ValueError("local compressor input must be exactly T / axis_size rows")
    slots = int(ratio_md.boundary_token_indices.shape[0])
    if tokens % n or tokens // n < 8 or (tokens // n) % ratio:
        if local:
            raise ValueError(
                "local compressor input needs T % n == 0, T/n >= 8 and T/n % ratio == 0"
            )
        return None
    # slot j is the group ending at token ratio*j + ratio - 1, so block i owns slots
    # [i*rows/ratio, (i+1)*rows/ratio); the boundary axis carries B extra slots
    # (capacity = T // ratio + B) that only the last block can own
    per_block = tokens // n // ratio
    extra = slots - n * per_block
    if extra < 0:
        return None
    return dict(axis_size=n, rows=tokens // n, slots=per_block, extra=extra, local=local)


def _row_sharded_compress(
    x,
    *,
    state,
    weights,
    rope_positions,
    metadata,
    ratio_md,
    ratio,
    head_dim,
    rope_head_dim,
    norm_eps,
    axis_size,
    rows,
    slots,
    extra,
    local=False,
):
    """``compress_chunk`` for one request with the chunk split across the axis.

    Device ``i`` compresses rows ``[i*rows - halo, (i+1)*rows)`` (halo = window - 1,
    so every boundary window inside the block is served from the slice; device 0
    reads the ring as usual) for boundary slots ``[i*slots, (i+1)*slots)`` — for a
    single aligned request slot ``j`` is the group ending at token ``4j+3``, so
    slots and rows partition alike. Records and their valid mask are all-gathered
    in slot order; the state ring is rewritten from the request's last ``window``
    tokens, re-projected from the full input, so no block writes a wrong tail.
    Returns ``(records, record_valid, new_state)``.
    """
    from sgl_jax.srt.layers.attention.dsv4.compressor import (
        project_tokens,
        state_window,
    )

    window = state_window(ratio)
    halo = window - 1
    index = jax.lax.axis_index(_ROW_SHARD_AXIS)
    tokens = int(metadata.query_positions.shape[0])
    start = index * rows
    length = rows + halo
    cu = jnp.asarray(metadata.cu_q_lens)
    q_len = cu[1] - cu[0]
    prefix = jnp.asarray(metadata.prefix_lens)[0]
    slot = jnp.asarray(metadata.request_slots)[0]
    positions = jnp.asarray(metadata.query_positions)

    if local:
        # x is this block only: rows [start, start + rows). The previous block's last
        # `halo` rows arrive by ppermute; the slice layout is [halo rows | block rows]
        # for every device. Block 0 keeps chunk_start = prefix so its (meaningless)
        # halo rows are never read as chunk rows: windows before the chunk come from
        # the ring, and the block rows start at row `halo` (cu_q_lens[0] = halo).
        prev_tail = jax.lax.ppermute(
            x[-halo:], _ROW_SHARD_AXIS, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
        )
        x_i = jnp.concatenate((prev_tail, x), axis=0)
        sstart = start - halo
        pos_i = prefix + sstart + jnp.arange(length, dtype=jnp.int32)
        real_rows = jnp.clip(q_len - start, 0, rows)
        first = index == 0
        prefix_i = jnp.where(first, prefix, prefix + sstart)
        row0 = jnp.where(first, halo, 0)
        cu_i = jnp.stack((row0, halo + real_rows)).astype(jnp.int32)
    else:
        sstart = jnp.maximum(start - halo, 0)  # device 0 reads [0, rows + halo)
        x_i = jax.lax.dynamic_slice_in_dim(x, sstart, length, axis=0)
        pos_i = jax.lax.dynamic_slice_in_dim(positions, sstart, length, axis=0)
        real_i = jnp.clip(q_len - sstart, 0, length)
        prefix_i = prefix + sstart
        cu_i = jnp.stack((jnp.zeros((), jnp.int32), real_i)).astype(jnp.int32)
    bidx = jnp.asarray(ratio_md.boundary_token_indices)
    bvalid = jnp.asarray(ratio_md.boundary_valid_mask, bool)
    rope_positions = jnp.asarray(rope_positions)
    num_slots = int(bidx.shape[0])
    # every block reads `slots + extra` consecutive slots from its own first slot; the
    # windows overlap by `extra` and the block-membership mask below dedups them
    width = slots + extra
    b0 = index * slots
    bidx_i = jax.lax.dynamic_slice_in_dim(bidx, b0, width, axis=0)
    bvalid_i = jax.lax.dynamic_slice_in_dim(bvalid, b0, width, axis=0)
    rope_i = jax.lax.dynamic_slice_in_dim(rope_positions, b0, width, axis=0)
    # a slot is this block's only if its token lies inside the block (not the halo)
    inside = (bidx_i >= start) & (bidx_i < start + rows) & (bidx_i < tokens)
    bvalid_i = bvalid_i & inside
    local_bidx = jnp.where(inside, bidx_i - sstart, 0)

    records_i, valid_i, _ = compress_chunk(
        x_i,
        state=state,
        positions=pos_i,
        query_request_ids=jnp.zeros((length,), jnp.int32),
        prefix_lens=jnp.reshape(prefix_i, (1,)).astype(jnp.int32),
        cu_q_lens=cu_i,
        state_slots=jnp.asarray(metadata.request_slots),
        boundary_token_indices=local_bidx,
        boundary_valid_mask=bvalid_i,
        boundary_compressed_pos=rope_i,
        ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        norm_eps=norm_eps,
        **weights,
    )

    def _reassemble(per_block):
        # [n, width, ...] -> slot order: each block's own `slots`, then the last block's extra
        g = jax.lax.all_gather(per_block, _ROW_SHARD_AXIS, axis=0, tiled=False)
        own = g[:, :slots].reshape((axis_size * slots,) + g.shape[2:])
        tail = g[axis_size - 1, slots:]
        return jnp.concatenate((own, tail), axis=0)[:num_slots]

    records = _reassemble(records_i)
    valid = _reassemble(valid_i)

    # carried state: the request's last `window` real tokens
    tail_start = jnp.clip(q_len - window, 0, tokens - window)
    if local:
        # only the block holding the last real token has all `window` tail rows (its
        # halo covers the spill into the previous block); it projects them, the
        # others contribute zeros, and a small psum shares the rows
        owner = jnp.clip((q_len - 1) // rows, 0, axis_size - 1) == index
        local_tail = jnp.clip(tail_start - sstart, 0, length - window)
        x_tail = jax.lax.dynamic_slice_in_dim(x_i, local_tail, window, axis=0)
        x_tail = jnp.where(owner, x_tail, jnp.zeros_like(x_tail))
        x_tail = jax.lax.psum(x_tail, _ROW_SHARD_AXIS)
        pos_tail = tail_start + jnp.arange(window, dtype=jnp.int32) + prefix
    else:
        x_tail = jax.lax.dynamic_slice_in_dim(x, tail_start, window, axis=0)
        pos_tail = jax.lax.dynamic_slice_in_dim(positions, tail_start, window, axis=0)
    keep = (tail_start + jnp.arange(window)) < q_len
    kv, score = project_tokens(
        x_tail,
        weights["wkv"],
        weights["wgate"],
        weights["ape"],
        pos_tail,
        ratio=ratio,
    )
    new_state = _save_tail_single(
        jnp.asarray(state, jnp.float32),
        jnp.concatenate((kv, score), axis=-1),
        pos_tail,
        slot,
        keep,
        window,
    )
    return records, valid, new_state


def _scatter_records(buffer, records, write_entries, valid, *, run: int | None = None):
    """Place records at their compressed addresses, dropping invalid boundaries.

    ``run`` is the number of entries per allocator page; with the paged writer
    enabled it is the DMA segment length.
    """
    buffer = jnp.asarray(buffer)
    entries = jnp.asarray(write_entries)
    keep = jnp.asarray(valid, bool) & (entries >= 0) & (entries < buffer.shape[0])
    if (
        _PAGED_RECORD_WRITE
        and run
        and run % 16 == 0
        and buffer.ndim == 2
        and buffer.dtype == jnp.bfloat16
        and int(jnp.shape(records)[0]) >= _PAGED_RECORD_MIN
    ):
        from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write

        return paged_row_write(
            buffer,
            records,
            entries,
            keep,
            run=run,
            interpret=os.environ.get("PALLAS_INTERPRET", "0") == "1",
        )
    entries = jnp.where(keep, entries, buffer.shape[0])
    return buffer.at[entries].set(jnp.asarray(records, buffer.dtype), mode="drop")
