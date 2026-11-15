"""FlyWheel attention backend (--attention-backend flywheel).

Runs MHA/GQA attention with the flywheel-tpu Pallas kernels
(https://github.com/heavyball-research/flywheel-tpu) on the same fused, head
interleaved KV pool as the FlashAttention (RPA v3) backend. The kernel
selection follows tpu-inference's flywheel backend
(tpu_inference/layers/common/flywheel_attention.py):

- DECODE: the paged decode kernel appends each request's new K/V token to its
  pages and attends them.
- EXTEND: a DMA writer copies the step's K/V rows into the pages, then
  lax.switch runs one kernel picked on the host: the paged varlen kernel when
  any request has a cached prefix (prefix cache hit, chunked prefill, mixed
  batch), else a dense or ragged prefill kernel over the step's own K/V.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

try:
    import flywheel_tpu
    from flywheel_tpu import flash_attn_func, flash_attn_with_kvcache
    from flywheel_tpu.flash_attn_interface import PAGE_SIZE_MULTIPLE
    from flywheel_tpu.pallas.block_sizes import FWD_BLOCKS, NUM_LANES
    from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "--attention-backend flywheel needs the flywheel-tpu package: "
        "pip install 'sglang-jax[flywheel]' or pip install flywheel-tpu"
    ) from e

logger = logging.getLogger(__name__)

# jit so that every layer calls one shared traced kernel, as
# flash_attn_with_kvcache and the KV writer already do.
flash_attn_varlen_func = jax.jit(
    flywheel_tpu.flash_attn_varlen_func,
    static_argnames=(
        "max_seqlen_q",
        "max_seqlen_k",
        "softmax_scale",
        "causal",
        "interpret",
        "token_major",
        "nheads_k",
    ),
)

# head_dims the paged decode kernel serves without padding.
SUPPORTED_HEAD_DIMS = (128, 256)

_EXTEND_KERNEL = "extend"
_DENSE_KERNEL = "dense"

# flash_attn_func pads the sequence axis to a multiple of NUM_LANES, then runs
# one block up to the largest of FWD_BLOCKS, else whole blocks of the largest
# FWD_BLOCKS size that divides the axis.
_DENSE_PAD_MULTIPLE = NUM_LANES
_DENSE_MAX_BLOCK = max(FWD_BLOCKS)

# Ragged prefill block spans: FWD_BLOCKS without the smallest, which never
# won on v6e (tpu-inference's measurement).
_VARLEN_BLOCK_SPANS = tuple(sorted(block for block in FWD_BLOCKS if block > min(FWD_BLOCKS)))

# Fixed cost of one ragged kernel step, as the side of a square tile of equal
# cost (fitted on v6e by tpu-inference).
_VARLEN_STEP_OVERHEAD_TOKENS = 704


def kv_write_slice_bound(num_tokens: int, num_seqs: int, page_size: int) -> int:
    """Upper bound on the page pieces one step writes: a request writing q_len
    tokens touches at most q_len // page_size + 2 pages."""
    return num_tokens // page_size + 2 * num_seqs


def build_kv_write_slices(
    cu_q_lens: np.ndarray,  # i32[num_seqs + 1]
    kv_lens: np.ndarray,  # i32[num_seqs], includes this step's tokens
    block_table: np.ndarray,  # i32[num_seqs, pages_per_seq]
    *,
    page_size: int,
    out: np.ndarray,  # i32[3, bound], overwritten
) -> int:
    """Fill out with the (cache_row_start, new_row_start, length) pieces that
    copy this step's new K/V rows into the pages and return their count.

    Rows are rows of the flat (num_pages * page_size) cache view and of the
    step's K/V; each piece lies inside one page. Later columns are zero.
    """
    num_seqs = kv_lens.shape[0]
    q_lens = cu_q_lens[1 : num_seqs + 1] - cu_q_lens[:num_seqs]
    kv_start = kv_lens - q_lens
    first_page = kv_start // page_size
    last_page = (kv_lens - 1) // page_size
    counts = np.where(q_lens > 0, last_page - first_page + 1, 0)
    total = int(counts.sum())
    if total > out.shape[1]:
        raise ValueError(f"KV write needs {total} slices but the table holds {out.shape[1]}.")
    seq_idx = np.repeat(np.arange(num_seqs), counts)
    piece_start = np.repeat(np.cumsum(counts) - counts, counts)
    page_slot = first_page[seq_idx] + (np.arange(total) - piece_start)
    page_start_token = page_slot * page_size
    write_start = np.maximum(kv_start[seq_idx], page_start_token)
    write_end = np.minimum(kv_lens[seq_idx], page_start_token + page_size)
    out[:] = 0
    out[0, :total] = block_table[seq_idx, page_slot] * page_size + (write_start - page_start_token)
    out[1, :total] = cu_q_lens[:num_seqs][seq_idx] + (write_start - kv_start[seq_idx])
    out[2, :total] = write_end - write_start
    return total


def is_dense_tileable(num_tokens: int) -> bool:
    """Whether the dense kernel can tile a (1, num_tokens) row."""
    return num_tokens % _DENSE_PAD_MULTIPLE == 0 and (
        num_tokens <= _DENSE_MAX_BLOCK or num_tokens % _DENSE_MAX_BLOCK == 0
    )


def varlen_block_spans(num_tokens: int) -> tuple[int, ...]:
    """The ragged prefill block spans compiled for a token bucket."""
    spans = tuple(s for s in _VARLEN_BLOCK_SPANS if s <= num_tokens)
    return spans or (num_tokens,)


def ragged_step_cost(lengths: Sequence[int], num_tokens: int, span: int) -> int:
    """Relative cost of one ragged prefill step at block span `span`.

    A sequence of t = ceil(length / span) blocks takes t * (t + 1) / 2 causal
    tile steps, the bucket's padded tail runs as one more sequence, and each
    step costs span**2 plus a fixed overhead of
    _VARLEN_STEP_OVERHEAD_TOKENS**2.
    """
    tail = num_tokens - sum(lengths)
    num_blocks = [-(-length // span) for length in [*lengths, tail] if length > 0]
    num_steps = sum(t * (t + 1) // 2 for t in num_blocks)
    return num_steps * (span * span + _VARLEN_STEP_OVERHEAD_TOKENS**2)


def varlen_block_span(lengths: Sequence[int], num_tokens: int) -> int:
    """The cheapest ragged block span; ties go to the wider span."""
    return min(
        reversed(varlen_block_spans(num_tokens)),
        key=lambda span: ragged_step_cost(lengths, num_tokens, span),
    )


def use_dense_prefill(lengths: Sequence[int], num_tokens: int, ragged_span: int) -> bool:
    """Whether a fresh prefill step runs the dense kernel: one request, a
    tileable bucket, and a dense cost no higher than the ragged one."""
    if len(lengths) != 1 or not is_dense_tileable(num_tokens):
        return False
    block = min(num_tokens, _DENSE_MAX_BLOCK)
    num_blocks = num_tokens // block
    dense_cost = num_blocks * (num_blocks + 1) // 2 * block * block
    return dense_cost <= ragged_step_cost(lengths, num_tokens, ragged_span)


def extend_kernels(num_tokens: int) -> tuple[str | int, ...]:
    """The EXTEND kernels compiled for a token bucket, in lax.switch order:
    the paged varlen kernel, the dense kernel if the bucket is tileable, and
    the ragged prefill kernel at each block span (an int)."""
    kernels: list[str | int] = [_EXTEND_KERNEL]
    if is_dense_tileable(num_tokens):
        kernels.append(_DENSE_KERNEL)
    kernels.extend(varlen_block_spans(num_tokens))
    return tuple(kernels)


def select_extend_kernel(q_lens: np.ndarray, kv_lens: np.ndarray, num_tokens: int) -> str | int:
    """Pick the EXTEND kernel of the active requests on the host."""
    if q_lens.size == 0 or np.any(kv_lens != q_lens):
        return _EXTEND_KERNEL
    lengths = [int(n) for n in q_lens]
    span = varlen_block_span(lengths, num_tokens)
    return _DENSE_KERNEL if use_dense_prefill(lengths, num_tokens, span) else span


def build_block_table(
    page_indices: np.ndarray,  # i32[num_pages_in_bucket], packed per request
    seq_lens: np.ndarray,  # i32[num_seqs]
    *,
    page_size: int,
    pages_per_seq: int,
) -> np.ndarray:
    """(num_seqs, pages_per_seq) page table from the packed page list.

    The scheduler packs each request's pages back to back, each request
    taking ceil(seq_len / page_size) pages (FlashAttention's cu_kv_lens).
    """
    num_pages = cdiv(seq_lens, page_size).astype(np.int64)
    if np.any(num_pages > pages_per_seq):
        raise ValueError(
            f"A request needs {int(num_pages.max())} pages; the flywheel page table "
            f"holds {pages_per_seq} per request (context length / page size)."
        )
    starts = np.cumsum(num_pages) - num_pages
    table = np.zeros((seq_lens.shape[0], pages_per_seq), dtype=np.int32)
    rows = np.repeat(np.arange(seq_lens.shape[0]), num_pages)
    cols = np.arange(int(num_pages.sum())) - np.repeat(starts, num_pages)
    table[rows, cols] = page_indices[np.repeat(starts, num_pages) + cols]
    return table


def _decode_rows(q, k, v, kv_cache, kv_lens, block_table, num_active, *, sm_scale):
    """One query token per request: the decode kernel appends each request's
    K/V token after its kv_len - 1 cached tokens and attends them."""
    num_seqs = min(block_table.shape[0], q.shape[0])
    output, kv_cache = flash_attn_with_kvcache(
        q[:num_seqs, None],
        kv_cache,
        None,
        k[:num_seqs, None],
        v[:num_seqs, None],
        # Padding rows go to -1, which the kernel never reads (num_active).
        cache_seqlens=kv_lens[:num_seqs] - 1,
        block_table=block_table[:num_seqs],
        num_active=num_active,
        softmax_scale=sm_scale,
        nheads_k=k.shape[1],
    )
    # Rows past num_active are unspecified on single-token decode.
    row = jnp.arange(num_seqs)[:, None, None]
    output = jnp.where(row < num_active, output[:, 0], 0).astype(q.dtype)
    if num_seqs < q.shape[0]:
        output = jnp.pad(output, ((0, q.shape[0] - num_seqs), (0, 0), (0, 0)))
    return output, kv_cache


def _extend_kernel_rows(
    kernel, q, k, v, kv_cache, kv_lens, block_table, cu_q_lens, num_active, *, sm_scale
):
    """One extend_kernels entry. kernel is a Python value, so each lax.switch
    branch traces only its own kernel. Only the paged kernel reads the cache,
    and no branch writes it."""
    if kernel == _EXTEND_KERNEL:
        page_size = kv_cache.shape[1]
        # max_seqlen_q/k are static block-size bounds, so one executable
        # serves every kv_len of a bucket.
        return flash_attn_varlen_func(
            q.transpose(1, 0, 2),
            kv_cache,
            None,
            cu_q_lens,
            None,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=block_table.shape[1] * page_size,
            softmax_scale=sm_scale,
            causal=True,
            token_major=False,
            block_table=block_table,
            seqused_k=kv_lens,
            num_active=num_active,
            nheads_k=k.shape[1],
        ).transpose(1, 0, 2)
    if kernel == _DENSE_KERNEL:
        # The one request fills the front of the bucket; the causal mask keeps
        # its queries off the padded tail.
        num_tokens, num_heads, head_dim = q.shape
        output = flash_attn_func(
            q.transpose(1, 0, 2)[:, None],
            k.transpose(1, 0, 2)[:, None],
            v.transpose(1, 0, 2)[:, None],
            causal=True,
            softmax_scale=sm_scale,
            token_major=False,
            heads_outer_fold=True,
        )
        return output.reshape(num_heads, num_tokens, head_dim).transpose(1, 0, 2)
    # An int: the ragged prefill block span.
    return flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        k.transpose(1, 0, 2),
        v.transpose(1, 0, 2),
        cu_q_lens,
        cu_q_lens,
        max_seqlen_q=kernel,
        max_seqlen_k=kernel,
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
    ).transpose(1, 0, 2)


def _extend_rows(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    block_table,
    cu_q_lens,
    kv_write_slices,
    kv_write_num_slices,
    num_active,
    kernel_index,
    *,
    sm_scale,
):
    """Write the step's K/V rows into the pages, then run the step's kernel."""
    kv_cache = write_kv_cache_pages(
        k, v, kv_cache, kv_write_slices.reshape(3, -1), kv_write_num_slices
    )
    branches = [
        functools.partial(_extend_kernel_rows, kernel, sm_scale=sm_scale)
        for kernel in extend_kernels(q.shape[0])
    ]
    output = lax.switch(
        kernel_index[0],
        branches,
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        block_table,
        cu_q_lens,
        num_active,
    )
    # Rows past the scheduled tokens are unspecified in every kernel.
    row = jnp.arange(q.shape[0])[:, None, None]
    output = jnp.where(row < cu_q_lens[num_active], output, 0).astype(q.dtype)
    return output, kv_cache


@register_pytree_node_class
@dataclass
class FlywheelAttentionMetadata:
    """Per-step metadata, every field data so one executable per bucket serves
    every step."""

    seq_lens: jax.Array = None  # i32[num_seqs], includes this step's tokens
    cu_q_lens: jax.Array = None  # i32[num_seqs + 1]
    block_table: jax.Array = None  # i32[num_seqs, pages_per_seq]
    num_active: jax.Array = None  # i32 scalar: leading rows that are requests
    kv_write_slices: jax.Array = None  # EXTEND only: i32[3 * bound]
    kv_write_num_slices: jax.Array = None  # EXTEND only: i32[1]
    kernel_index: jax.Array = None  # EXTEND only: i32[1], into extend_kernels

    def tree_flatten(self):
        children = (
            self.seq_lens,
            self.cu_q_lens,
            self.block_table,
            self.num_active,
            self.kv_write_slices,
            self.kv_write_num_slices,
            self.kernel_index,
        )
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class FlywheelAttention(AttentionBackend):
    """--attention-backend flywheel: MHA/GQA attention with flywheel-tpu."""

    def __init__(
        self,
        num_attn_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        context_len: int,
        mesh: jax.sharding.Mesh,
        kv_partition_axis: str = "tensor",
        attention_data_partition_axis: str = "data",
    ):
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise NotImplementedError(
                f"--attention-backend flywheel supports head_dim in {SUPPORTED_HEAD_DIMS}; "
                f"got {head_dim}."
            )
        if page_size % PAGE_SIZE_MULTIPLE:
            raise ValueError(
                f"--attention-backend flywheel needs a page size that is a multiple of "
                f"{PAGE_SIZE_MULTIPLE}; got {page_size}."
            )
        if mesh is not None and mesh.shape.get(attention_data_partition_axis, 1) != 1:
            raise NotImplementedError("--attention-backend flywheel supports dp_size=1 only.")
        self.num_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.context_len = context_len
        self.pages_per_seq = cdiv(context_len, page_size)
        self.mesh = mesh
        self.kv_partition_axis = kv_partition_axis
        self.attention_data_partition_axis = attention_data_partition_axis
        self.forward_metadata = nnx.data(FlywheelAttentionMetadata())

    def tree_flatten(self):
        aux_data = {
            "num_attn_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "context_len": self.context_len,
            "mesh": self.mesh,
            "kv_partition_axis": self.kv_partition_axis,
            "attention_data_partition_axis": self.attention_data_partition_axis,
        }
        return (self.forward_metadata,), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(**aux_data)
        obj.forward_metadata = children[0]
        return obj

    def get_forward_metadata(self, batch: ModelWorkerBatch) -> FlywheelAttentionMetadata:
        if batch.dp_size != 1:
            raise NotImplementedError("--attention-backend flywheel supports dp_size=1 only.")
        if batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise NotImplementedError(
                f"--attention-backend flywheel does not support {batch.forward_mode}."
            )
        seq_lens = np.asarray(batch.seq_lens, dtype=np.int32)
        num_seqs = seq_lens.shape[0]
        num_tokens = len(batch.input_ids)
        # Requests come first and padding rows have seq_len 0, as RPA's
        # distribution also assumes.
        num_active = int(np.sum(seq_lens > 0))

        page_indices = (np.asarray(batch.cache_loc)[:: self.page_size] // self.page_size).astype(
            np.int32
        )
        block_table = build_block_table(
            page_indices,
            seq_lens,
            page_size=self.page_size,
            pages_per_seq=self.pages_per_seq,
        )

        metadata = FlywheelAttentionMetadata()
        sharding = NamedSharding(self.mesh, P())
        if batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.arange(num_seqs + 1, dtype=np.int32)
            (
                metadata.seq_lens,
                metadata.cu_q_lens,
                metadata.block_table,
                metadata.num_active,
            ) = device_array(
                (seq_lens, cu_q_lens, block_table, np.int32(num_active)), sharding=sharding
            )
            return metadata

        q_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
        cu_q_lens = np.zeros(num_seqs + 1, dtype=np.int32)
        cu_q_lens[1:] = np.cumsum(q_lens)
        bound = kv_write_slice_bound(num_tokens, num_seqs, self.page_size)
        kv_write_slices = np.zeros((3, bound), dtype=np.int32)
        num_slices = build_kv_write_slices(
            cu_q_lens, seq_lens, block_table, page_size=self.page_size, out=kv_write_slices
        )
        kernel = select_extend_kernel(q_lens[:num_active], seq_lens[:num_active], num_tokens)
        kernel_index = extend_kernels(num_tokens).index(kernel)
        (
            metadata.seq_lens,
            metadata.cu_q_lens,
            metadata.block_table,
            metadata.num_active,
            metadata.kv_write_slices,
            metadata.kv_write_num_slices,
            metadata.kernel_index,
        ) = device_array(
            (
                seq_lens,
                cu_q_lens,
                block_table,
                np.int32(num_active),
                kv_write_slices.ravel(),
                np.array([num_slices], dtype=np.int32),
                np.array([kernel_index], dtype=np.int32),
            ),
            sharding=sharding,
        )
        return metadata

    @named_scope
    def __call__(
        self,
        q: jax.Array,  # [T, num_heads * head_dim] or [T, num_heads, head_dim]
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        causal: int = 1,
        attention_sink: jax.Array = None,
        **kwargs,
    ):
        unsupported = {
            "attention sinks": attention_sink is not None,
            "sliding window": layer.sliding_window_size is not None,
            "logit soft cap": layer.logit_cap is not None,
            "xai temperature": layer.xai_temperature_len > 0,
            "softmax_dtype": layer.softmax_dtype is not None,
            "non-causal attention": not causal
            or getattr(getattr(layer, "attn_type", None), "value", None) == "encoder_only",
            "MSA index keys": kwargs.get("index_k") is not None,
        }
        for name, is_unsupported in unsupported.items():
            if is_unsupported:
                raise NotImplementedError(f"--attention-backend flywheel does not support {name}.")

        head_dim = getattr(layer, "head_dim", self.head_dim)
        scale = 1.0 / float(np.sqrt(head_dim)) if layer.scaling is None else float(layer.scaling)
        kv_cache_5d = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        if kv_cache_5d.dtype != jnp.bfloat16:
            raise NotImplementedError(
                f"--attention-backend flywheel supports a bfloat16 KV cache; got {kv_cache_5d.dtype}."
            )

        md = self.forward_metadata
        is_decode = forward_batch.forward_mode == ForwardMode.DECODE

        data, heads = self.attention_data_partition_axis, self.kv_partition_axis
        qkv_spec = P(data, heads, None)
        cache_spec = P(data, None, heads, None, None)

        def _merged(kv_cache_5d):
            # The pool's [pages, page_size, 2H / packing, packing, D] shard is
            # flywheel's merged [pages, page_size, 2H, D] cache: both order the
            # head rows k0, v0, k1, v1, ...
            pages, page_size, heads_per_pack, packing, dim = kv_cache_5d.shape
            return kv_cache_5d.reshape(pages, page_size, heads_per_pack * packing, dim)

        if is_decode:

            def _attention(q, k, v, kv_cache_5d, seq_lens, block_table, num_active):
                output, kv_cache = _decode_rows(
                    q,
                    k,
                    v,
                    _merged(kv_cache_5d),
                    seq_lens,
                    block_table,
                    num_active,
                    sm_scale=scale,
                )
                return output, kv_cache.reshape(kv_cache_5d.shape)

            args = (md.seq_lens, md.block_table, md.num_active)
        else:

            def _attention(q, k, v, kv_cache_5d, *metadata):
                output, kv_cache = _extend_rows(
                    q, k, v, _merged(kv_cache_5d), *metadata, sm_scale=scale
                )
                return output, kv_cache.reshape(kv_cache_5d.shape)

            args = (
                md.seq_lens,
                md.block_table,
                md.cu_q_lens,
                md.kv_write_slices,
                md.kv_write_num_slices,
                md.num_active,
                md.kernel_index,
            )

        output, kv_cache = jax.shard_map(
            _attention,
            mesh=self.mesh,
            in_specs=(qkv_spec, qkv_spec, qkv_spec, cache_spec) + (P(),) * len(args),
            out_specs=(qkv_spec, cache_spec),
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, head_dim),
            k.reshape(k.shape[0], -1, head_dim),
            v.reshape(v.shape[0], -1, head_dim),
            kv_cache_5d,
            *args,
        )
        return output.reshape(q.shape[0], -1), kv_cache

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # The page table lives in SMEM like RPA's page indices: same budget.
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        if res <= 0:
            raise ValueError(
                f"max running requests {res} must be positive; increase the page size "
                "or decrease the context length."
            )
        return res
