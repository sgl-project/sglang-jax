"""Host-resident PLE table: 320,001,446 rows x 160 bf16 = 95.4 GiB.

It does not fit in v6e HBM, and XLA:TPU cannot gather from a pinned_host
array, so the host gathers 16 rows per token and only [T, 16*160] crosses to
the device. The gather is memory-latency-bound, so _thread_count scales the
thread count with the row count.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

from sgl_jax.srt.layers.ngram_embedding import NGramHashParams, build_hash_params

logger = logging.getLogger(__name__)

_ROWS_PER_THREAD = 1024  # 128 rows -> 1t, 4,096 -> 4t, 131,072 -> 32t
_MAX_THREADS = 32  # past this the box fights itself


def _thread_count(rows: int, cap: int | None = None) -> int:
    cap = _MAX_THREADS if cap is None else cap
    return max(1, min(cap, rows // _ROWS_PER_THREAD))


@dataclass(frozen=True)
class ShardPlacement:
    """Where checkpoint shard ``index`` lands in the full table."""

    index: int
    start: int
    rows: int


def shard_placements(total_rows: int, split_parts: int) -> list[ShardPlacement]:
    """Row range of every ``ngram_embedding.shard_{i}.weight``.

    ``total_rows`` is the padded height: the exporter shards after padding.
    """
    if split_parts <= 0:
        raise ValueError(f"split_ngram_parts must be positive, got {split_parts}")
    shard_size = (total_rows + split_parts - 1) // split_parts
    out = []
    for i in range(split_parts):
        start = i * shard_size
        rows = max(0, min(shard_size, total_rows - start))
        if rows:
            out.append(ShardPlacement(index=i, start=start, rows=rows))
    return out


class NGramTable:
    """data [padded_rows, dim] uint16 -- numpy has no bfloat16 and the gather
    is pure byte movement; :meth:`gather` views the result back to bfloat16."""

    def __init__(self, params: NGramHashParams, dim: int, padded_rows: int | None = None):
        self.params = params
        self.dim = int(dim)
        self.rows = int(params.total_vocab_size)
        self.padded_rows = int(padded_rows) if padded_rows else self.rows
        if self.padded_rows < self.rows:
            raise ValueError(
                f"padded_rows {self.padded_rows} is smaller than the table's {self.rows}"
            )
        self.data = np.zeros((self.padded_rows, self.dim), dtype=np.uint16)
        self._pool: ThreadPoolExecutor | None = None
        logger.info(
            "N-gram table allocated: %s rows x %d = %.1f GiB host RAM",
            f"{self.padded_rows:,}",
            self.dim,
            self.data.nbytes / 2**30,
        )

    # -- gather --------------------------------------------------------------

    def gather(self, ids: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """ids [T, HEADS] -> [T, HEADS*dim] bf16, the batch's ple_embeddings.

        bf16 not uint16: device_array would otherwise put an integer array on
        the device and key_proj would matmul the bit patterns. ``out``, if
        given, is [T, HEADS, dim] uint16 and is reused across steps.
        """
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.shape[1] != self.params.ngram_heads:
            raise ValueError(f"ids must be [T, {self.params.ngram_heads}], got {tuple(ids.shape)}")
        num_tokens, heads = ids.shape  # T, HEADS
        if out is None:
            out = np.empty((num_tokens, heads, self.dim), np.uint16)  # [T, HEADS, dim]
        flat_ids = ids.reshape(-1)  # [T*HEADS]
        flat_out = out.reshape(-1, self.dim)  # [T*HEADS, dim]

        nthreads = _thread_count(flat_ids.shape[0])
        if nthreads == 1:
            np.take(self.data, flat_ids, axis=0, out=flat_out)
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=_MAX_THREADS, thread_name_prefix="ngram-gather"
                )
            bounds = np.linspace(0, flat_ids.shape[0], nthreads + 1).astype(int)
            list(  # np.take drops the GIL, so these actually overlap
                self._pool.map(
                    lambda k: np.take(
                        self.data,
                        flat_ids[bounds[k] : bounds[k + 1]],
                        axis=0,
                        out=flat_out[bounds[k] : bounds[k + 1]],
                    ),
                    range(nthreads),
                )
            )
        return out.reshape(num_tokens, heads * self.dim).view(ml_dtypes.bfloat16)

    # -- checkpoint ----------------------------------------------------------

    def load_shard(self, placement: ShardPlacement, weight: np.ndarray) -> None:
        """Copy one checkpoint shard into its row range."""
        if weight.shape != (placement.rows, self.dim):
            raise ValueError(
                f"shard {placement.index}: expected {(placement.rows, self.dim)}, "
                f"got {tuple(weight.shape)}"
            )
        dst = self.data[placement.start : placement.start + placement.rows]  # [rows, dim]
        dst[:] = weight.view(np.uint16) if weight.dtype != np.uint16 else weight

    @classmethod
    def from_safetensors(
        cls,
        weight_files: dict[str, str | Path],
        config,
        *,
        ple_dense_layer_id: int = 0,
        prefix: str = "model.language_model.layers.1.ple.ple_embedding",
    ) -> NGramTable:
        """Stream the shards into RAM. ``weight_files`` is the checkpoint
        index's weight_map (tensor name -> safetensors file)."""
        from safetensors import safe_open

        params = build_hash_params(
            ngram_size=config.ngram_size,
            heads_per_ngram=config.heads_per_ngram,
            vocab_size=config.vocab_size,
            ngram_vocab_size_base=config.ngram_vocab_size_base,
            eos_token_id=_eos_token_id(config),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)  # 128
        padded = ((params.total_vocab_size + divisor - 1) // divisor) * divisor  # 320,001,536
        dim = int(config.ple_embed_dim) // params.ngram_heads  # 2560 // 16 = 160
        table = cls(params, dim, padded_rows=padded)

        # Group by file so each one is opened once, in index order.
        placements = shard_placements(table.padded_rows, int(config.split_ngram_parts))
        by_file: dict[str, list[ShardPlacement]] = {}
        for p in placements:
            name = f"{prefix}.ngram_embedding.shard_{p.index}.weight"
            if name not in weight_files:
                raise KeyError(f"checkpoint is missing {name}")
            by_file.setdefault(str(weight_files[name]), []).append(p)

        done = 0
        for path, group in by_file.items():
            with safe_open(path, framework="np") as f:
                for p in group:
                    table.load_shard(
                        p, f.get_tensor(f"{prefix}.ngram_embedding.shard_{p.index}.weight")
                    )
                    done += 1
            logger.info("N-gram table: %d/%d shards", done, len(placements))

        table.verify_against_checkpoint(weight_files, prefix=prefix)
        return table

    def verify_against_checkpoint(self, weight_files, *, prefix: str) -> None:
        """Check the derived hash layout against the exporter's buffers. A wrong
        multiplier would silently read valid but wrong rows."""
        from safetensors import safe_open

        expected = {
            "layer_multipliers": self.params.multipliers,
            "ngram_heads_vocab_sizes": self.params.sizes,
            "ngram_heads_offsets": self.params.offsets,
        }
        for short, derived in expected.items():
            name = f"{prefix}.{short}"
            path = weight_files.get(name)
            if path is None:
                logger.warning("checkpoint has no %s; keeping the derived values", name)
                continue
            with safe_open(str(path), framework="np") as f:
                shipped = np.asarray(f.get_tensor(name), dtype=np.int64)
            if not np.array_equal(shipped, derived):
                bad = int(np.argmax(shipped != derived))
                raise ValueError(
                    f"{name} does not match the derived hash layout at index {bad}: "
                    f"checkpoint {shipped[bad]} vs derived {derived[bad]}. "
                    "Every N-gram lookup would silently read the wrong row."
                )


def _eos_token_id(config) -> int:
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    if eos is None:
        raise ValueError("config has no eos_token_id; the N-gram hash needs it as a barrier")
    return int(eos)


# -- process-global handle: one 95 GiB table per process ---------------------

_TABLE: NGramTable | None = None


def set_ngram_table(table: NGramTable | None) -> None:
    """Install the table for the scheduler's host-side lookup. Call once at
    weight load; ``None`` uninstalls it."""
    global _TABLE
    _TABLE = table


def get_ngram_table() -> NGramTable | None:
    """The installed table, or None for every model without a PLE layer."""
    return _TABLE


__all__ = [
    "NGramTable",
    "ShardPlacement",
    "get_ngram_table",
    "set_ngram_table",
    "shard_placements",
]
