"""A fixed paged embedding buffer; lengths and page locations are host data."""

from __future__ import annotations

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.transfer_layout import (
    ENCODER_PAGE_SIZE,
    encoder_pool_block_shape,
)

logger = logging.getLogger(__name__)


def _write_rows(pool, embeddings, destination_rows, source_rows):
    rows = embeddings[jnp.maximum(source_rows, 0)]
    rows = jnp.where((source_rows >= 0)[:, None], rows, 0)
    page_size = pool.shape[1]
    width = math.prod(pool.shape[2:])
    rows = jnp.pad(rows, ((0, 0), (0, width - embeddings.shape[1])))
    # Destinations contain complete pages, including zeroed request tails.
    # Convert only the updates; flattening the pool reformats the entire buffer.
    pages = rows.reshape(-1, *pool.shape[1:])
    page_ids = destination_rows[::page_size] // page_size
    pool = pool.at[page_ids].set(pages, mode="drop")
    # A separate completion value survives the next donation of the pool.
    row = jnp.minimum(destination_rows[0], pool.shape[0] * page_size - 1)
    ready = pool[row // page_size, row % page_size, 0, 0, 0]
    return pool, ready


class RaidenPool:
    """Host page allocation plus one registered buffer. Callers serialize ownership."""

    def __init__(self, shape, dtype, sharding, *, capacity: int, max_batch_size: int = 8):
        self.page_size, self.width = map(int, shape)
        if self.page_size < 2 or capacity <= 0:
            raise ValueError("Raiden needs page_size >= 2 and positive page capacity")
        self.dtype = jnp.dtype(dtype)
        self.sharding = sharding
        self.num_pages = int(capacity)
        self.max_batch_size = max(1, int(max_batch_size))
        self._free_pages = list(range(self.num_pages - 1, -1, -1))
        self.buffer = jnp.zeros(
            (self.num_pages, *encoder_pool_block_shape(shape)),
            self.dtype,
            device=sharding,
        )
        jax.block_until_ready(self.buffer)
        self.num_shards = 1 if sharding.is_fully_replicated else len(sharding.device_set)
        writer = _write_rows
        if self.num_shards > 1:
            # Keep global indexing at the API boundary and execute only local writes.
            def write_local(pool, embeddings, destinations, sources):
                rank = 0
                for axis in sharding.mesh.axis_names:
                    rank = rank * sharding.mesh.shape[axis] + jax.lax.axis_index(axis)
                destinations = destinations - rank * pool.shape[0] * self.page_size
                sources = jnp.where(sources >= 0, sources - rank * embeddings.shape[0], -1)
                pool, ready = _write_rows(pool, embeddings, destinations, sources)
                return pool, ready.reshape(1)

            writer = jax.shard_map(
                write_local,
                mesh=sharding.mesh,
                in_specs=(sharding.spec,) * 4,
                out_specs=(sharding.spec, sharding.spec),
                check_vma=False,
            )
        self._write = jax.jit(writer, donate_argnums=(0,))

    @property
    def available_pages(self) -> int:
        return len(self._free_pages)

    def pages_needed(self, tokens: int) -> int:
        if tokens <= 0 or tokens > self.num_pages * self.page_size:
            raise ValueError("Encoder request exceeds the pool token capacity")
        return (tokens + self.page_size - 1) // self.page_size

    def allocate(self, tokens: int, *, shard: int | None = None) -> tuple[int, ...] | None:
        count = self.pages_needed(tokens)
        if shard is not None:
            size = self.num_pages // len(self.sharding.device_set)
            pages = tuple(page for page in reversed(self._free_pages) if page // size == shard)[
                :count
            ]
            if len(pages) != count:
                return None
            chosen = set(pages)
            self._free_pages = [page for page in self._free_pages if page not in chosen]
            return pages
        if count > self.available_pages:
            return None
        return tuple(self._free_pages.pop() for _ in range(count))

    def available_pages_by_shard(self) -> list[int]:
        size = self.num_pages // len(self.sharding.device_set)
        return np.bincount(
            np.asarray(self._free_pages, np.int32) // size,
            minlength=len(self.sharding.device_set),
        ).tolist()

    def release(self, page_ids: tuple[int, ...]) -> None:
        self._free_pages.extend(reversed(page_ids))

    def _write_capacity(self, token_capacity: int) -> int:
        # At most one partial tail per request. No independent page-count bucket.
        local_capacity = token_capacity // self.num_shards
        requests = self.max_batch_size
        pages = (local_capacity + self.page_size - 1) // self.page_size + requests - 1
        return min(self.num_pages, pages * self.num_shards) * self.page_size

    def warmup(self, token_capacity: int) -> None:
        """Warm one writer per ViT capacity, before registering the buffer with Raiden."""
        embeddings = jnp.zeros((token_capacity, self.width), self.dtype, device=self.sharding)
        capacity = self._write_capacity(token_capacity)
        destinations = jax.device_put(
            np.full(capacity, self.num_pages * self.page_size, np.int32), self.sharding
        )
        sources = jax.device_put(np.full(capacity, -1, np.int32), self.sharding)
        compiled = self._write.lower(self.buffer, embeddings, destinations, sources).compile()
        stats = compiled.memory_analysis()
        if (
            stats is None
            or stats.alias_size_in_bytes <= 0
            or stats.alias_size_in_bytes * 100 < stats.output_size_in_bytes * 99
        ):
            raise RuntimeError("Raiden embedding writer must alias its donated buffer")
        self.buffer, ready = self._write(self.buffer, embeddings, destinations, sources)
        jax.block_until_ready((self.buffer, ready))
        logger.info("Encoder pool warmed: token_capacity=%d", token_capacity)

    def write(self, embeddings, allocations, token_counts, *, source_rows=None) -> jax.Array:
        if (
            len(allocations) != len(token_counts)
            or not allocations
            or len(allocations) > self.max_batch_size * self.num_shards
            or embeddings.ndim != 2
            or embeddings.shape[1] != self.width
            or embeddings.dtype != self.dtype
            or not embeddings.sharding.is_equivalent_to(self.sharding, 2)
            or sum(token_counts) > embeddings.shape[0]
        ):
            raise ValueError("Encoder output does not match its page allocations")
        capacity = self._write_capacity(embeddings.shape[0])
        destinations = np.full(capacity, self.num_pages * self.page_size, np.int32)
        sources = np.full(capacity, -1, np.int32)
        row_offsets = np.arange(self.num_shards) * (capacity // self.num_shards)
        token_offset = 0
        for pages, tokens in zip(allocations, token_counts, strict=True):
            if len(pages) != self.pages_needed(tokens):
                raise ValueError("Encoder allocation does not match its token count")
            rank = pages[0] // (self.num_pages // self.num_shards)
            row_offset = row_offsets[rank]
            end = row_offset + len(pages) * self.page_size
            destinations[row_offset:end] = (
                np.asarray(pages, np.int32)[:, None] * self.page_size + np.arange(self.page_size)
            ).reshape(-1)
            # Zero the page tail as well, so no stale rows are exposed by transport.
            sources[row_offset : row_offset + tokens] = (
                np.arange(token_offset, token_offset + tokens, dtype=np.int32)
                if source_rows is None
                else source_rows[token_offset : token_offset + tokens]
            )
            row_offsets[rank] = end
            token_offset += tokens
        self.buffer, ready = self._write(
            self.buffer,
            embeddings,
            jax.device_put(destinations, self.sharding),
            jax.device_put(sources, self.sharding),
        )
        return ready


def create_encoder_pool(server_args, model_config, mesh) -> RaidenPool:
    vision = getattr(model_config.hf_config, "vision_config", None)
    width = model_config.hidden_size * (1 + len(getattr(vision, "deepstack_visual_indexes", ())))
    shards = mesh.size
    pages = math.ceil(server_args.encoder_transfer_max_tokens / (ENCODER_PAGE_SIZE * shards))
    spec = jax.sharding.PartitionSpec(tuple(mesh.axis_names))
    return RaidenPool(
        (ENCODER_PAGE_SIZE, width),
        model_config.dtype,
        jax.sharding.NamedSharding(mesh, spec),
        capacity=pages * shards,
        max_batch_size=server_args.encoder_max_batch_size,
    )
