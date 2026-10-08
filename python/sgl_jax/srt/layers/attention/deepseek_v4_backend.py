"""Unified V4 attention entry point, host metadata, and functional updates.

Adapted from primatrix/sglang-jax epic/dsv4 at ce1ebb637. The backend
derives a host metadata vector from C's request ledger, dispatches SWA/CSA/HCA,
and packages replacement arrays. R owns device transport, commits, scheduling,
and request-resource reclamation; host resource owners stay outside the model.
"""

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.dsv4.execution import (
    CompressorWeights,
    padded_read_tables,
    run_dsv4_attention,
)
from sgl_jax.srt.layers.attention.dsv4.hca import (
    DeepseekV4HCABackendMixin,
    DeepseekV4HCAMetadata,
)
from sgl_jax.srt.layers.attention.dsv4.metadata import (
    DeepseekV4AttentionMetadata,
    derive_attention_metadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

_PACK_LAYOUTS: dict = {}


def pack_metadata(*trees):
    """Flatten integer/bool pytrees into one int32 vector plus a static layout.

    The per-step attention metadata and read tables are ~60 small arrays. Uploading
    them one by one costs ~230 us each on an 8-device mesh (each leaf is copied to
    every device), ~15 ms per decode step. Packing them into a single vector makes it
    one transfer; `unpack_metadata` restores the pytrees with static slices inside jit.
    """
    leaves, treedef = jax.tree_util.tree_flatten(trees)
    arrays = [np.asarray(leaf) for leaf in leaves]
    # The layout depends only on the (bucketed) shapes and dtypes: compute it once
    # per signature instead of re-deriving ~60 specs (and dtype names) every step.
    key = (treedef, tuple((a.shape, a.dtype.str) for a in arrays))
    entry = _PACK_LAYOUTS.get(key)
    if entry is None:
        specs, offset = [], 0
        for array in arrays:
            if array.dtype == np.bool_:
                kind = "bool"
            elif np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize <= 4:
                kind = str(array.dtype)
            else:
                raise TypeError(f"metadata leaf dtype {array.dtype} cannot be packed as int32")
            specs.append((offset, array.size, tuple(array.shape), kind))
            offset += array.size
        if len(_PACK_LAYOUTS) >= 1024:
            _PACK_LAYOUTS.clear()
        entry = _PACK_LAYOUTS[key] = (tuple(specs), offset)
    specs, total = entry
    packed = np.empty((total,), np.int32)
    for array, (offset, size, _, _) in zip(arrays, specs):
        packed[offset : offset + size] = array.reshape(-1)
    return packed, (treedef, specs)


def unpack_metadata(packed, layout):
    """Inverse of `pack_metadata`; works on host arrays and inside jit."""
    treedef, specs = layout
    leaves = []
    for offset, size, shape, kind in specs:
        leaf = jax.lax.slice_in_dim(packed, offset, offset + size).reshape(shape)
        if kind == "bool":
            leaf = leaf != 0
        elif kind != "int32":
            leaf = leaf.astype(kind)
        leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, leaves)


@jax.tree_util.register_pytree_node_class
@dataclass
class DeepseekV4RuntimeMetadata(DeepseekV4HCAMetadata):
    # Each leaf concatenates equal-sized DP-local arrays. Indices remain local
    # to a rank, including cu_q_lens and the compression-boundary sentinels.
    attention: DeepseekV4AttentionMetadata | None = None
    read_tables: tuple = ()
    # When set, `attention`/`read_tables` travel as one int32 vector (see pack_metadata)
    # and `resolve()` rebuilds them; `layout` is static.
    packed: jax.Array | None = None
    layout: tuple | None = None
    # Target sharding for the shared runtime transfer path. The backend's
    # metadata producer returns a host vector and never submits a transfer.
    sharding: NamedSharding | None = None

    def has_metadata(self) -> bool:
        return self.packed is not None or (self.attention is not None and bool(self.read_tables))

    def _unpacked(self):
        """All packed trees, unpacked once per instance (== once per trace)."""
        cached = self.__dict__.get("_unpacked_trees")
        if cached is None:
            packed = self.packed
            if self.sharding is not None and isinstance(packed, jax.Array):
                packed = jax.sharding.reshard(packed, self.sharding)
            cached = tuple(unpack_metadata(packed, self.layout))
            self.__dict__["_unpacked_trees"] = cached
        return cached

    def hca_metadata(self, mesh=None):
        """The HCA view ``(kernel, schedule, uniform, state_init_slots)`` of this step.

        Leaves unpacked inside the jitted step come out replicated; the HCA shard_map
        expects every metadata leaf on ``P("data")`` (the spec the host upload used),
        so reshard them when a mesh is given (a no-op layout for dp == 1).
        """
        from sgl_jax.srt.layers.attention.dsv4.hca import DeepseekV4HCAMetadata

        if self.packed is None or len(self._unpacked()) < 4:
            return DeepseekV4HCAMetadata(
                self.kernel,
                self.schedule,
                self.use_uniform_prefill_fast_path,
                self.state_init_slots,
            )
        cached = self.__dict__.get("_hca_view")
        if cached is None:
            trees = self._unpacked()
            kernel, init_slots = trees[2], trees[3]
            if mesh is not None:
                sharding = NamedSharding(mesh, P("data"))
                kernel = jax.tree.map(lambda a: jax.sharding.reshard(a, sharding), kernel)
                init_slots = jax.sharding.reshard(init_slots, sharding)
            cached = DeepseekV4HCAMetadata(
                kernel, self.schedule, self.use_uniform_prefill_fast_path, init_slots
            )
            self.__dict__["_hca_view"] = cached
        return cached

    def resolve(self):
        """``(attention, read_tables)`` whether or not the metadata is packed.

        The unpacked tree is memoized on the instance: every decoder layer calls the
        backend with the same metadata object, so without the memo each layer re-emits
        the ~60 static slices (thousands of small copy ops per step at 64 layers).
        The memo never crosses a trace: jit rebuilds the object per trace and the host
        builds a new one per step.
        """
        if self.packed is None:
            return self.attention, self.read_tables
        trees = self._unpacked()
        return trees[0], tuple(trees[1])

    def tree_flatten(self):
        return (
            self.kernel,
            self.state_init_slots,
            self.attention,
            self.read_tables,
            self.packed,
        ), (self.schedule, self.use_uniform_prefill_fast_path, self.layout, self.sharding)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(
            children[0],
            aux[0],
            aux[1],
            children[1],
            children[2],
            children[3],
            children[4],
            aux[2],
            aux[3] if len(aux) > 3 else None,
        )


class _PrecompileContextBox:
    """Mutable holder that hashes by identity so its value stays out of jit cache keys."""

    __slots__ = ("context_len", "capacities")

    def __init__(self):
        self.context_len: int | None = None
        self.capacities: tuple[int, int, int] | None = None


class DeepseekV4AttentionBackend(AttentionBackend, DeepseekV4HCABackendMixin):
    """Model-facing owner of V4 metadata, layer routing, and cache updates.

    Like SGLang's V4 backend, one entry point handles ratios 0/4/128. HCA helpers
    are mixed into this backend and share its configuration and metadata owner.
    Unlike PyTorch, JAX returns replacement cache arrays through the model jit.
    """

    def __init__(self, *, mesh, page_size, max_context_len, config=None):
        if mesh is None:
            raise ValueError("V4 attention requires the SGLang device mesh")
        if page_size not in (128, 256):
            raise ValueError("V4 attention requires original-token page size 128 or 256")
        if max_context_len <= 0:
            raise ValueError("V4 context capacity must be positive")
        self.mesh = mesh
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.request_capacity = 1
        self.window_size = int(getattr(config, "sliding_window", 128))
        # This selects a shape-specialized TPU implementation, not a different
        # attention algorithm. The general ratio-128 path uses all visible C128
        # records without an indexer, and remains reachable for other geometries.
        from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

        self.use_pallas_hca = is_tpu_runtime(mesh) and (
            getattr(config, "hidden_size", 4096),
            getattr(config, "num_attention_heads", 64),
            getattr(config, "head_dim", 512),
            getattr(config, "qk_rope_head_dim", 64),
            self.window_size,
        ) == (4096, 64, 512, 64, 128)
        self.resources_bound = False
        self.forward_metadata = nnx.data(DeepseekV4RuntimeMetadata())
        # Set by the compilation manager while precompiling dummy batches: derive the
        # read-table capacity buckets from this context length instead of the (zero)
        # dummy sequence lengths, so every power-of-two bucket a real request can reach
        # is compiled at startup rather than on first use (~48 s per bucket on v7x).
        # Kept in an identity-hashed box: a plain attribute would become part of the
        # nnx graphdef and therefore of the jit cache key, so precompiled executables
        # (context_len=N) would never match runtime calls (context_len=None) and every
        # first use would still re-trace (~6 s each with the persistent cache).
        self._precompile_box = _PrecompileContextBox()

    @property
    def precompile_context_len(self) -> int | None:
        return self._precompile_box.context_len

    @precompile_context_len.setter
    def precompile_context_len(self, value: int | None) -> None:
        self._precompile_box.context_len = value

    @property
    def precompile_capacity_override(self):
        return self._precompile_box.capacities

    @precompile_capacity_override.setter
    def precompile_capacity_override(self, value):
        # R can prepare total-history and per-request HCA buckets independently.
        # This host-only override stays outside executable/nnx graph keys.
        self._precompile_box.capacities = value

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # TpWorker combines this kernel metadata limit with the actual request
        # pool capacity. Reuse the HCA scalar-prefetch budget for mixed V4 layers.
        pages_per_request = (max_context_len + page_size - 1) // page_size
        return max(1, 1024 * 1024 // 2 // pages_per_request // 4)

    def bind_resources(self, request_pool, allocator):
        if allocator.dp_size != self.mesh.shape["data"] or allocator.page_size != self.page_size:
            raise ValueError("V4 runtime and resource geometry disagree")
        self.request_capacity = request_pool.size
        self.resources_bound = True

    def hca_entry_bucket(self, batch) -> int | None:
        """Ratio-128 capacity bucket that sizes the HCA compressed tile.

        Uses the same power-of-two rule and precompile-ladder override as the
        ratio-128 read table, so the kernel variant set equals the read-table
        bucket set.  Returns ``None`` (whole-context sizing) when disabled.
        """
        if os.environ.get("DSV4_HCA_TILE_BUCKET", "1") != "1":
            return None
        if self.precompile_capacity_override is not None:
            return self.precompile_capacity_override[2]
        if self.precompile_context_len is not None:
            return precompile_capacities(self.precompile_context_len)[128]
        dp = int(self.mesh.shape["data"])
        lengths = np.asarray(batch.seq_lens, np.int32).reshape(dp, -1)
        if batch.forward_mode == ForwardMode.DECODE:
            queries = (lengths > 0).astype(np.int32)
        else:
            queries = np.asarray(batch.extend_seq_lens, np.int32).reshape(dp, -1)
        per_request = np.where(queries > 0, lengths // 128, 0)
        if os.environ.get("DSV4_HCA_TILE_BUCKET_SUM", "0") == "1":
            # Previous behaviour: bucket the batch total. The compressed tile is a
            # per-request, per-token quantity, so this over-sized decode tiles
            # (64 x 9K requests -> 4608 records -> a 2048+ tile with 72 live
            # entries) and never matched the per-request precompile ladder.
            count = int(np.max(np.sum(per_request, axis=1)))
        else:
            count = int(np.max(per_request))
        return capacity_bucket(count)

    def get_forward_metadata(self, batch, *, request_pool, allocator):
        if not self.resources_bound:
            raise RuntimeError("V4 runtime resources must be bound after pool initialization")
        hca = (
            self._get_hca_metadata(
                batch,
                request_pool=request_pool,
                allocator=allocator,
                fixed_bucket=True,
                device=False,
                max_compressed_entries=self.hca_entry_bucket(batch),
            )
            if self.use_pallas_hca
            else DeepseekV4HCAMetadata()
        )
        dp = int(self.mesh.shape["data"])
        lengths = np.asarray(batch.seq_lens, np.int32).reshape(dp, -1)
        slots = np.asarray(batch.req_pool_indices, np.int32).reshape(dp, -1)
        positions = np.asarray(batch.positions, np.int32).reshape(dp, -1)
        history = np.asarray(batch.out_cache_loc, np.int32).reshape(dp, -1)
        if history.shape != positions.shape:
            raise ValueError("V4 output addresses must match the padded query token axis")
        queries = (
            (lengths > 0).astype(np.int32)
            if batch.forward_mode == ForwardMode.DECODE
            else np.asarray(batch.extend_seq_lens, np.int32).reshape(dp, -1)
        )
        if batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise ValueError("V4 supports ordinary EXTEND and DECODE only")
        active = lengths > 0
        if np.any(lengths < 0) or np.any(lengths > self.max_context_len):
            raise ValueError("V4 request length exceeds the configured context")
        if np.any(queries < 0) or np.any(queries > lengths) or np.any((queries > 0) != active):
            raise ValueError("invalid V4 query lengths")
        if np.any(queries.sum(axis=1) > positions.shape[1]):
            raise ValueError("V4 queries exceed the padded token capacity")
        if np.any(slots[active] < 0) or np.any(slots[active] >= self.request_capacity):
            raise ValueError("active V4 requests require allocated request slots")
        if len(np.unique(slots[active])) != active.sum():
            raise ValueError("active V4 requests must own distinct slots")
        if batch.forward_mode == ForwardMode.EXTEND and not np.array_equal(
            np.asarray(batch.extend_prefix_lens).reshape(lengths.shape), lengths - queries
        ):
            raise ValueError("V4 prefix plus query length must equal sequence length")
        local = []
        tables_by_ratio = {ratio: [] for ratio in (0, 4, 128)}
        # All DP ranks must have identical local array extents for shard_map.
        # Only requests with queries contribute rows to read_tables; bound their
        # completed groups with a power-of-two bucket shared across ranks.
        compressed_capacities = {0: 1}
        for ratio in (4, 128):
            count = int(np.max(np.sum(np.where(queries > 0, lengths // ratio, 0), axis=1)))
            compressed_capacities[ratio] = capacity_bucket(count)
        decode_capacity = None
        if batch.forward_mode == ForwardMode.DECODE and self.page_size == 128:
            decode_capacity = capacity_bucket(int(np.max(lengths // 4)))
        if self.precompile_context_len is not None:
            ladder = precompile_capacities(self.precompile_context_len)
            compressed_capacities.update(ladder)
            if decode_capacity is not None:
                decode_capacity = ladder[4]
        if self.precompile_capacity_override is not None:
            c4, c128, _ = self.precompile_capacity_override
            compressed_capacities.update({4: c4, 128: c128})
            if decode_capacity is not None:
                decode_capacity = c4
        for rank in range(dp):
            live = int(queries[rank].sum())
            mapping = allocator.full_to_swa_index_mapping
            mapping = mapping[rank] if isinstance(mapping, list) else mapping
            writes = history[rank, :live]
            if np.any((writes < self.page_size) | (writes >= len(mapping))):
                raise ValueError("V4 live query addresses must name allocated original-token slots")
            prefixes = lengths[rank] - queries[rank]
            expected = np.concatenate(
                [
                    request_pool.req_to_token[slot, pre:end]
                    for slot, pre, end, n in zip(
                        slots[rank], prefixes, lengths[rank], queries[rank], strict=True
                    )
                    if n
                ]
                or [np.empty(0, np.int32)]
            )
            if not np.array_equal(writes, expected) or np.any(mapping[writes] == 0):
                raise ValueError("V4 query writes disagree with the request/SWA ownership map")
            swa = np.full(positions.shape[1], -1, np.int32)
            swa[:live] = mapping[writes]
            for ratio in tables_by_ratio:
                tables_by_ratio[ratio].append(
                    padded_read_tables(
                        request_pool=request_pool,
                        allocator=allocator,
                        slots=slots[rank],
                        lengths=lengths[rank],
                        q_lens=queries[rank],
                        ratio=ratio,
                        window_size=self.window_size,
                        page_size=self.page_size,
                        max_context_len=self.max_context_len,
                        token_capacity=positions.shape[1],
                        rank=rank,
                        compressed_capacity=compressed_capacities[ratio],
                        decode_capacity=decode_capacity if ratio == 4 else None,
                        minimal=ratio == 128 and self.use_pallas_hca,
                    )
                )
            local.append(
                derive_attention_metadata(
                    q_lens=queries[rank],
                    prefix_lens=prefixes,
                    positions=positions[rank],
                    request_slots=slots[rank],
                    history_write_loc=history[rank],
                    swa_write_loc=swa,
                    pages_per_request=(lengths[rank] + self.page_size - 1) // self.page_size,
                    page_size=self.page_size,
                    window_size=self.window_size,
                    state_init_mask=(queries[rank] > 0) & (prefixes == 0),
                )
            )
        sharding = NamedSharding(self.mesh, P("data"))
        # Concatenate rank-local sections into a single host vector. R places this
        # vector in the shared transfer path; this producer performs no device I/O.
        host_attention = jax.tree.map(lambda *arrays: np.concatenate(arrays), *local)
        host_tables = tuple(
            jax.tree.map(lambda *arrays: np.concatenate(arrays), *tables)
            for tables in tables_by_ratio.values()
        )
        # The HCA kernel table (14 leaves) and the init slots ride in the same vector;
        # `hca_metadata()` rebuilds the HCA view inside the jitted step.
        packed_host, layout = pack_metadata(
            host_attention, host_tables, hca.kernel, hca.state_init_slots
        )
        return DeepseekV4RuntimeMetadata(
            None,
            hca.schedule,
            hca.use_uniform_prefill_fast_path,
            None,
            None,
            (),
            packed_host,
            layout,
            sharding,
        )

    def layer_ratio(self, layer, token_to_kv_pool) -> int:
        """Compression ratio of a layer, from C1's spec -- the single classification.

        C1's `spec.compress_ratios` is the same list `configs/deepseek_v4.classify_layers`
        reads, so this does not add a third derivation.
        """
        layer_id = int(layer.layer_id)
        ratios = token_to_kv_pool.spec.compress_ratios
        if not 0 <= layer_id < len(ratios):
            raise ValueError(f"layer {layer_id} is outside the V4 backbone")
        return int(ratios[layer_id])

    def __call__(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        token_to_kv_pool,
        *,
        compressor_state_pool,
        compressor_input=None,
        compressor=None,
        indexer=None,
        attention_sink=None,
        rope_head_dim=64,
        norm_eps=1e-6,
        index_topk=None,
        compressor_input_local=False,
        **kwargs,
    ):
        ratio = self.layer_ratio(layer, token_to_kv_pool)
        if compressor_input_local and ratio != 4:
            raise ValueError(
                "a row-local compressor input is only supported on CSA (ratio 4) layers"
            )
        if compressor is None and "wkv" in kwargs:
            # The standalone HCA interface also accepts separate cosine/sine tables.
            compressor = CompressorWeights(
                kwargs["wkv"],
                kwargs["wgate"],
                kwargs["ape"],
                kwargs["norm_weight"],
                jnp.concatenate((kwargs["cos"], kwargs["sin"]), axis=-1),
            )
        if ratio == 128 and self.use_pallas_hca:
            if compressor is None:
                raise ValueError("HCA requires model compressor weights")
            cache = compressor.cos_sin_cache
            output, (state, window, history) = self._forward_hca(
                q,
                k,
                v,
                layer,
                forward_batch,
                token_to_kv_pool,
                compressor_state_pool=compressor_state_pool,
                compressor_input=compressor_input,
                wkv=compressor.wkv,
                wgate=compressor.wgate,
                ape=compressor.ape,
                norm_weight=compressor.norm_weight,
                cos=(
                    compressor.cos_table
                    if getattr(compressor, "cos_table", None) is not None
                    else cache[:, : cache.shape[-1] // 2]
                ),
                sin=(
                    compressor.sin_table
                    if getattr(compressor, "sin_table", None) is not None
                    else cache[:, cache.shape[-1] // 2 :]
                ),
                attention_sink=attention_sink,
                fused_weight=getattr(compressor, "fused", None),
                metadata=self.forward_metadata.hca_metadata(self.mesh),
            )
            return output.reshape(q.shape), {"state": state, "swa": window, "compressed": history}
        md = self.forward_metadata
        if not md.has_metadata():
            raise RuntimeError("V4 attention metadata has not been prepared")
        attention, read_tables = md.resolve()
        return run_dsv4_attention(
            self.mesh,
            q,
            k[:, 0] if k.ndim == 3 else k,
            hidden_states=compressor_input,
            layer_id=int(layer.layer_id),
            ratio=ratio,
            metadata=attention,
            tables=read_tables[(0, 4, 128).index(ratio)],
            token_to_kv_pool=token_to_kv_pool,
            compressor_state_pool=compressor_state_pool,
            compressor=compressor,
            indexer=indexer,
            attention_sink=attention_sink,
            softmax_scale=float(layer.scaling),
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
            index_topk=index_topk,
            hidden_local=compressor_input_local,
        )

    @staticmethod
    def pack_pool_updates(layer_updates, token_to_kv_pool, compressor_state_pool):
        kv_updates, state_updates = {}, {}
        for layer_id, updates in layer_updates.items():
            kv_updates[layer_id], state_updates[layer_id] = {}, {}
            for name, array in updates.items():
                if name in ("state", "indexer_state"):
                    resource = "compressor" if name == "state" else "indexer"
                    state_updates[layer_id][resource] = array
                else:
                    kv_updates[layer_id][name] = array
        return {
            "token_to_kv_pool": token_to_kv_pool.build_buffer_updates(kv_updates),
            "compressor_state_pool": compressor_state_pool.build_buffer_updates(state_updates),
        }


def capacity_bucket(count: int) -> int:
    """Power-of-two read-table capacity for ``count`` completed entries (minimum 128)."""
    return max(128, 1 << (max(1, count) - 1).bit_length())


def precompile_capacities(context_len: int) -> dict[int, int]:
    """Capacity buckets a request of ``context_len`` tokens reaches, per compression ratio."""
    return {ratio: capacity_bucket(context_len // ratio) for ratio in (4, 128)}
