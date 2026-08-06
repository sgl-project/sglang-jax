from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from sgl_jax.srt.utils.common_utils import (
    PRECOMPILE_DEFAULT_BS_PADDINGS,
    PRECOMPILE_DEFAULT_TOKEN_PADDINGS,
)

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CompilationManager:
    """Owns bucket computation, dummy batch construction, and pre-compilation."""

    def __init__(
        self,
        server_args: ServerArgs,
        max_padded_batch_size: int,
        max_padded_num_tokens: int,
        dp_size: int,
        tp_size: int,
        page_size: int,
        max_req_len: int,
        vocab_size: int,
        max_total_num_tokens: int = 0,
        precompile_in_model_multimodal: bool = False,
        capture_hidden_states: bool = False,
        has_recurrent_state: bool = False,
        supports_recurrent_cow: bool = False,
        supports_recurrent_track: bool = False,
        moe_backend: str | None = None,
    ):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.page_size = page_size
        self.max_req_len = max_req_len
        self.max_total_num_tokens = max_total_num_tokens
        self.max_padded_batch_size = max_padded_batch_size
        self.max_padded_num_tokens = max_padded_num_tokens
        self.vocab_size = vocab_size
        self.precompile_in_model_multimodal = precompile_in_model_multimodal
        self.capture_hidden_states = capture_hidden_states
        self.has_recurrent_state = has_recurrent_state
        self.supports_recurrent_cow = supports_recurrent_cow
        self.supports_recurrent_track = supports_recurrent_track
        # Callers pass the *effective* backend (ModelConfig.moe_backend), which
        # resolves architectures that hard-code FusedEPMoE (e.g. Qwen3.5) to
        # "fused" so the bs-bucket filter below applies. Fall back to the raw
        # server_args string for callers that don't have a ModelConfig yet.
        self.moe_backend = moe_backend if moe_backend is not None else server_args.moe_backend
        self.enable_static_lora = server_args.enable_static_lora

        self.token_buckets = self._compute_token_buckets(server_args.precompile_token_paddings)
        self.bs_buckets = self._compute_bs_buckets(server_args.precompile_bs_paddings)
        self.cache_loc_buckets = self._compute_cache_loc_buckets()
        self._compiled_variants: set[tuple] = set()

    def _compute_token_buckets(self, user_paddings: list[int] | None) -> list[int]:
        dp_size = self.dp_size
        if user_paddings is None:
            user_paddings = [item * dp_size for item in PRECOMPILE_DEFAULT_TOKEN_PADDINGS]

        buckets = []
        for item in user_paddings:
            if item % dp_size != 0:
                item = (item // dp_size) * dp_size
            if (
                item >= self.max_padded_batch_size
                and item <= self.max_padded_num_tokens
                and item >= dp_size
            ):
                buckets.append(item)

        buckets.sort()
        if len(buckets) == 0 or buckets[-1] < self.max_padded_num_tokens:
            buckets.append(self.max_padded_num_tokens)

        return buckets

    def _compute_bs_buckets(self, user_paddings: list[int] | None) -> list[int]:
        bs_list = user_paddings if user_paddings is not None else PRECOMPILE_DEFAULT_BS_PADDINGS
        is_fused_moe = self.moe_backend in ("fused", "fused_v2")
        min_fused_bs = self.tp_size * 2
        if is_fused_moe and self.max_padded_batch_size < min_fused_bs:
            raise ValueError(
                f"max_padded_batch_size={self.max_padded_batch_size} is below the fused-MoE "
                f"minimum 2 * mesh_ep_size={min_fused_bs}. Increase --max-running-requests "
                "or reduce the EP group size."
            )

        buckets = []
        for bs in bs_list:
            if (
                bs <= self.max_padded_batch_size
                and (not is_fused_moe or bs >= min_fused_bs)
                and bs >= self.dp_size
            ):
                buckets.append(bs)
        buckets.sort()
        if len(buckets) == 0 or buckets[-1] < self.max_padded_batch_size:
            buckets.append(self.max_padded_batch_size)
        return buckets

    def _compute_cache_loc_buckets(self) -> list[int]:
        # bs reqs together can never exceed max_total_num_tokens, so cap the
        # per-bs bucket at the pool size (helps Pathways gRPC H2D; see tp_worker
        # for why the cap is proxy-only).
        pages_per_req = (self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
        pool_aligned = (
            (self.max_total_num_tokens + self.page_size - 1) // self.page_size * self.page_size
            if self.max_total_num_tokens
            else None
        )
        return [
            min(bs * pages_per_req, pool_aligned) if pool_aligned else bs * pages_per_req
            for bs in self.bs_buckets
        ]

    def _extend_variant_names(self) -> tuple[str, ...]:
        variants = ["text"]
        if self.precompile_in_model_multimodal:
            variants.append("multimodal")
        return tuple(variants)

    @staticmethod
    def _populate_dummy_multimodal_inputs(batch, model_runner: ModelRunner) -> None:
        """Populate the array leaves produced by the runtime vision merge.

        The values are irrelevant for compilation. Shapes, dtypes, and shardings
        are established by ``ForwardBatch.init_new`` to match real VLM EXTEND
        batches.
        """
        hidden_size = model_runner.model_config.hidden_size
        num_tokens = len(batch.input_ids)
        dtype = np.dtype(model_runner.model_config.dtype)
        batch.input_embedding = np.zeros((num_tokens, hidden_size), dtype=dtype)

        deepstack_layers = getattr(model_runner.model, "deepstack_visual_layers", 0)
        if isinstance(deepstack_layers, int) and deepstack_layers > 0:
            batch.deepstack_visual_embedding = np.zeros(
                (deepstack_layers, num_tokens, hidden_size),
                dtype=dtype,
            )
            # Real Qwen3-VL requests carry True. Keeping the dummy identical also
            # exercises the DeepStack addition branch while adding only zeros.
            batch.apply_for_deepstack = True

    @staticmethod
    def _packed_multimodal_shapes(model_runner: ModelRunner) -> tuple[tuple[int, int], ...]:
        getter = getattr(
            model_runner.model,
            "get_multimodal_embedding_packed_shapes",
            None,
        )
        if not callable(getter):
            return ()
        shapes = tuple((int(rows), int(cap)) for rows, cap in getter())
        if any(rows <= 0 or cap <= 0 for rows, cap in shapes):
            raise ValueError(f"invalid multimodal packed shapes: {shapes}")
        return shapes

    @staticmethod
    def _embedding_pool(model_runner: ModelRunner):
        from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool

        pool = getattr(model_runner, "embedding_pool", None)
        return pool if isinstance(pool, EmbeddingPool) else None

    @staticmethod
    def _warm_multimodal_merge(
        forward_batch,
        input_embedding: Callable,
        packed_shapes: tuple[tuple[int, int], ...] = (),
        embedding_pool=None,
        mesh=None,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPoolEntry
        from sgl_jax.srt.multimodal.in_model.host_orchestration import (
            ItemTask,
            _gather_from_pool,
            _gather_merge,
            _MergeMapping,
        )
        from sgl_jax.srt.multimodal.in_model.interface import PackedMultimodalEmbedding

        if mesh is None:
            mesh = getattr(getattr(forward_batch.input_ids, "sharding", None), "mesh", None)
        with jax.set_mesh(mesh) if mesh is not None else nullcontext():
            target = input_embedding(forward_batch.input_ids)  # [T, H]
            num_tokens, hidden = target.shape[0], target.shape[1]

            def _replicated(shape) -> jax.Array:
                zeros = jnp.zeros(shape, target.dtype)
                if mesh is not None:
                    zeros = jax.device_put(
                        zeros,
                        NamedSharding(mesh, PartitionSpec(*([None] * len(shape)))),
                    )
                return zeros

            deepstack = getattr(forward_batch, "deepstack_visual_embedding", None)
            deepstack_dim = deepstack.shape[0] if deepstack is not None else 0
            shapes = packed_shapes or ((1, num_tokens),)
            running, merged_deepstack = target, None
            for num_lanes, cap in shapes:
                length = min(num_tokens, cap)
                task = ItemTask(
                    item=None,
                    output_len=length,
                    merge_mappings=(_MergeMapping(0, 0, length),),
                )
                packed = PackedMultimodalEmbedding(
                    output=_replicated((num_lanes, cap, hidden * (1 + deepstack_dim))),
                    placements=((0, 0, length),),
                    deepstack_dim=deepstack_dim,
                )
                running, merged_deepstack = _gather_merge(target, None, packed, (task,), mesh)
                jax.block_until_ready(running)
                if merged_deepstack is not None:
                    jax.block_until_ready(merged_deepstack)

            if embedding_pool is not None:
                length = min(num_tokens, embedding_pool.page_size)
                task = ItemTask(
                    item=None,
                    output_len=length,
                    merge_mappings=(_MergeMapping(0, 0, length),),
                )
                entry = EmbeddingPoolEntry(np.asarray([0], dtype=np.int32), length)
                running, merged_deepstack = _gather_from_pool(
                    target,
                    None,
                    embedding_pool,
                    (task,),
                    (entry,),
                    mesh,
                )
                jax.block_until_ready(running)
                if merged_deepstack is not None:
                    jax.block_until_ready(merged_deepstack)

        forward_batch.input_embedding = running
        if merged_deepstack is not None:
            forward_batch.deepstack_visual_embedding = merged_deepstack

    # ---- Pre-compilation ----

    def precompile_all(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None = None,
        future_token_ids_map=None,
    ):
        self._precompile_extend(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )
        if self.precompile_in_model_multimodal:
            model_runner.model.precompile_multimodal()
            embedding_pool = self._embedding_pool(model_runner)
            if embedding_pool is not None:
                for num_lanes, cap in self._packed_multimodal_shapes(model_runner):
                    embedding_pool.precompile_packed_write(num_lanes, cap)
        self._precompile_decode(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )

    def _precompile_extend(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None,
        future_token_ids_map,
    ):
        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        start_time = time.perf_counter()
        packed_shapes = self._packed_multimodal_shapes(model_runner)
        embedding_pool = self._embedding_pool(model_runner)
        bs = self.max_padded_batch_size
        variant_names = self._extend_variant_names()
        logger.info(
            "[EXTEND] Begin to precompile variants=%s bs_paddings=%s token_paddings=%s",
            variant_names,
            [bs],
            self.token_buckets,
        )

        pairs = list(itertools.product(variant_names, [bs], self.token_buckets))
        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                variant_name, bs_val, num_tokens = pair
                pbar.set_postfix(variant=variant_name, bs=bs_val, tokens=num_tokens)
                if bs_val > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs_val, num_tokens)
                    continue
                batch = self._make_dummy_batch(
                    bs_val,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.cache_loc_buckets[-1],
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
                )
                if variant_name == "multimodal":
                    self._populate_dummy_multimodal_inputs(batch, model_runner)
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if variant_name == "multimodal":
                    self._warm_multimodal_merge(
                        batch.forward_batch,
                        model_runner.model.get_input_embeddings(),
                        packed_shapes,
                        embedding_pool,
                        mesh,
                    )
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import resolve_future_token_ids

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=variant_name == "multimodal",
                    sampling_metadata=sampling_metadata,
                )
                if variant_name == "text":
                    variant_key = (ForwardMode.EXTEND, num_tokens, bs_val, False)
                else:
                    variant_key = ("VLM_EXTEND", num_tokens, bs_val)
                self._compiled_variants.add(variant_key)

        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def _precompile_decode(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None,
        future_token_ids_map,
    ):
        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        start_time = time.perf_counter()
        logger.info(
            "[DECODE] Begin to precompile bs_paddings=%s",
            self.bs_buckets,
        )

        with tqdm(
            enumerate(self.bs_buckets),
            desc="[DECODE] PRECOMPILE",
            leave=False,
            total=len(self.bs_buckets),
        ) as pbar:
            for i, bs_val in pbar:
                pbar.set_postfix(bs=bs_val)
                aligned_cache_loc_size = self.cache_loc_buckets[i]
                batch = self._make_dummy_batch(
                    bs_val,
                    bs_val,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
                )
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import (
                        resolve_future_token_ids,
                        set_future_token_ids,
                    )

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                result = forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=False,
                    sampling_metadata=sampling_metadata,
                )
                if future_token_ids_map is not None:
                    _, next_token_ids, _ = result
                    from sgl_jax.srt.managers.utils import future_slot_indices

                    slots = future_slot_indices(
                        np.asarray(batch.seq_lens),
                        np.asarray(batch.req_pool_indices),
                        future_token_ids_map.shape[0],
                    )
                    set_future_token_ids(future_token_ids_map, slots, next_token_ids, mesh)
                self._compiled_variants.add((ForwardMode.DECODE, bs_val, bs_val, False))

        end_time = time.perf_counter()
        logger.info("[DECODE] Precompile finished in %.0f secs", end_time - start_time)

    # ---- Dummy batch construction ----

    def _make_dummy_batch(
        self,
        bs: int,
        num_tokens: int,
        mode,
        max_cache_loc_size: int,
        speculative_algorithm=None,
        dp_size: int = 1,
        per_dp_bs_size: int = 0,
    ):
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import (
            ForwardMode,
            ModelWorkerBatch,
            ModelWorkerSamplingInfo,
        )
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        # Runtime ScheduleBatch.spec_algorithm is always SpeculativeAlgorithm
        # enum (.from_string(None) -> .NONE). Default to .NONE so the dummy
        # batch's pytree aux matches and precompile shares the cache key with
        # the no-spec runtime path.
        if speculative_algorithm is None:
            spec_algorithm_value = SpeculativeAlgorithm.NONE
        else:
            spec_algorithm_value = speculative_algorithm

        valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
        invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        valid_out_cache_loc = np.arange(1, bs + 1, dtype=jnp.int32)
        invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
        valid_positions = np.array([0] * bs, dtype=jnp.int32)
        invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        invalid_cache_loc_size = max_cache_loc_size - bs
        if invalid_cache_loc_size < 0:
            raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

        valid_cache_loc = np.arange(bs)
        invalid_cache_loc = np.array([0] * invalid_cache_loc_size, dtype=jnp.int32)
        lora_ids = ["0"] * bs

        extend_seq_lens = np.array([1] * bs) if mode == ForwardMode.EXTEND else None
        logits_indices = np.array([0] * bs) if mode == ForwardMode.EXTEND else None

        if speculative_algorithm is None:
            sampling_info = ModelWorkerSamplingInfo.generate_for_precompile(bs, self.vocab_size)
            return_output_logprob_only = True
        else:
            sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
                bs, self.vocab_size
            )
            sampling_info.vocab_mask = None
            return_output_logprob_only = False

        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=np.array([1] * bs, dtype=np.int32),
            out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            return_logprob=False,
            return_output_logprob_only=return_output_logprob_only,
            sampling_info=sampling_info,
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=(np.array([0] * bs) if mode == ForwardMode.EXTEND else None),
            extend_seq_lens=extend_seq_lens,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            logits_indices=logits_indices,
            input_logprob_indices=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.capture_hidden_states else CaptureHiddenMode.NULL
            ),
            spec_algorithm=spec_algorithm_value,
            lora_ids=lora_ids,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs_size,
            real_bs_per_dp=[per_dp_bs_size] * dp_size,
            logits_indices_selector=np.arange(bs, dtype=np.int32),
            # Hybrid recurrent backends (e.g. KDA) require these per-batch
            # arrays even at precompile time; slot 0 is RecurrentStatePool's
            # per-rank dummy slot, safe to point at. Leave None otherwise so
            # non-recurrent backends are unaffected.
            recurrent_indices=(np.zeros(bs, dtype=np.int32) if self.has_recurrent_state else None),
            has_initial_state=(np.zeros(bs, dtype=np.bool_) if self.has_recurrent_state else None),
            recurrent_cow_src_indices=(
                np.zeros(bs, dtype=np.int32)
                if self.supports_recurrent_cow and mode == ForwardMode.EXTEND
                else None
            ),
            recurrent_track_indices=(
                np.zeros(bs, dtype=np.int32) if self.supports_recurrent_track else None
            ),
            recurrent_track_mask=(
                np.zeros(bs, dtype=np.int32) if self.supports_recurrent_track else None
            ),
        )

    # ---- Lazy compilation tracking ----

    def register_variant_if_new(self, variant_key: tuple) -> bool:
        """Register a compilation variant and return True if it was not seen before.

        Used to detect first-time compilation of a (mode, num_tokens, bs, logprob)
        shape tuple so the caller can log or act on cold-compile events.
        TODO: add runtime consumer that warns on cache misses (issue #609).
        """
        if variant_key in self._compiled_variants:
            return False
        self._compiled_variants.add(variant_key)
        return True
