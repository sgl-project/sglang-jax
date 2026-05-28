from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
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
        multimodal: bool = False,
        has_recurrent_state: bool = False,
    ):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.page_size = page_size
        self.max_req_len = max_req_len
        self.max_padded_batch_size = max_padded_batch_size
        self.max_padded_num_tokens = max_padded_num_tokens
        self.vocab_size = vocab_size
        self.multimodal = multimodal
        self.has_recurrent_state = has_recurrent_state
        self.moe_backend = server_args.moe_backend
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
        buckets = []
        for bs in bs_list:
            if (
                bs <= self.max_padded_batch_size
                and (self.moe_backend != "fused" or bs >= self.tp_size * 2)
                and bs >= self.dp_size
            ):
                buckets.append(bs)
        buckets.sort()
        if len(buckets) == 0 or buckets[-1] < self.max_padded_batch_size:
            buckets.append(self.max_padded_batch_size)
        return buckets

    def _compute_cache_loc_buckets(self) -> list[int]:
        pages_per_req = (self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
        return [bs * pages_per_req for bs in self.bs_buckets]

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
        self._precompile_decode(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )
        # Multimodal models additionally need the ViT and splice JITs warmed
        # up — they are NOT exercised by the text-only dummy batches above.
        if getattr(model_runner, "is_multimodal_model", False):
            self._precompile_visual_encode(model_runner)
            self._precompile_splice(model_runner)
            # Multimodal-extend LLM JIT: same `jitted_run_model` as text-only
            # extend, but the orchestrator pre-populates `input_embedding` so
            # the pytree footprint of `forward_batch` differs (jax.Array leaf
            # instead of None). Without this pass, the first multimodal
            # request per batch shape cold-compiles `jitted_run_model`.
            self._precompile_extend_with_input_embedding(
                forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
            )
        # Note: do NOT add a "greedy sampler precompile" pass here. The
        # non-greedy and all-greedy `ModelWorkerSamplingInfo` factories
        # produce identical pytree structure (only field values differ). The
        # sampler itself branches on `apply_vocab_mask` etc. via `lax.cond`,
        # which traces both branches into one graph. So an explicit greedy
        # pass adds zero new cache entries — confirmed empirically (a prior
        # attempt finished in <1s with no JIT compile events).

    def _precompile_visual_encode(self, model_runner: ModelRunner):
        """AOT-compile `jitted_visual_encode` for every power-of-two patch
        bucket. Fixes `n_real_images=1` (single-image case dominates eval and
        production traffic); multi-image variants lazy-compile on first
        occurrence with the `_pick_bucket` warning surfacing the recompile.
        """
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import get_multimodal_patch_buckets

        patch_buckets = get_multimodal_patch_buckets()
        if not patch_buckets:
            return
        start_time = time.perf_counter()
        logger.info(
            "[VISUAL_ENCODE] Begin to precompile patch_buckets=%s (n_real_images=1)",
            patch_buckets,
        )
        # Read pixel feature dim from the existing scheduler-side constant so
        # we don't have to duplicate the value across files.
        from sgl_jax.srt.managers.schedule_batch import _MM_PIXEL_FEATURE_DIM

        with tqdm(patch_buckets, desc="[VISUAL_ENCODE] PRECOMPILE", leave=False) as pbar:
            for n_patches in pbar:
                pbar.set_postfix(patches=n_patches)
                # Pick (h, w) close to a square so the static aux key resembles
                # real traffic. Any (h, w) with h*w == n_patches and both even
                # (spatial_merge_size=2) is valid; the ViT input shape only
                # depends on n_patches, not the (h, w) split.
                k = n_patches.bit_length() - 1  # floor(log2(n_patches))
                h = 1 << ((k + 1) // 2)
                w = 1 << (k // 2)
                assert h * w == n_patches, (n_patches, h, w)
                pixel_values = jnp.zeros((n_patches, _MM_PIXEL_FEATURE_DIM), dtype=jnp.bfloat16)
                cu_seqlens = jnp.array([0, n_patches], dtype=jnp.int32)
                model_runner.jitted_visual_encode(
                    pixel_values,
                    ((1, h, w),),  # image_grid_thw (static)
                    cu_seqlens,
                    1,  # n_real_images (static)
                )

        end_time = time.perf_counter()
        logger.info("[VISUAL_ENCODE] Precompile finished in %.0f secs", end_time - start_time)

    def _precompile_splice(self, model_runner: ModelRunner):
        """AOT-compile `jitted_splice_embeds` for the Cartesian product of
        token_buckets × (patch_buckets / spatial_merge_unit). Each splice
        compile is a single scatter — cheap, but we cover all combinations so
        runtime never hits a cold compile.
        """
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import (
            _MM_PATCHES_PER_TOKEN,
            get_multimodal_patch_buckets,
        )

        patch_buckets = get_multimodal_patch_buckets()
        if not patch_buckets:
            return
        # N_padded_tokens = n_patches / spatial_merge_unit (4 for Qwen3-VL).
        token_padding_buckets = tuple(
            n_patches // _MM_PATCHES_PER_TOKEN for n_patches in patch_buckets
        )
        # Read hidden dims from the model. text_config carries hidden_size for
        # the LLM-side embedding; vision_config carries deepstack count.
        text_config = getattr(model_runner.model, "text_config", None)
        vision_config = getattr(model_runner.model, "vision_config", None)
        if text_config is None or vision_config is None:
            logger.warning("[SPLICE] model missing text_config / vision_config — skip precompile")
            return
        hidden = int(text_config.hidden_size)
        n_deepstack = len(vision_config.deepstack_visual_indexes)
        deepstack_dim = hidden * n_deepstack

        start_time = time.perf_counter()
        pairs = [
            (seq_len, n_padded)
            for seq_len in self.token_buckets
            for n_padded in token_padding_buckets
            if n_padded <= seq_len
        ]
        logger.info(
            "[SPLICE] Begin to precompile %d (seq_len, n_padded_tokens) pairs",
            len(pairs),
        )
        with tqdm(pairs, desc="[SPLICE] PRECOMPILE", leave=False) as pbar:
            for seq_len, n_padded in pbar:
                pbar.set_postfix(seq=seq_len, padded=n_padded)
                input_ids = jnp.zeros((seq_len,), dtype=jnp.int32)
                vision_main = jnp.zeros((n_padded, hidden), dtype=jnp.bfloat16)
                vision_deepstack = jnp.zeros((n_padded, deepstack_dim), dtype=jnp.bfloat16)
                # Spread placeholder positions across the seq_len so the
                # scatter exercises non-trivial indices (rather than a tight
                # block at the front). The actual values don't affect compile.
                positions = jnp.linspace(0, seq_len - 1, n_padded, dtype=jnp.float32).astype(
                    jnp.int32
                )
                model_runner.jitted_splice_embeds(
                    input_ids, vision_main, vision_deepstack, positions
                )

        end_time = time.perf_counter()
        logger.info("[SPLICE] Precompile finished in %.0f secs", end_time - start_time)

    def _precompile_extend_with_input_embedding(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None,
        future_token_ids_map,
    ):
        """Warm `jitted_run_model` for the multimodal-extend pytree footprint.

        The orchestrator (`run_model_wrapper`) populates
        `forward_batch.input_embedding` (jax.Array leaf) when a request
        carries pixel_values; the LLM JIT then sees a different pytree
        footprint than the text-only `input_embedding=None` case covered by
        `_precompile_extend`. Without this pass each multimodal batch shape
        cold-compiles `jitted_run_model` on first request — surfaced as
        ~1 cache_miss per Prefill batch in production runs.
        """
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        # Read hidden dims from the loaded model — same fields the splice
        # precompile uses, so we keep the helpers consistent.
        text_config = getattr(model_runner.model, "text_config", None)
        vision_config = getattr(model_runner.model, "vision_config", None)
        if text_config is None or vision_config is None:
            logger.warning(
                "[EXTEND_WITH_INPUT_EMBEDDING] model missing text_config / "
                "vision_config — skip precompile"
            )
            return
        hidden = int(text_config.hidden_size)
        n_deepstack = len(vision_config.deepstack_visual_indexes)
        deepstack_dim = hidden * n_deepstack

        start_time = time.perf_counter()
        bs = self.max_padded_batch_size
        logger.info(
            "[EXTEND_WITH_INPUT_EMBEDDING] Begin to precompile bs=%d token_paddings=%s",
            bs,
            self.token_buckets,
        )

        pairs = list(itertools.product([bs], self.token_buckets))
        with tqdm(pairs, desc="[EXTEND_WITH_INPUT_EMBEDDING] PRECOMPILE", leave=False) as pbar:
            for bs_val, num_tokens in pbar:
                pbar.set_postfix(bs=bs_val, tokens=num_tokens)
                if bs_val > num_tokens:
                    continue
                input_embedding = jnp.zeros((num_tokens, hidden), dtype=jnp.bfloat16)
                deepstack_visual_embedding = jnp.zeros(
                    (num_tokens, deepstack_dim), dtype=jnp.bfloat16
                )
                batch = self._make_dummy_batch(
                    bs_val,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.cache_loc_buckets[-1],
                    dp_size=self.dp_size,
                    per_dp_bs_size=bs_val // self.dp_size,
                    input_embedding=input_embedding,
                    deepstack_visual_embedding=deepstack_visual_embedding,
                )
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import resolve_future_token_ids

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=False,
                    sampling_metadata=sampling_metadata,
                )

        end_time = time.perf_counter()
        logger.info(
            "[EXTEND_WITH_INPUT_EMBEDDING] Precompile finished in %.0f secs",
            end_time - start_time,
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
        bs = self.max_padded_batch_size
        logger.info(
            "[EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            [bs],
            self.token_buckets,
        )

        pairs = list(itertools.product([bs], self.token_buckets))
        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                bs_val, num_tokens = pair
                pbar.set_postfix(bs=bs_val, tokens=num_tokens)
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
                if prepare_lora_fn is not None:
                    prepare_lora_fn(batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    batch, 0, mesh, self.vocab_size
                )
                batch.forward_batch = ForwardBatch.init_new(batch, model_runner)
                if future_token_ids_map is not None:
                    from sgl_jax.srt.managers.utils import resolve_future_token_ids

                    batch.forward_batch.input_ids = resolve_future_token_ids(
                        batch.forward_batch.input_ids, future_token_ids_map, mesh
                    )
                forward_fn(
                    batch,
                    launch_done=None,
                    skip_sample=False,
                    sampling_metadata=sampling_metadata,
                )
                self._compiled_variants.add((ForwardMode.EXTEND, num_tokens, bs_val, False))

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
                    set_future_token_ids(future_token_ids_map, 0, next_token_ids, mesh)
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
        *,
        input_embedding=None,
        deepstack_visual_embedding=None,
    ):
        import jax.numpy as jnp

        from sgl_jax.srt.managers.schedule_batch import (
            ForwardMode,
            ModelWorkerBatch,
            ModelWorkerSamplingInfo,
        )
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

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
        else:
            sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
                bs, self.vocab_size
            )

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
            return_output_logprob_only=True,
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
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.multimodal else CaptureHiddenMode.NULL
            ),
            spec_algorithm=speculative_algorithm,
            lora_ids=lora_ids,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs_size,
            real_bs_per_dp=[bs] * dp_size,
            logits_indices_selector=np.arange(bs, dtype=np.int32),
            # Hybrid recurrent backends (e.g. KDA) require these per-batch
            # arrays even at precompile time; slot 0 is RecurrentStatePool's
            # per-rank dummy slot, safe to point at. Leave None otherwise so
            # non-recurrent backends are unaffected.
            recurrent_indices=(np.zeros(bs, dtype=np.int32) if self.has_recurrent_state else None),
            has_initial_state=(np.zeros(bs, dtype=np.bool_) if self.has_recurrent_state else None),
            # Multimodal-extend LLM JIT precompile coverage: when these are
            # populated (Fix A — `_precompile_extend_with_input_embedding`),
            # `run_model_wrapper` skips the orchestrator's visual encode/splice
            # and feeds `jitted_run_model` a forward_batch with `input_embedding`
            # already set — exactly the cache key that multimodal extends hit
            # at runtime.
            input_embedding=input_embedding,
            deepstack_visual_embedding=deepstack_visual_embedding,
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
