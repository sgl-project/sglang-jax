from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from math import gcd as _gcd
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
        # Fused MoE shards tokens across EP (local = num_tokens // ep_size) AND the kernel
        # requires local_num_tokens % t_packing == 0, where t_packing = 32 // dtype_bits
        # (=2 for bf16 activations). So every compiled num_tokens must be a multiple of
        # ep_size * t_packing. Use bf16 packing (=2); for fp32 (packing=1) ep_size alone
        # suffices, and a multiple of 2*ep_size is still valid.
        self.ep_size = getattr(server_args, "ep_size", 1) or 1
        self.moe_token_align = self.ep_size * 2

        self.token_buckets = self._compute_token_buckets(server_args.precompile_token_paddings)
        self.bs_buckets = self._compute_bs_buckets(server_args.precompile_bs_paddings)
        self.cache_loc_buckets = self._compute_cache_loc_buckets()

        self._compiled_variants: set[tuple] = set()

    def _compute_token_buckets(self, user_paddings: list[int] | None) -> list[int]:
        dp_size = self.dp_size
        if user_paddings is None:
            user_paddings = [item * dp_size for item in PRECOMPILE_DEFAULT_TOKEN_PADDINGS]

        fused = self.moe_backend == "fused"
        # Fused MoE also splits prefill/extend tokens across EP, so token buckets must be
        # multiples of ep_size (in addition to dp_size).
        align = dp_size
        if fused:
            align = (
                dp_size * self.moe_token_align // _gcd(dp_size, self.moe_token_align)
            )  # lcm(dp, ep*packing)

        buckets = []
        for item in user_paddings:
            if item % align != 0:
                item = ((item + align - 1) // align) * align
            if (
                item >= self.max_padded_batch_size
                and item <= self.max_padded_num_tokens
                and item >= dp_size
            ):
                buckets.append(item)

        buckets.sort()
        max_bucket = self.max_padded_num_tokens
        if fused and max_bucket % self.moe_token_align != 0:
            max_bucket = (
                (max_bucket + self.moe_token_align - 1) // self.moe_token_align
            ) * self.moe_token_align
        if len(buckets) == 0 or buckets[-1] < max_bucket:
            buckets.append(max_bucket)

        return buckets

    def _compute_bs_buckets(self, user_paddings: list[int] | None) -> list[int]:
        bs_list = user_paddings if user_paddings is not None else PRECOMPILE_DEFAULT_BS_PADDINGS
        fused = self.moe_backend == "fused"
        # Fused MoE requires every compiled batch size (decode num_tokens) to be a
        # multiple of ep_size, not merely >= tp_size*2 (with separate EP/TP these differ,
        # e.g. ep_size=16, attn tp_size=4 — bs=8 passed the old guard but crashes the
        # fused kernel's num_tokens // ep_size split).
        buckets = []
        for bs in bs_list:
            if bs > self.max_padded_batch_size or bs < self.dp_size:
                continue
            if fused and bs % self.moe_token_align != 0:
                continue
            buckets.append(bs)
        buckets.sort()
        # Force-append a final bucket covering max batch; align it to ep_size for fused.
        max_bucket = self.max_padded_batch_size
        if fused and max_bucket % self.moe_token_align != 0:
            max_bucket = (
                (max_bucket + self.moe_token_align - 1) // self.moe_token_align
            ) * self.moe_token_align
        if len(buckets) == 0 or buckets[-1] < max_bucket:
            buckets.append(max_bucket)
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
            input_logprob_indices=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.multimodal else CaptureHiddenMode.NULL
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
