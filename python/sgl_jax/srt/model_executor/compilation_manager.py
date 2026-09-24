from __future__ import annotations

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
    """Owns serving compile plans and executable compilation, loading, and storage."""

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
            # The static defaults top out at 8192 * dp_size. When the token
            # budget is larger, keep doubling so the final max-size bucket is
            # not the only one above 8192 (every mid-size prefill would pad to
            # max otherwise). No-op when max_padded_num_tokens <= 8192 * dp_size.
            item = user_paddings[-1] * 2
            while item < self.max_padded_num_tokens:
                user_paddings.append(item)
                item *= 2

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

    # ---- Pre-compilation ----

    @staticmethod
    def restore_aot_defaults(server_args):
        """Reuse resolved capacity defaults when loading an offline serving bundle."""
        import json
        from pathlib import Path

        path = Path(server_args.aot_model_dir) / "serving.json"
        if not path.exists():
            return  # Individual IR exports contain executable.json only.
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"AOT serving export is incomplete: {path}")
        for name in ("max_running_requests", "max_total_tokens", "max_recurrent_state_size"):
            if getattr(server_args, name) is None:
                setattr(server_args, name, manifest[name])

    @staticmethod
    def resolve_max_running_requests(
        server_args, context_len, attn_backend, token_capacity, pool_limit, moe_backend
    ):
        # Calculate max_running_requests from different constraints
        attn_backend_limit = (
            attn_backend.get_max_running_reqests(
                context_len,
                server_args.page_size,
            )
            * server_args.dp_size
        )
        server_limit = (
            token_capacity // 2
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )
        constraints = [server_limit, pool_limit, attn_backend_limit]
        max_running_requests = min(constraints)
        # Log each constraint for debugging
        logger.info("Max running requests constraints:")
        logger.info(
            "  - Server limit: %s %s",
            server_limit,
            (
                "(max_total_tokens//2)"
                if server_args.max_running_requests is None
                else "(configured)"
            ),
        )
        logger.info("  - Token pool size: %s", pool_limit)
        logger.info(
            "  - Attention backend: %s (context_len=%s, page_size=%s)",
            attn_backend_limit,
            context_len,
            server_args.page_size,
        )
        logger.info("  → Final max_running_requests: %s", max_running_requests)

        # Validate and adjust max_running_requests for Data Parallelism
        dp_size = server_args.dp_size
        if max_running_requests < dp_size:
            raise ValueError(
                f"max_running_requests ({max_running_requests}) is less than dp_size ({dp_size}). "
                f"Please increase memory allocation or reduce dp_size."
            )
        if max_running_requests % dp_size != 0:
            original_value = max_running_requests
            max_running_requests = (max_running_requests // dp_size) * dp_size
            logger.warning(
                "Adjusted max_running_requests from %s to %s to be divisible by dp_size (%s)",
                original_value,
                max_running_requests,
                dp_size,
            )

        # fused_ep_moe derives its EP group from the mesh (get_ep_size(mesh) =
        # mesh['data'] * mesh['tensor']), not from --ep-size, so align against
        # the actual mesh shape. Use the *resolved* backend from ModelConfig so
        # architectures that hard-code FusedEPMoE (e.g. Qwen3.5 MoE) are
        # covered even when the raw server_args string stays at "epmoe".
        mesh_ep_size = server_args.tp_size
        if moe_backend in ("fused", "fused_v2") and mesh_ep_size > 1:
            from sgl_jax.srt.utils.common_utils import align_bs_for_fused_ep

            assert mesh_ep_size % dp_size == 0, (
                f"fused MoE requires mesh_ep_size ({mesh_ep_size}) to be a multiple "
                f"of dp_size ({dp_size}) so the ep-aligned cap stays dp-aligned"
            )
            aligned = align_bs_for_fused_ep(max_running_requests, mesh_ep_size)
            if aligned != max_running_requests:
                logger.warning(
                    "Adjusted max_running_requests from %s to %s for fused MoE "
                    "(mesh_ep_size=%s, bt must be in {2,4,8k})",
                    max_running_requests,
                    aligned,
                    mesh_ep_size,
                )
                max_running_requests = aligned

        assert max_running_requests > 0, "max_running_request is zero"

        return max_running_requests

    @staticmethod
    def get_max_padded_size(server_args, max_running_requests):
        per_dp_tokens = server_args.max_prefill_tokens
        if server_args.chunked_prefill_size > 0:
            per_dp_tokens = min(per_dp_tokens, server_args.chunked_prefill_size)
        num_tokens = per_dp_tokens * server_args.dp_size
        batch_size = min(max_running_requests, num_tokens)
        if batch_size % server_args.dp_size:
            raise ValueError("max_padded_batch_size must be divisible by dp_size")
        return batch_size, num_tokens

    def iter_model_shapes(self, mode):
        """The model shapes used by both serving warmup and offline export."""
        if mode.is_extend():
            for tokens in self.token_buckets:
                yield self.max_padded_batch_size, tokens, self.cache_loc_buckets[-1]
        elif mode.is_decode():
            for bs, cache_loc in zip(self.bs_buckets, self.cache_loc_buckets):
                yield bs, bs, cache_loc
        else:
            raise ValueError(f"No serving precompile shapes for {mode}")

    @staticmethod
    def compiler_options(backend, batch=None, overrides=None):
        from sgl_jax.srt.model_executor.aot_dispatch import (
            decode_no_sc_gather_compiler_options_fn,
        )
        from sgl_jax.srt.utils.common_utils import get_bool_env_var
        from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

        options = dict(getattr(backend, "compiler_options", None) or {})
        if is_tpu_runtime() and get_bool_env_var("SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER"):
            options["xla_tpu_enable_log_recorder"] = "true"
        if batch is not None:
            decode_options = decode_no_sc_gather_compiler_options_fn()
            if decode_options is not None:
                options.update(decode_options((batch,)) or {})
        options.update(overrides or {})
        return options

    @staticmethod
    def get_executable(lowered, mesh=None, compiler_options=None, *, store=None, output=None):
        """Acquire an executable for a lowering, optionally persisting it.

        A store selects strict offline loading: a miss never invokes the compiler.
        Shape caching and dispatch remain in AotDispatcher for both sources.
        """
        if store is not None:
            if output is not None:
                raise ValueError("Choose executable loading or export, not both")
            return store.load(lowered, compiler_options)
        compiled = lowered.compile(compiler_options=compiler_options or None)
        if output is not None:
            from sgl_jax.srt.model_executor.aot_executable import save_executable

            save_executable(compiled, lowered, mesh, compiler_options, output)
        return compiled

    def precompile_all(
        self,
        forward_fn: Callable,
        model_runner: ModelRunner,
        mesh,
        prepare_lora_fn: Callable | None = None,
        future_token_ids_map=None,
    ):
        self._precompile_encode(model_runner)
        self._precompile_extend(
            forward_fn,
            model_runner,
            mesh,
            prepare_lora_fn,
            future_token_ids_map,
        )
        self._precompile_decode(
            forward_fn, model_runner, mesh, prepare_lora_fn, future_token_ids_map
        )

    def _precompile_encode(self, model_runner) -> None:
        if not self.precompile_in_model_multimodal:
            return
        from sgl_jax.srt.multimodal.in_model.host_orchestration import (
            precompile_multimodal_encoder,
        )
        from sgl_jax.srt.multimodal.in_model.lane_packing import encoder_num_lanes

        config = model_runner.model_config.hf_config
        precompile_multimodal_encoder(
            model_runner.model,
            model_runner.embedding_pool,
            [t for t in self.token_buckets if t >= self.max_padded_batch_size],
            num_lanes=encoder_num_lanes(model_runner.mesh, config.vision_encoder_parallel == "tp"),
            patch_paddings=config.precompile_vision_patch_paddings,
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

        with tqdm(
            self.iter_model_shapes(ForwardMode.EXTEND),
            desc="[EXTEND] PRECOMPILE",
            leave=False,
            total=len(self.token_buckets),
        ) as pbar:
            for bs_val, num_tokens, cache_loc_size in pbar:
                pbar.set_postfix(bs=bs_val, tokens=num_tokens)
                if bs_val > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs_val, num_tokens)
                    continue
                batch = self._make_dummy_batch(
                    bs_val,
                    num_tokens,
                    ForwardMode.EXTEND,
                    cache_loc_size,
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
            self.iter_model_shapes(ForwardMode.DECODE),
            desc="[DECODE] PRECOMPILE",
            leave=False,
            total=len(self.bs_buckets),
        ) as pbar:
            for bs_val, num_tokens, aligned_cache_loc_size in pbar:
                pbar.set_postfix(bs=bs_val)
                batch = self._make_dummy_batch(
                    bs_val,
                    num_tokens,
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
                        get_token_ids_gather,
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
                    set_future_token_ids(
                        future_token_ids_map,
                        batch.forward_batch.seq_lens,
                        batch.forward_batch.req_pool_indices,
                        next_token_ids,
                        mesh,
                    )
                    get_token_ids_gather(mesh)(next_token_ids).block_until_ready()
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
