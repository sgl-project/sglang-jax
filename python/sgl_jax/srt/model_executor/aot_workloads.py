"""Workload-specific abstract inputs for the shared serving model forward."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    FlashAttentionMetadata,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.mla_backend import (
    MLAAttentionBackend,
    MLAAttentionMetadata,
)
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    request_count: int
    input_token_count: int
    dp_size: int
    context_length: int
    page_size: int
    tokens_per_request: int | None
    chunked_prefill_size: int | None
    mtp_layer_idx: int | None


@dataclass(frozen=True)
class InputContext:
    model_config: ModelConfig
    mesh: jax.sharding.Mesh
    backend: AttentionBackend
    memory_pools: MemoryPools

    def shaped(self, shape, dtype=jnp.int32):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(self.mesh, P("data")))

    def vector(self, length, dtype=jnp.int32):
        return self.shaped((length,), dtype)


@dataclass(frozen=True)
class WorkloadInputs:
    batch: ForwardBatch
    logits: LogitsMetadata
    attention_metadata: object


def _attention_metadata(backend, context, spec, *, page_count=None, swa=True):
    """Use serving's metadata pytrees and per-DP layouts without allocating arrays."""
    vector = context.vector
    bs, dp = spec.request_count, spec.dp_size
    if isinstance(backend, HybridLinearAttnBackend):
        return HybridLinearAttnBackendMetadata(
            full_attn_metadata=_attention_metadata(backend.full_attn_backend, context, spec),
            linear_attn_metadata=LinearRecurrentAttnBackendMetadata(
                cu_q_lens=vector(bs + dp),
                recurrent_indices=vector(bs),
                has_initial_state=vector(bs, jnp.bool_),
            ),
        )
    if isinstance(backend, NativeAttention):
        return None
    if page_count is None:
        page_count = bs * -(-spec.context_length // spec.page_size)
    fields = {
        "cu_q_lens": vector(bs + dp),
        "cu_kv_lens": vector(bs + dp),
        "page_indices": vector(page_count),
        "seq_lens": vector(bs),
        "distribution": vector(3 * dp),
    }
    if isinstance(backend, FlashAttention):
        return FlashAttentionMetadata(
            **fields, swa_page_indices=vector(page_count) if swa else None
        )
    if isinstance(backend, MLAAttentionBackend):
        return MLAAttentionMetadata(**fields)
    raise ValueError(
        f"No offline attention metadata constructor for backend {type(backend).__name__}"
    )


class WorkloadInputBuilder(ABC):
    model_role = "target"
    spec_algorithm = SpeculativeAlgorithm.NONE
    capture_hidden_mode = CaptureHiddenMode.NULL
    uses_draft_width = False

    def __init__(self, options):
        if options.mtp_layer_idx < 0 or (options.mtp_layer_idx and self.model_role != "draft"):
            raise ValueError(
                "mtp_layer_idx must be nonnegative and is only used by MTP draft workloads"
            )
        if self.uses_draft_width:
            if options.draft_token_num is None or options.draft_token_num < 2:
                raise ValueError(
                    "verify/draft-extend requires --draft-token-num >= 2 (including seed)"
                )
            if options.draft_token_num > options.context_length:
                raise ValueError("context_length must include the entire verify/draft-extend block")
        elif options.draft_token_num is not None:
            raise ValueError("--draft-token-num is only used by target-verify or mtp-draft-extend")
        tokens, width = self._token_shape(options)
        if options.moe_backend == "fused_v2" and tokens % options.ep_size:
            raise ValueError("fused_v2 requires the input token count divisible by ep_size")
        self.spec = WorkloadSpec(
            name=options.workload,
            request_count=options.batch_size,
            input_token_count=tokens,
            dp_size=options.dp_size,
            context_length=options.context_length,
            page_size=options.page_size,
            tokens_per_request=width,
            chunked_prefill_size=options.chunked_prefill_size,
            mtp_layer_idx=options.mtp_layer_idx if self.model_role == "draft" else None,
        )

    def _token_shape(self, options):
        if options.chunked_prefill_size is not None or options.num_tokens is not None:
            raise ValueError("--chunked-prefill-size and --num-tokens are only used by prefill")
        width = options.draft_token_num if self.uses_draft_width else 1
        return options.batch_size * width, width

    @abstractmethod
    def build(self, context: InputContext) -> WorkloadInputs:
        """Construct final batch, logits, and attention metadata for this workload."""

    def _batch(self, context, metadata, **fields):
        spec, vector = self.spec, context.vector
        padded_context = -(-spec.context_length // spec.page_size) * spec.page_size
        recurrent_indices = None
        if context.memory_pools.recurrent_state_pool is not None:
            recurrent_indices = metadata.linear_attn_metadata.recurrent_indices
        return ForwardBatch(
            bid=0,
            forward_mode=self.forward_mode,
            batch_size=spec.request_count,
            input_ids=vector(spec.input_token_count),
            req_pool_indices=vector(spec.request_count),
            seq_lens=vector(spec.request_count),
            out_cache_loc=vector(spec.input_token_count),
            positions=vector(spec.input_token_count),
            attn_backend=context.backend,
            cache_loc=vector(spec.request_count * padded_context),
            recurrent_indices=recurrent_indices,
            spec_algorithm=self.spec_algorithm,
            capture_hidden_mode=self.capture_hidden_mode,
            **fields,
        )

    def _logits(self, context, **fields):
        fields.setdefault("logits_indices", context.vector(self.spec.request_count))
        return LogitsMetadata(
            forward_mode=self.forward_mode,
            capture_hidden_mode=self.capture_hidden_mode,
            **fields,
        )


class DecodeInputBuilder(WorkloadInputBuilder):
    forward_mode = ForwardMode.DECODE

    def build(self, context):
        metadata = _attention_metadata(context.backend, context, self.spec)
        return WorkloadInputs(self._batch(context, metadata), self._logits(context), metadata)


class PrefillInputBuilder(WorkloadInputBuilder):
    forward_mode = ForwardMode.EXTEND

    def _token_shape(self, options):
        chunk = options.chunked_prefill_size
        if chunk is None or chunk <= 0 or chunk % options.page_size:
            raise ValueError("prefill requires --chunked-prefill-size > 0, divisible by page_size")
        # Serving maintains a chunk budget per DP rank. Requests on a rank
        # share that budget; their individual extend lengths remain dynamic.
        tokens = options.num_tokens if options.num_tokens is not None else chunk * options.dp_size
        if tokens < options.batch_size or tokens % options.dp_size:
            raise ValueError("num_tokens must cover batch_size and be divisible by dp_size")
        if tokens > chunk * options.dp_size:
            raise ValueError(
                "num_tokens exceeds the global chunk budget (chunked_prefill_size * dp_size)"
            )
        if tokens > options.batch_size * options.context_length:
            raise ValueError("num_tokens exceeds batch_size * context_length")
        return tokens, None

    def build(self, context):
        metadata = _attention_metadata(context.backend, context, self.spec)
        extend_lens = context.vector(self.spec.request_count)
        batch = self._batch(
            context,
            metadata,
            extend_prefix_lens=context.vector(self.spec.request_count),
            extend_seq_lens=extend_lens,
        )
        return WorkloadInputs(batch, self._logits(context, extend_seq_lens=extend_lens), metadata)


class SpeculativeInputBuilder(WorkloadInputBuilder):
    spec_algorithm = SpeculativeAlgorithm.NEXTN
    capture_hidden_mode = CaptureHiddenMode.FULL

    def _metadata(self, context):
        if not isinstance(context.backend, FlashAttention):
            raise ValueError(
                "Speculative export requires the FlashAttention speculative metadata interface"
            )
        spec = self.spec
        pages_per_request = -(-spec.context_length // spec.page_size)
        if self.forward_mode == ForwardMode.DECODE:
            page_count = 16384  # Serving get_eagle_multi_step_metadata's per-step page buffer.
            if spec.request_count * pages_per_request > page_count:
                raise ValueError("MTP draft pages exceed the serving 16384-entry page buffer")
        else:
            page_count = spec.request_count * max(16, 1 << (pages_per_request - 1).bit_length())
        return _attention_metadata(
            context.backend,
            context,
            spec,
            page_count=page_count,
            swa=self.forward_mode != ForwardMode.DECODE,
        )


class MtpDraftInputBuilder(SpeculativeInputBuilder):
    model_role = "draft"
    forward_mode = ForwardMode.DECODE
    capture_hidden_mode = CaptureHiddenMode.LAST

    def build(self, context):
        metadata = self._metadata(context)
        spec_info = EagleDraftInput(
            hidden_states=context.shaped(
                (self.spec.input_token_count, context.model_config.hidden_size), jnp.bfloat16
            ),
            capture_hidden_mode=self.capture_hidden_mode,
        )
        return WorkloadInputs(
            self._batch(context, metadata, spec_info=spec_info), self._logits(context), metadata
        )


class MtpDraftExtendInputBuilder(SpeculativeInputBuilder):
    model_role = "draft"
    forward_mode = ForwardMode.DRAFT_EXTEND
    uses_draft_width = True

    def build(self, context):
        metadata = self._metadata(context)
        extend_lens = context.vector(self.spec.request_count)
        accept_lens = context.vector(self.spec.request_count)
        spec_info = EagleDraftInput(
            hidden_states=context.shaped(
                (self.spec.input_token_count, context.model_config.hidden_size), jnp.bfloat16
            ),
            accept_length=accept_lens,
        )
        batch = self._batch(
            context,
            metadata,
            spec_info=spec_info,
            extend_prefix_lens=context.vector(self.spec.request_count),
            extend_seq_lens=extend_lens,
        )
        # Serving returns all hidden rows, but only each request's last
        # accepted token's logits. Acceptance itself is outside this forward.
        logits = self._logits(context, extend_seq_lens=extend_lens, accept_lens=accept_lens)
        return WorkloadInputs(batch, logits, metadata)


class TargetVerifyInputBuilder(SpeculativeInputBuilder):
    forward_mode = ForwardMode.TARGET_VERIFY
    uses_draft_width = True

    def build(self, context):
        metadata = self._metadata(context)
        # NEXTN topk=1 is a causal chain. Sampling/acceptance arrays are not
        # consumed by the model forward and remain absent.
        spec_info = EagleVerifyInput(
            draft_token=None,
            custom_mask=None,
            positions=None,
            retrive_index=None,
            retrive_next_token=None,
            retrive_next_sibling=None,
            spec_steps=self.spec.tokens_per_request - 1,
            draft_token_num=self.spec.tokens_per_request,
        )
        batch = self._batch(context, metadata, spec_info=spec_info)
        logits = self._logits(context, logits_indices=context.vector(self.spec.input_token_count))
        return WorkloadInputs(batch, logits, metadata)


WORKLOAD_BUILDERS = {
    "decode": DecodeInputBuilder,
    "prefill": PrefillInputBuilder,
    "mtp-draft": MtpDraftInputBuilder,
    "mtp-draft-extend": MtpDraftExtendInputBuilder,
    "target-verify": TargetVerifyInputBuilder,
}


def get_input_builder(options) -> WorkloadInputBuilder:
    try:
        builder = WORKLOAD_BUILDERS[options.workload]
    except KeyError:
        raise ValueError(
            f"Unknown workload {options.workload!r}; choose from {', '.join(WORKLOAD_BUILDERS)}"
        ) from None
    return builder(options)
