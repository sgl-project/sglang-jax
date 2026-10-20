"""Central registry for sgl-jax environment variables.

First, intentionally small version of the pattern used by sglang's
python/sglang/srt/environ.py: each variable is declared once here with its
type and default instead of scattering ``os.environ.get`` calls across files.
New environment variables should be added to ``Envs`` below; existing
scattered variables can migrate over time.
"""

import os
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError

    def get(self) -> Any:
        value = os.environ.get(self.name)
        if value is None:
            return self.default
        return self.parse(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"Invalid boolean value for {self.name}: {value!r}")


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        return int(value)


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class Envs:
    # Compute the GPT-J (interleaved) rotary embedding directly in the
    # interleaved domain instead of strided even/odd slices plus a
    # stack+reshape re-interleave. Bit-identical; escape hatch only.
    SGLANG_JAX_ROTARY_INTERLEAVED = EnvBool(True)

    # Merge the rotated rope dims back into the DSA indexer query/key with
    # concatenate instead of a read-modify-write ``at[...].set``. Values are
    # identical; escape hatch only.
    SGLANG_JAX_INDEXER_ROPE_CONCAT = EnvBool(True)

    # gmm_v2 with int4 weights on chips that have an fp8 MXU but no int8 MXU
    # (v7x): quantize activations to e4m3 and upcast the int4 weight tile
    # in-kernel (W4A8) instead of running bf16 activations on the bf16 MXU
    # (W4A16). Escape hatch only; chips with an int8 MXU always use it.
    SGLANG_JAX_GMM_INT4_A8 = EnvBool(True)

    # gmm_v2 activation (lhs) quantization block along K when the weight scale
    # is per-channel (block-scaled weights always use the weight block). 0 =
    # kernel default (512; 128 for int4 weights, matching the granularity of
    # the FP8 block-128 production path).
    SGLANG_JAX_GMM_LHS_QBLOCK = EnvInt(0)

    # EPMoE: concatenate each layer's gate (wi_0) and up (wi_1) expert weights
    # into one [E, K, 2N] grouped matmul (interleaved in 256-column chunks so
    # GEMM1 emits silu(gate)*up from the gmm_v2 epilogue). Per-channel-scaled or
    # unquantized experts with moe tensor parallel 1 only; block-scaled experts
    # keep the two-call path. Opt-in (bit-identical to the two-call path; GLM-5.2
    # serving on v7x sets it).
    SGLANG_JAX_EPMOE_FUSED_GATE_UP = EnvBool(False)

    # GLM-5 prefill micro-batching: an EXTEND chunk of at least
    # DSA_PREFILL_MICROBATCH_MIN_TOKENS (padded) tokens runs as this many
    # consecutive token windows on the block-interleaved schedule so each
    # window's MoE all-reduce overlaps the other window's compute. 1 = off
    # (default; GLM-5.2 serving on v7x sets 2, together with the prefill XLA
    # options the tagged windows need).
    DSA_PREFILL_MICROBATCH = EnvInt(1)
    DSA_PREFILL_MICROBATCH_MIN_TOKENS = EnvInt(2048)

    # One-shot (Pallas) all-reduce for small tensor-parallel psums: operands of
    # MIN..MAX bytes (2-D, lanes % 128 == 0, bf16/f32, even device count) at the
    # listed call sites ("fused" = the per-layer MoE routed+shared reduce taken
    # at the model level, "qmm" = the quantized-matmul wrapper reduce (o_proj,
    # dense MLP), "moe" = the EPMoE-internal psum) go to the split_flat kernel
    # instead of XLA's all-reduce. MIN_BYTES=0 turns the kernel off.
    # 0 = off (default; GLM-5.2 serving on v7x sets 4096).
    SGLANG_JAX_ONESHOT_AR_MIN_BYTES = EnvInt(0)
    SGLANG_JAX_ONESHOT_AR_MAX_BYTES = EnvInt(512 << 10)
    SGLANG_JAX_ONESHOT_AR_SITES = EnvStr("fused,qmm")

    # Per-executable XLA options ("flag=value,flag=value"). PREFILL applies to
    # EXTEND / MIXED executables of at least PREFILL_XLA_OPTIONS_MIN_TOKENS
    # (padded) tokens, DECODE to decode-shaped ones (DECODE, TARGET_VERIFY,
    # DRAFT_EXTEND); neither touches the other side. Empty = no options (default).
    # The GLM-5.2 v7x production prefill set is
    # xla_jf_crs_combiner_threshold_in_bytes=1 (keep the two micro-batch windows'
    # all-reduces apart), xla_jf_enable_multi_output_fusion=false and
    # xla_jf_enable_producer_consumer_multi_output_fusion=false (the scheduling
    # groups reject fused producers),
    # xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true,
    # xla_tpu_sparse_core_all_reduce_latency_multiplier=8,
    # xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true.
    SGLANG_JAX_PREFILL_XLA_OPTIONS = EnvStr("")
    SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS = EnvInt(2048)
    SGLANG_JAX_DECODE_XLA_OPTIONS = EnvStr("")

    # Decode-shaped executables keep all-reduces smaller than this many bytes
    # on the TensorCore (xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes):
    # on v7x the SparseCore all-reduce is 21-26 us slower per call for the
    # [bs, hidden] bf16 decode all-reduces; the offload only pays from ~48 MB
    # (prefill chunks). TPU only. 0 = leave XLA's cost model alone (default);
    # GLM-5.2 serving on v7x sets 33554432 (32 MiB).
    SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES = EnvInt(0)

    # GLM-5: keep the token-embedding table replicated on every device instead
    # of vocab-sharded on "tensor". The sharded lookup is an XLA gather plus one
    # bf16[tokens, hidden] all-reduce per lookup (3 per fused draft-extend step
    # and 1 per verify step at cc64, 63-88 us each; one 96 MiB all-reduce per
    # 8k prefill window); the replicated lookup is a local gather, same values.
    # Cost: +vocab x hidden x 2 B HBM per device (GLM-5.2: 1.9 GB).
    SGLANG_JAX_EMBED_REPLICATED = EnvBool(True)


envs = Envs()
