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

    # DSA prefill: EXTEND batches whose longest sequence is <= this many tokens
    # run plain dense MLA attention instead of indexer page-top-k + sparse
    # attend (the top-k would select every page anyway, so it is the same
    # function without the scoring). Clamped to the model's index_topk by the
    # backend. 0 = always sparse. GLM-5.2 production: 2048.
    # 0 = always the sparse arm (default); GLM-5.2 serving on v7x sets 2048.
    DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT = EnvInt(0)

    # DSA decode: batches with at least this many requests reshard queries
    # head-TP -> request-sharded (all_to_all inside the shard_map), attend only
    # the local request slice and a2a back (a2a-DP attend; the decode twin of
    # the prefill TSA sandwich). Below it the head-TP per-request attend stays
    # (two a2a are pure latency at bs1). 0 = never. GLM-5.2 production: 64.
    # 0 = never (default); GLM-5.2 serving on v7x sets 64.
    DSA_DP_DECODE_BS = EnvInt(0)

    # DSA decode page top-k identity shortcut: when the page budget covers the
    # whole KV bucket the top-k selects every valid page anyway, so skip the
    # gather + scoring + top_k and return the pages in position order -- but
    # only for buckets of at most this many pages (10-04 gp380: the attend
    # kernel's cost under identity grows with pages; the 32-page bucket lost
    # 15 % throughput while 16-page buckets gained). 0 = no cap, -1 = never.
    DSA_PAGE_TOPK_IDENTITY_MAX_PAGES = EnvInt(32)

    # MLA v2 attend on EXTEND batches takes head-major q/o (the kernel rows are
    # a permutation of the token-major order; bit-identical) so the W_UK dot
    # output feeds the kernel and the kernel output feeds the W_UV dot without
    # the two tile-level relayouts XLA otherwise inserts around the custom call.
    # Decode / verify keep the token-major path. Escape hatch only.
    # Opt-in (default off): the head-major form needs max_num_tokens and the tuned
    # bq to be multiples of 16 and raises otherwise. GLM-5.2 serving on v7x sets 1.
    SGLANG_JAX_MLA_Q_HEAD_MAJOR = EnvBool(False)

    # Decode KV capacity ladder (non-speculative decode): per-request cache_loc
    # capacity steps (tokens, page-aligned) so short-context decode batches run
    # on smaller cache_loc shapes than max_req_len; the max-capacity bucket
    # stays the fallback and the same steps are precompiled. Batches below
    # LADDER_MIN_BS requests skip steps under LADDER_SMALLBS_FLOOR (bs1 at the
    # 2048 step was +0.16 ms/step slower than at 4096). Empty = off (default:
    # every rung is one more decode executable to precompile per bs bucket, so
    # the ladder is opt-in; GLM-5.2 serving on v7x uses 2048,4096,8192,16384,65536).
    SGLANG_JAX_DECODE_KV_LADDER = EnvStr("")
    SGLANG_JAX_DECODE_KV_LADDER_MIN_BS = EnvInt(8)
    SGLANG_JAX_DECODE_KV_LADDER_SMALLBS_FLOOR = EnvInt(4096)


envs = Envs()
