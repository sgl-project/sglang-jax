"""Guard rejecting chunked prefill under PD disaggregation.

PD prefill replaces process_batch_result with process_prefill_chunk, which
cannot advance a chunked req past its first chunk — it re-prefills chunk1
forever and leaks KV until OOM. Until chunk-prefill-transfer lands, reject
any PD request whose prompt exceeds chunked_prefill_size.
"""

from __future__ import annotations

from types import SimpleNamespace

from sgl_jax.srt.managers.utils import validate_pd_no_chunked_prefill


def _req(seqlen: int) -> SimpleNamespace:
    return SimpleNamespace(origin_input_ids=list(range(seqlen)))


def test_non_pd_mode_never_rejects():
    err = validate_pd_no_chunked_prefill(_req(100000), "null", 4096)
    assert err is None


def test_pd_under_limit_passes():
    err = validate_pd_no_chunked_prefill(_req(4096), "prefill", 4096)
    assert err is None


def test_pd_over_limit_rejected():
    err = validate_pd_no_chunked_prefill(_req(4097), "prefill", 4096)
    assert err is not None
    assert "chunked_prefill_size" in err


def test_pd_decode_mode_over_limit_rejected():
    err = validate_pd_no_chunked_prefill(_req(5000), "decode", 4096)
    assert err is not None


def test_disabled_chunked_prefill_never_rejects():
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", None) is None
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", 0) is None
    assert validate_pd_no_chunked_prefill(_req(100000), "prefill", -1) is None
