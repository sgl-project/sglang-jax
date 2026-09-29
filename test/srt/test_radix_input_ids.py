from types import SimpleNamespace

from sgl_jax.srt.managers.utils import validate_input_length
from sgl_jax.srt.mem_cache.radix_cache import build_radix_key


def test_auto_truncate_preserves_radix_input_ids_invariant():
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3, 4],
        radix_input_ids=[1, 101, 101, 4],
    )

    validate_input_length(req, max_req_input_len=3, allow_auto_truncate=True)
    assert req.origin_input_ids == [1, 2, 3]
    assert req.radix_input_ids == [1, 101, 101]


def test_radix_key_uses_only_canonical_prompt_identity():
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        radix_input_ids=[1, 101, 101],
        output_ids=[4],
        fill_ids=[999, 999, 999, 999],
        extra_key="adapter",
        dp_rank=1,
    )

    key = build_radix_key(req, key_len=4)
    assert key.token_ids == [1, 101, 101, 4]
    assert (key.extra_key, key.dp_rank) == ("adapter", 1)
