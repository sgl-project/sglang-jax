import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention import ragged_paged_attention_v3 as rpa


def test_decode_block_sizes_keep_nonzero_kv_chunk_for_empty_page_bucket(monkeypatch):
    monkeypatch.setattr(rpa, "get_tpu_version", lambda: 7)

    block_sizes = rpa.get_default_block_sizes(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=8,
        actual_num_kv_heads=2,
        head_dim=128,
        page_size=64,
        max_num_tokens=1,
        max_num_seqs=1,
        pages_per_seq=0,
        case=rpa.RpaCase.DECODE,
    )

    assert block_sizes == {"bq_sz": 1, "bkv_sz": 64, "bq_csz": 1, "bkv_csz": 64}
