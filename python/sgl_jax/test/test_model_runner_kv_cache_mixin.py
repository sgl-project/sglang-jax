import types

import pytest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
)
from sgl_jax.srt.server_args import ServerArgs


def test_recurrent_state_legacy_disable_radix_cache_passes():
    sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
    _enforce_recurrent_state_server_constraints(sa)  # no raise


def test_recurrent_state_radix_cache_requires_unified_radix_tree():
    sa = ServerArgs(
        model_path="dummy",
        disable_radix_cache=False,
        enable_unified_radix_tree=False,
    )
    with pytest.raises(AssertionError, match="unified-radix-tree"):
        _enforce_recurrent_state_server_constraints(sa)


def _fake_runner(max_recurrent_state_size, dp_size):
    sa = ServerArgs(
        model_path="dummy",
        enable_unified_radix_tree=True,
        disable_radix_cache=False,
        dp_size=dp_size,
        max_recurrent_state_size=max_recurrent_state_size,
    )
    return types.SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=None,
        linear_recurrent_config=object(),
        server_args=sa,
        dp_size=dp_size,
    )


def test_recurrent_admission_cap_is_dp_aligned():
    # factor=2: 12 // 2 = 6 is off the dp_size=4 grid, so the cap rounds down to 4.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(12, 4), 1000)
    assert cap == 4
    assert cap % 4 == 0


def test_recurrent_admission_cap_already_aligned_unchanged():
    # 16 // 2 = 8 is already a dp_size=4 multiple.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(16, 4), 1000)
    assert cap == 8


class _CellSizeRunner(ModelRunnerKVCacheMixin):
    """Minimal stand-in exposing only what _compute_cell_size reads."""

    def __init__(
        self,
        attention_backend,
        num_layers=78,
        page_size=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
    ):
        import jax.numpy as jnp

        self.kv_cache_dtype = jnp.bfloat16
        self.page_size = page_size
        self.use_mla_backend = True
        self._num_layers = num_layers
        self.server_args = ServerArgs(model_path="dummy", attention_backend=attention_backend)
        # GLM-5.2 shape: "full" at layers 0-2, then every 4th, giving 21 full layers.
        indexer_types = ["shared"] * num_layers
        for i in (0, 1, 2, *range(6, num_layers, 4)):
            indexer_types[i] = "full"
        self.model_config = types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                num_hidden_layers=num_layers,
                index_head_dim=index_head_dim,
                index_skip_topk_offset=3,
                indexer_types=indexer_types,
            )
        )

    def _kv_pool_layer_count(self):
        return self._num_layers


def _allocated_bytes_per_token(kv_dim, page_size=128):
    """Per-token bytes the POOL would allocate for a buffer of this width.

    Derived from `get_kv_cache_shape` — what `_create_buffers` allocates with —
    so these assertions pin the budget against the allocation, and follow it if
    the alignment or packing rule changes.
    """
    import jax.numpy as jnp

    from sgl_jax.srt.kernels.mla.v2.kernel import get_kv_cache_shape

    pages, rows, packing, dim = get_kv_cache_shape(
        total_num_pages=1, page_size=page_size, kv_dim=kv_dim, kv_dtype=jnp.bfloat16
    )
    per_page = pages * rows * packing * dim * jnp.dtype(jnp.bfloat16).itemsize
    return per_page // page_size


# GLM-5.2: latent width align128(512) + align128(64) = 640, over 78 layers.
_MLA_BYTES_PER_TOKEN = _allocated_bytes_per_token(640) * 78
# Indexer keys: align128(128) = 128, over the 21 "full" layers.
_INDEXER_BYTES_PER_TOKEN = _allocated_bytes_per_token(128) * 21


def test_cell_size_excludes_indexer_for_non_dsa_mla():
    """`fa` allocates no indexer buffer, so it must not be charged for one."""
    assert _CellSizeRunner("fa")._compute_cell_size() == _MLA_BYTES_PER_TOKEN


def test_cell_size_includes_dsa_indexer_cache():
    """`MLATokenToKVPool._create_buffers` allocates a second paged buffer for
    DSA indexer keys. Budgeting only the latent KV over-provisions the pool, and
    the excess is taken from the activation reserve rather than from KV."""
    cell = _CellSizeRunner("dsa_sparse")._compute_cell_size()
    assert cell == _MLA_BYTES_PER_TOKEN + _INDEXER_BYTES_PER_TOKEN
    # Guard the regression direction explicitly: under-counting here is what
    # silently eats the (1 - mem_fraction) reserve.
    assert cell > _MLA_BYTES_PER_TOKEN


@pytest.mark.parametrize("page_size", [1, 64, 128])
def test_cell_size_pads_page_size_to_the_kv_packing_boundary(page_size):
    """With bf16 (packing=2) and page_size=1 a page holds 2 slots for 1 token,
    so the per-token cost is double the naive width. Both terms pay it."""
    cell = _CellSizeRunner("dsa_sparse", page_size=page_size)._compute_cell_size()
    assert cell == _allocated_bytes_per_token(640, page_size) * 78 + (
        _allocated_bytes_per_token(128, page_size) * 21
    )


def test_cell_size_pads_latent_segments_independently():
    """The latent width is align(lora,128) + align(rope,128), NOT
    align(lora + rope, 128) — the kernel slices nope and rope as adjacent
    buffers. lora=192/rope=64 separates the two formulas (384 vs 256); the
    GLM/DeepSeek shape (512/64) does not, since both give 640."""
    cell = _CellSizeRunner("fa", kv_lora_rank=192, qk_rope_head_dim=64)._compute_cell_size()
    assert cell == _allocated_bytes_per_token(256 + 128) * 78


def test_dsa_indexer_params_report_the_full_layer_count():
    """21 of GLM-5.2's 78 layers are "full" and each gets an indexer buffer."""
    assert _CellSizeRunner("dsa_sparse")._dsa_indexer_cache_params() == (128, 21)
    assert _CellSizeRunner("fa")._dsa_indexer_cache_params() == (0, 0)
