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


# ---------------------------------------------------------------------------
# _compute_cell_size must budget the DSA indexer key cache
# ---------------------------------------------------------------------------


class _CellSizeRunner(ModelRunnerKVCacheMixin):
    """Minimal stand-in exposing only what _compute_cell_size reads."""

    def __init__(self, attention_backend, num_layers=78, page_size=128):
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
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_hidden_layers=num_layers,
                index_head_dim=128,
                index_skip_topk_offset=3,
                indexer_types=indexer_types,
            )
        )

    def _kv_pool_layer_count(self):
        return self._num_layers


# align128(512) + align128(64) = 640 dims; 640 * 128 * 2 // 128 = 1280 B/layer.
_MLA_BYTES_PER_TOKEN = 1280 * 78
# The indexer cache uses the same layout with kv_dim = align128(128) = 128,
# i.e. 256 B/layer over the 21 "full" layers.
_INDEXER_BYTES_PER_TOKEN = 256 * 21


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


def test_dsa_indexer_params_match_index_share_map():
    """The helper must agree with the map the pool itself is built from."""
    from sgl_jax.srt.kernels.dsa.ref import build_index_share_map

    runner = _CellSizeRunner("dsa_sparse")
    cfg = runner.model_config.hf_text_config
    _, _, num_full = build_index_share_map(
        cfg.indexer_types, cfg.index_skip_topk_offset, cfg.num_hidden_layers
    )
    assert runner._dsa_indexer_cache_params() == (cfg.index_head_dim, num_full)
    assert _CellSizeRunner("fa")._dsa_indexer_cache_params() == (0, 0)
