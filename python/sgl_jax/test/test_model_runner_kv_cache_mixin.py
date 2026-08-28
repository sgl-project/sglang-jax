import types

import jax.numpy as jnp
import pytest

from sgl_jax.srt.mem_cache.mla_cache_layout import MLACacheLayout
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
        self.embedding_pool_bytes = 0
        self._num_layers = num_layers
        self.server_args = ServerArgs(model_path="dummy", attention_backend=attention_backend)
        # GLM-5.2 shape: "full" at layers 0-2, then every 4th, giving 21 full layers.
        indexer_types = ["shared"] * num_layers
        for i in (0, 1, 2, *range(6, num_layers, 4)):
            indexer_types[i] = "full"
        self.model_config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(),
            hf_text_config=types.SimpleNamespace(
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                num_hidden_layers=num_layers,
                index_head_dim=index_head_dim,
                index_skip_topk_offset=3,
                indexer_types=indexer_types,
            ),
        )

    def _kv_pool_layer_count(self):
        return self._num_layers


def _allocated_bytes_per_page(kv_dim, page_size=128):
    """Bytes the kernel-native shape allocates for one logical page.

    Derived from `get_kv_cache_shape` — what `_create_buffers` allocates with —
    so these assertions pin the budget against the allocation, and follow it if
    the alignment or packing rule changes.
    """
    import jax.numpy as jnp

    from sgl_jax.srt.kernels.mla.v2.kernel import get_kv_cache_shape

    pages, rows, packing, dim = get_kv_cache_shape(
        total_num_pages=1, page_size=page_size, kv_dim=kv_dim, kv_dtype=jnp.bfloat16
    )
    return pages * rows * packing * dim * jnp.dtype(jnp.bfloat16).itemsize


def _allocated_cell_size(runner):
    """Conservative integer bytes/token derived from complete physical pages."""
    cfg = runner.model_config.hf_text_config
    latent_dim = ((cfg.kv_lora_rank + 127) // 128 * 128) + (
        (cfg.qk_rope_head_dim + 127) // 128 * 128
    )
    indexer_dim, num_indexer_layers = runner._dsa_indexer_cache_params()
    page_bytes = _allocated_bytes_per_page(latent_dim, runner.page_size) * runner._num_layers
    if indexer_dim:
        page_bytes += _allocated_bytes_per_page(indexer_dim, runner.page_size) * num_indexer_layers
    return (page_bytes + runner.page_size - 1) // runner.page_size


# GLM-5.2: latent width align128(512) + align128(64) = 640, over 78 layers.
_MLA_BYTES_PER_TOKEN = _allocated_bytes_per_page(640) // 128 * 78
# Indexer keys: align128(128) = 128, over the 21 "full" layers.
_INDEXER_BYTES_PER_TOKEN = _allocated_bytes_per_page(128) // 128 * 21


def test_mla_cache_layout_exposes_latent_and_indexer_roles():
    """Callers select the semantic buffer role; feature width never selects it."""
    import jax.numpy as jnp

    layout = MLACacheLayout(
        page_size=128,
        dtype=jnp.bfloat16,
        kv_lora_rank=192,
        qk_rope_head_dim=64,
        indexer_key_dim=128,
    )

    assert layout.latent_shape(total_num_pages=2) == (2, 64, 2, 384)
    assert layout.indexer_shape(total_num_pages=2) == (2, 64, 2, 128)


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


@pytest.mark.parametrize("page_size", [1, 128])
def test_cell_size_pads_page_size_to_the_kv_packing_boundary(page_size):
    """With bf16 (packing=2) and page_size=1 a page holds 2 slots for 1 token,
    so the per-token cost is double the naive width. Both terms pay it.

    The production page size verifies the ordinary no-padding case."""
    runner = _CellSizeRunner("dsa_sparse", page_size=page_size)
    assert runner._compute_cell_size() == _allocated_cell_size(runner)


def test_cell_size_rounds_up_fractional_page_cost():
    """Capacity profiling must never truncate a partial byte/token ratio."""
    import jax.numpy as jnp

    layout = MLACacheLayout(
        page_size=3,
        dtype=jnp.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    page_bytes = layout.latent_bytes(total_num_pages=1)

    assert page_bytes % layout.page_size != 0
    assert (
        layout.bytes_per_token(num_latent_layers=1, num_indexer_layers=0)
        == (page_bytes + layout.page_size - 1) // layout.page_size
    )


def test_cell_size_pads_latent_segments_independently():
    """The latent width is align(lora,128) + align(rope,128), NOT
    align(lora + rope, 128) — the kernel slices nope and rope as adjacent
    buffers. lora=192/rope=64 separates the two formulas (384 vs 256); the
    GLM/DeepSeek shape (512/64) does not, since both give 640."""
    cell = _CellSizeRunner("fa", kv_lora_rank=192, qk_rope_head_dim=64)._compute_cell_size()
    assert cell == _allocated_bytes_per_page(256 + 128) // 128 * 78


def _single_device_mesh():
    import jax

    return jax.make_mesh((1,), ("data",))


def _build_pool(runner, size, mesh, dp_size=1):
    """Build the real pool this runner's config would allocate at `size` tokens."""
    import jax.numpy as jnp

    from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

    cfg = runner.model_config.hf_text_config
    indexer_key_dim, num_indexer_layers = runner._dsa_indexer_cache_params()
    return MLATokenToKVPool(
        size=size,
        page_size=runner.page_size,
        dtype=jnp.bfloat16,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        layer_num=runner._kv_pool_layer_count(),
        mesh=mesh,
        dp_size=dp_size,
        indexer_key_dim=indexer_key_dim,
        num_indexer_layers=num_indexer_layers,
    )


def _resident_bytes(pool):
    """Bytes actually held by the pool's live buffers, measured not derived."""
    return sum(b.nbytes for b in pool.kv_buffer) + sum(b.nbytes for b in pool.indexer_key_buffer)


def test_allocated_bytes_match_the_budget_and_the_report():
    """Budget, report, and allocation must agree on one live pool.

    The other tests compare the budget against `get_kv_cache_shape`; this one
    compares it against the bytes of the arrays that actually got allocated,
    which is the invariant the DSA indexer buffer broke.

    The pool allocates `size + page_size * dp_size` tokens' worth of pages —
    one spare page per DP rank, per layer — which `_compute_cell_size` has
    never counted. That slack is pre-existing and independent of DSA; expressing
    it as `cell * page_size * dp_size` keeps it visible instead of absorbed.
    """
    runner = _CellSizeRunner("dsa_sparse", num_layers=8)
    assert runner._dsa_indexer_cache_params()[1] > 0, "test needs a DSA indexer buffer"

    size = 256
    pool = _build_pool(runner, size, _single_device_mesh())
    resident = _resident_bytes(pool)

    # Reporting: what the scheduler's `kvcache` gauge and the startup log show.
    assert pool.get_kv_size_bytes() == resident
    # `mem_usage` (the startup log) is the same total, not a parallel sum.
    assert pool.mem_usage == pytest.approx(resident / 1024**3)

    # Budgeting: what the profiler sizes the pool against.
    cell = runner._compute_cell_size()
    assert resident == cell * (size + runner.page_size * 1)


def test_mla_pool_pytree_round_trip_preserves_layout_contract():
    """JAX transformations must retain the semantic layout after unflattening."""
    import jax

    runner = _CellSizeRunner("dsa_sparse", num_layers=8, index_head_dim=96)
    pool = _build_pool(runner, size=256, mesh=_single_device_mesh())

    leaves, tree = jax.tree_util.tree_flatten(pool)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert pool.cache_layout.indexer_key_dim == 96
    assert pool.indexer_key_dim == 128
    assert restored.cache_layout == pool.cache_layout
    assert restored.indexer_key_dim == pool.indexer_key_dim
    assert restored.cache_layout.latent_shape(1) == pool.cache_layout.latent_shape(1)
    assert restored.cache_layout.indexer_shape(1) == pool.cache_layout.indexer_shape(1)
    assert restored.get_kv_size_bytes() == pool.get_kv_size_bytes()


def test_latent_only_budget_overshoots_the_available_memory():
    """Reproduce the production over-provisioning without a TPU.

    Sizing a DSA pool with the pre-fix, latent-only budget (what a non-DSA
    config is charged) makes it allocate more than the profiler said was
    available — the excess coming out of the activation reserve. The fixed
    budget fits.
    """
    runner = _CellSizeRunner("dsa_sparse", num_layers=8)
    available = 16 * 1024 * 1024
    runner.get_available_device_memory = lambda: available
    runner.mem_fraction_static = 1.0

    budget = runner._profile_available_bytes(total_device_memory=0)
    assert budget == available

    mesh = _single_device_mesh()
    page = runner.page_size

    def pool_size_for(cell_size):
        """Page-aligned token count that fits the budget at this per-token cost."""
        return (budget // cell_size) // page * page

    # Every pool carries one unbudgeted spare page per DP rank per layer
    # (`_create_buffers` allocates `size + page_size * dp_size` tokens). That
    # slack predates DSA and bounds how far a correctly budgeted pool may
    # exceed the profiled budget.
    slack = runner._compute_cell_size() * page

    # Pre-fix: the latent-only cell size, i.e. what a non-DSA config is charged.
    latent_only_cell = _CellSizeRunner("fa", num_layers=8)._compute_cell_size()
    over = _resident_bytes(_build_pool(runner, pool_size_for(latent_only_cell), mesh))
    assert over - budget > slack

    # Post-fix: sized with the indexer counted, only the spare page is left over.
    fitted = _resident_bytes(_build_pool(runner, pool_size_for(runner._compute_cell_size()), mesh))
    assert 0 < fitted <= budget + slack


class _HybridDSARunner(_CellSizeRunner):
    """A DSA runner that also reports as hybrid-recurrent.

    `linear_recurrent_config` is a read-only property derived from `hf_config`;
    overriding it here keeps the test off the per-architecture config sniffing.
    """

    @property
    def linear_recurrent_config(self):
        return types.SimpleNamespace(full_attention_layer_ids=[0, 4])


def test_pool_initialization_rejects_hybrid_dsa():
    """The DSA indexer slot space spans every layer; the hybrid pool indexes
    full-attention layers only, and never exposes `get_indexer_key_buffer`.
    Such a config would allocate and budget the indexer buffers, then die on
    the first full-indexer forward — so refuse it at startup instead."""
    runner = _HybridDSARunner("dsa_sparse", num_layers=8)
    with pytest.raises(ValueError, match="dsa_sparse"):
        runner._init_pools(max_num_reqs=1, dp_size=1)


@pytest.mark.parametrize(("embedding_pool_bytes", "expected"), [(0, 700), (100, 600)])
def test_profile_available_bytes_reserves_static_and_embedding_pool(embedding_pool_bytes, expected):
    runner = types.SimpleNamespace(
        get_available_device_memory=lambda: 900,
        mem_fraction_static=0.8,
        embedding_pool_bytes=embedding_pool_bytes,
        linear_recurrent_config=None,
    )
    assert ModelRunnerKVCacheMixin._profile_available_bytes(runner, 1000) == expected


def test_embedding_pool_bytes_only_for_in_model_prefill():
    from sgl_jax.srt.model_executor.model_runner import _embedding_pool_bytes

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        page_size=64,
        max_prefill_tokens=100,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    assert _embedding_pool_bytes(config, args) == 128 * 8 * 2
    args.disaggregation_mode = "decode"
    assert _embedding_pool_bytes(config, args) == 0
    args.disaggregation_mode = "null"
    args.multimodal = True
    assert _embedding_pool_bytes(config, args) == 0


def test_deepstack_embedding_pool_uses_packed_feature_width():
    from sgl_jax.srt.model_executor.model_runner import (
        ModelRunner,
        _embedding_pool_bytes,
    )

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        page_size=64,
        max_prefill_tokens=100,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    model = types.SimpleNamespace(deepstack_visual_layers=3)
    budget = _embedding_pool_bytes(config, args, multimodal_model=model)
    assert budget == 128 * 32 * 2

    runner = types.SimpleNamespace(
        embedding_pool_bytes=budget,
        server_args=args,
        model_config=config,
        model=model,
        dtype=jnp.bfloat16,
        mesh=None,
        embedding_pool=None,
    )
    ModelRunner._build_embedding_pool(runner)

    assert runner.embedding_pool.hidden == 32
    assert runner.embedding_pool.num_pages == 2
    assert runner.embedding_pool.pages.shape == (2, 64, 32)


def test_embedding_pool_capacity_and_pages_follow_lm_limits():
    from sgl_jax.srt.model_executor.model_runner import (
        ModelRunner,
        _embedding_pool_bytes,
    )

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        page_size=128,
        max_prefill_tokens=129,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    model = types.SimpleNamespace(deepstack_visual_layers=3)
    budget = _embedding_pool_bytes(config, args, multimodal_model=model)
    assert budget == 256 * 32 * 2

    runner = types.SimpleNamespace(
        embedding_pool_bytes=budget,
        server_args=args,
        model_config=config,
        model=model,
        dtype=jnp.bfloat16,
        mesh=None,
        embedding_pool=None,
    )
    ModelRunner._build_embedding_pool(runner)

    assert runner.embedding_pool.hidden == 32
    assert runner.embedding_pool.page_size == 128
    assert runner.embedding_pool.num_pages == 2
