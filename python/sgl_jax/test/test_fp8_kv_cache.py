"""Tests for FP8 KV cache support in sglang-jax."""

import types
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.kernels.mla.v2.tuned_block_sizes import get_tuned_block_sizes_mla
from sgl_jax.srt.kernels.paged_attention.paged_attention import paged_attention
from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import (
    get_tuned_block_sizes_v3,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import (
    align_to,
    get_tpu_version,
    is_fp8_dtype,
    quantize_to_kv_dtype,
)
from sgl_jax.srt.mem_cache.hca_pool import HCAKVPool
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MSATokenToKVPool,
    merge_kv,
)
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sgl_jax.srt.server_args import ServerArgs


class _DummyRunner(ModelRunnerKVCacheMixin):
    """Minimal runner stub for testing _init_kv_cache_dtype and _compute_cell_size."""

    def __init__(
        self,
        kv_cache_dtype_str="auto",
        attention_backend="fa",
        is_msa=False,
        num_layers=4,
        actual_num_kv_heads=2,
        head_dim=128,
        page_size=128,
        swa_ratio=0.8,
        swa_layers=0,
    ):
        self.dtype = jnp.bfloat16
        self.server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype=kv_cache_dtype_str,
            attention_backend=attention_backend,
            swa_full_tokens_ratio=swa_ratio,
        )
        self.is_msa = is_msa
        self.num_layers = num_layers
        self.actual_num_kv_heads = actual_num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.use_mla_backend = False
        self.embedding_pool_bytes = 0
        self.attention_tp_size = 1
        self.attn_backend = None

        hf_config = types.SimpleNamespace()
        if swa_layers > 0:
            hf_config.swa_num_key_value_heads = actual_num_kv_heads

        self.model_config = types.SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=types.SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_hidden_layers=num_layers,
            ),
            head_dim=head_dim,
            get_num_kv_heads=lambda tp_size: actual_num_kv_heads,
            get_hybrid_layer_counts=lambda: (swa_layers, num_layers - swa_layers),
        )

        # Layer configuration
        self.is_swa_layer = [i < swa_layers for i in range(num_layers)]
        self._init_kv_cache_dtype()

    def _kv_pool_layer_count(self):
        return self.num_layers


# ==============================================================================
# 1. Quantize Helper Tests
# ==============================================================================


def test_quantize_to_kv_dtype_e4m3fn_bounds():
    """Verify saturating clipping to float8_e4m3fn finite range [-448, 448]."""
    x = jnp.array([500.0, -10000.0, 448.0, -448.0, 100.0, 0.0], dtype=jnp.float32)
    q = quantize_to_kv_dtype(x, jnp.float8_e4m3fn)
    assert q.dtype == jnp.float8_e4m3fn
    assert not jnp.isnan(q.astype(jnp.float32)).any(), "Quantized tensor contains NaNs"

    q_f32 = q.astype(jnp.float32)
    assert q_f32[0] == 448.0
    assert q_f32[1] == -448.0
    assert q_f32[2] == 448.0
    assert q_f32[3] == -448.0


def test_quantize_to_kv_dtype_e5m2_bounds():
    """Verify saturating clipping to float8_e5m2 max finite value 57344.0."""
    x = jnp.array([100000.0, -100000.0, 57344.0, 10.0], dtype=jnp.float32)
    q = quantize_to_kv_dtype(x, jnp.float8_e5m2)
    assert q.dtype == jnp.float8_e5m2
    assert not jnp.isnan(q.astype(jnp.float32)).any(), "Quantized tensor contains NaNs"

    q_f32 = q.astype(jnp.float32)
    assert q_f32[0] == 57344.0
    assert q_f32[1] == -57344.0


def test_quantize_to_kv_dtype_bf16_identity():
    """Verify quantize helper is a strict identity / no-op for bfloat16."""
    x = jnp.array([1.5, 2.5, -3.5, 1000.0], dtype=jnp.bfloat16)
    q = quantize_to_kv_dtype(x, jnp.bfloat16)
    assert q.dtype == jnp.bfloat16
    assert jnp.array_equal(q, x)


def test_quantize_to_kv_dtype_scale_division():
    """Verify scaling behavior on write path."""
    x = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float32)
    # With scale 2.0, effective inputs are [1.0, 2.0, 4.0]
    q = quantize_to_kv_dtype(x, jnp.float8_e4m3fn, scale=2.0)
    expected = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float8_e4m3fn)
    assert jnp.array_equal(q, expected)

    # Scale None or 1.0 does not scale
    q_none = quantize_to_kv_dtype(x, jnp.float8_e4m3fn, scale=None)
    q_one = quantize_to_kv_dtype(x, jnp.float8_e4m3fn, scale=1.0)
    assert jnp.array_equal(q_none, q_one)


def test_is_fp8_dtype():
    """Verify is_fp8_dtype helper accurately detects float8 types."""
    assert is_fp8_dtype(jnp.float8_e4m3fn) is True
    assert is_fp8_dtype(jnp.float8_e5m2) is True
    assert is_fp8_dtype(jnp.bfloat16) is False
    assert is_fp8_dtype(jnp.float32) is False
    assert is_fp8_dtype(jnp.int32) is False
    assert is_fp8_dtype(None) is False


# ==============================================================================
# 2. Dtype Parsing and ServerArgs Tests
# ==============================================================================


@pytest.mark.parametrize(
    "kv_str,expected_dtype",
    [
        ("auto", jnp.bfloat16),
        ("bf16", jnp.bfloat16),
        ("bfloat16", jnp.bfloat16),
        ("fp8", jnp.float8_e4m3fn),
        ("fp8_e4m3", jnp.float8_e4m3fn),
        ("fp8_e5m2", jnp.float8_e5m2),
    ],
)
def test_init_kv_cache_dtype_mapping(kv_str, expected_dtype):
    runner = _DummyRunner(kv_cache_dtype_str=kv_str)
    assert runner.kv_cache_dtype == expected_dtype


def test_init_kv_cache_dtype_invalid_value():
    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
        _DummyRunner(kv_cache_dtype_str="int8")


def test_server_args_kv_cache_dtype_choices():
    sa = ServerArgs(model_path="dummy", kv_cache_dtype="fp8")
    assert sa.kv_cache_dtype == "fp8"

    sa_e4m3 = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e4m3")
    assert sa_e4m3.kv_cache_dtype == "fp8_e4m3"

    sa_e5m2 = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e5m2")
    assert sa_e5m2.kv_cache_dtype == "fp8_e5m2"


# ==============================================================================
# 3. P1-1: BF16 Strict No-Op Invariant Tests
# ==============================================================================


@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("actual_kv_heads", [1, 2, 4, 8])
def test_bf16_strict_no_op_cell_size(head_dim, actual_kv_heads):
    """Assert BF16 cell size matches the origin/main formula exactly."""
    num_layers = 8
    page_size = 128

    # Formula from origin/main:
    # cell_size = num_layers * (2 * actual_kv_heads) * align128(head_dim) * itemsize
    # where itemsize = 2 for bfloat16
    head_dim_aligned = (head_dim + 127) // 128 * 128
    expected_main_cell_size = num_layers * (actual_kv_heads * 2) * head_dim_aligned * 2

    runner = _DummyRunner(
        kv_cache_dtype_str="bf16",
        num_layers=num_layers,
        actual_num_kv_heads=actual_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    assert runner._compute_cell_size() == expected_main_cell_size


@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("actual_kv_heads", [1, 2, 4, 8])
def test_bf16_strict_no_op_swa_cell_size(head_dim, actual_kv_heads):
    """Assert BF16 cell size for SWA layers matches the origin/main formula."""
    num_layers = 8
    swa_layers = 4
    page_size = 128
    swa_ratio = 0.5

    head_dim_aligned = (head_dim + 127) // 128 * 128
    expected_main_cell_size = num_layers * (actual_kv_heads * 2) * head_dim_aligned * 2

    runner = _DummyRunner(
        kv_cache_dtype_str="bf16",
        num_layers=num_layers,
        actual_num_kv_heads=actual_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        swa_ratio=swa_ratio,
        swa_layers=swa_layers,
    )
    assert runner._compute_cell_size() == expected_main_cell_size


@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("actual_kv_heads", [1, 2, 4, 8])
def test_bf16_strict_no_op_pool_construction(head_dim, actual_kv_heads):
    """Assert real MHATokenToKVPool construction for BF16 yields identical shape and bytes to main."""
    num_layers = 4
    page_size = 128
    cache_size = 256
    dp_size = 1
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("data", "tensor"))

    pool = MHATokenToKVPool(
        size=cache_size,
        page_size=page_size,
        dtype=jnp.bfloat16,
        head_num=actual_kv_heads,
        head_dim=head_dim,
        layer_num=num_layers,
        mesh=mesh,
        dp_size=dp_size,
        abstract=True,
    )

    # Main formula:
    # packing = 2 (for bf16)
    # shape: [(size + page_size * dp_size) // page_size, page_size, actual_kv_heads, 2, align_to(head_dim, 128)]
    expected_shape = (
        (cache_size + page_size * dp_size) // page_size,
        page_size,
        actual_kv_heads,
        2,
        align_to(head_dim, 128),
    )
    assert pool.kv_buffer[0].shape == expected_shape
    assert pool.kv_buffer[0].dtype == jnp.bfloat16

    expected_bytes = (
        (cache_size + page_size * dp_size)
        * (actual_kv_heads * 2)
        * align_to(head_dim, 128)
        * 2
        * num_layers
    )
    expected_k_bytes = expected_bytes // 2
    assert pool.get_kv_size_bytes() == (expected_k_bytes, expected_k_bytes)


# ==============================================================================
# 4. FP8 Pool Geometry and Padding Tests
# ==============================================================================


@pytest.mark.parametrize("actual_kv_heads", [1, 2, 4, 8])
def test_fp8_vs_bf16_pool_sizes(actual_kv_heads):
    """Verify FP8 packing=4 per-shard head alignment and memory savings."""
    num_layers = 4
    head_dim = 128
    page_size = 128

    runner_bf16 = _DummyRunner(
        kv_cache_dtype_str="bf16",
        num_layers=num_layers,
        actual_num_kv_heads=actual_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    runner_fp8 = _DummyRunner(
        kv_cache_dtype_str="fp8",
        num_layers=num_layers,
        actual_num_kv_heads=actual_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )

    bf16_cell_size = runner_bf16._compute_cell_size()
    fp8_cell_size = runner_fp8._compute_cell_size()

    # When actual_kv_heads == 1:
    # BF16: heads_x2 = 2, packing = 2 -> 2 * head_dim * 2 = 4 * head_dim bytes
    # FP8: heads_x2 = 2, align_to(2, 4) = 4, packing = 4 -> 4 * head_dim * 1 = 4 * head_dim bytes
    # Bytes are equal because of alignment padding.
    if actual_kv_heads == 1:
        assert fp8_cell_size == bf16_cell_size
    else:
        # For 2, 4, 8 heads: FP8 achieves exactly 50% memory consumption of BF16.
        assert fp8_cell_size == bf16_cell_size // 2


def test_merge_kv_padding():
    """Verify merge_kv pads heads to align_to(num_heads_x2, packing)."""
    # 1 KV head per shard: k, v shape (tokens, heads, hdim)
    k = jnp.ones((2, 1, 128), dtype=jnp.float8_e4m3fn)
    v = jnp.full((2, 1, 128), 2.0, dtype=jnp.float8_e4m3fn)

    # For FP8, packing = 4: heads_x2 = 2 -> padded to 4
    # Output shape: [tokens, 1, heads_x2 // packing, packing, head_dim] -> [2, 1, 1, 4, 128]
    merged_fp8 = merge_kv(k, v)
    assert merged_fp8.shape == (2, 1, 1, 4, 128)
    assert jnp.array_equal(merged_fp8[:, :, 0, 0, :], k[:, :, :])
    assert jnp.array_equal(merged_fp8[:, :, 0, 1, :], v[:, :, :])
    assert (merged_fp8[:, :, 0, 2:, :] == 0).all()

    # For BF16, packing = 2: heads_x2 = 2 -> already aligned to 2
    # Output shape: [tokens, 1, 1, 2, 128]
    k_bf16 = jnp.ones((2, 1, 128), dtype=jnp.bfloat16)
    v_bf16 = jnp.full((2, 1, 128), 2.0, dtype=jnp.bfloat16)
    merged_bf16 = merge_kv(k_bf16, v_bf16)
    assert merged_bf16.shape == (2, 1, 1, 2, 128)
    assert jnp.array_equal(merged_bf16[:, :, 0, 0, :], k_bf16[:, :, :])
    assert jnp.array_equal(merged_bf16[:, :, 0, 1, :], v_bf16[:, :, :])


# ==============================================================================
# 5. P1-3: Tuned Table Fallback Tests
# ==============================================================================


@patch(
    "sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3.get_tpu_version",
    return_value=6,
)
@patch(
    "sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes.get_device_name",
    return_value="TPU v6e",
)
def test_tuned_block_sizes_v3_fallback(mock_dev, mock_ver):
    """Verify get_tuned_block_sizes_v3 falls back to bfloat16 entry on float8 miss."""
    # Key present in TUNED_BLOCK_SIZES_V3["TPU v6e"]:
    # ("d", 128, "bfloat16", "bfloat16", 16, 1, 256, 128, 1) -> (1, 256, 1, 256)
    hit_e4m3 = get_tuned_block_sizes_v3(
        stage="d",
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.float8_e4m3fn,
        actual_num_q_heads=16,
        actual_num_kv_heads=1,
        head_dim=256,
        page_size=128,
        max_num_tokens=1,
        sliding_window=128,
    )
    assert hit_e4m3 == (1, 256, 1, 256)

    # float8_e5m2 should also fall back to bfloat16
    hit_e5m2 = get_tuned_block_sizes_v3(
        stage="d",
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.float8_e5m2,
        actual_num_q_heads=16,
        actual_num_kv_heads=1,
        head_dim=256,
        page_size=128,
        max_num_tokens=1,
        sliding_window=128,
    )
    assert hit_e5m2 == (1, 256, 1, 256)

    # Missing shape returns None
    miss = get_tuned_block_sizes_v3(
        stage="d",
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.float8_e4m3fn,
        actual_num_q_heads=8,
        actual_num_kv_heads=2,
        head_dim=999,
        page_size=128,
        max_num_tokens=1,
    )
    assert miss is None


@patch("sgl_jax.srt.kernels.mla.v2.tuned_block_sizes.get_tpu_version", return_value=6)
@patch("sgl_jax.srt.kernels.mla.v2.tuned_block_sizes.get_device_name", return_value="TPU v6e")
def test_tuned_block_sizes_mla_fallback(mock_dev, mock_ver):
    """Verify get_tuned_block_sizes_mla falls back to bfloat16 entry on float8 miss."""
    # From TUNED_BLOCK_SIZES_MLA["TPU v6e"]:
    # ("decode", "bfloat16", "bfloat16", 16, 512, 64, 128, 1) -> (16, 1, 2)
    hit = get_tuned_block_sizes_mla(
        case_label="decode",
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.float8_e4m3fn,
        actual_num_q_heads=16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=128,
        max_num_tokens=1,
    )
    assert hit == (16, 1, 2)


# ==============================================================================
# 6. P0-3: Unsupported Backends Guard Tests
# ==============================================================================


def test_msa_pool_rejects_fp8():
    """MSATokenToKVPool must raise ValueError for FP8 dtypes."""
    with pytest.raises(ValueError, match="MSATokenToKVPool does not support FP8"):
        MSATokenToKVPool(
            sparse_layer_ids=[0],
            index_head_dim=128,
            mesh=None,
            actual_num_kv_heads=2,
            head_dim=128,
            page_size=128,
            size=10,
            dtype=jnp.float8_e4m3fn,
        )


def test_hca_pool_rejects_fp8():
    """HCAKVPool must raise ValueError for FP8 dtypes."""
    with pytest.raises(ValueError, match="HCAKVPool does not support FP8"):
        HCAKVPool(
            size=128,
            page_size=128,
            dtype=jnp.float8_e4m3fn,
            layer_num=1,
            mesh=None,
            max_num_requests=256,
            max_context_len=4096,
        )


def test_recurrent_state_pool_rejects_fp8():
    """RecurrentStatePool must raise ValueError for FP8 temporal or conv dtypes."""
    with pytest.raises(ValueError, match="RecurrentStatePool does not support FP8"):
        RecurrentStatePool(
            linear_recurrent_layer_ids=[0],
            size=8,
            num_heads=2,
            head_dim=128,
            conv_kernel_size=4,
            mesh=None,
            temporal_dtype=jnp.float8_e4m3fn,
        )


def test_paged_attention_kernel_rejects_fp8():
    """paged_attention kernel must raise ValueError for FP8 dtypes."""
    dummy_pages = jnp.zeros((2, 128, 2, 128), dtype=jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="does not support FP8"):
        paged_attention(
            q=jnp.zeros((1, 2, 128), dtype=jnp.bfloat16),
            k_pages=dummy_pages,
            v_pages=dummy_pages,
            block_tables=jnp.zeros((1, 1), dtype=jnp.int32),
            lengths=jnp.array([1], dtype=jnp.int32),
        )


def test_runner_rejects_tt_backend_fp8():
    """ModelRunnerKVCacheMixin must raise ValueError if backend is 'tt' with FP8."""
    with pytest.raises(ValueError, match="TT attention backend does not support FP8"):
        _DummyRunner(kv_cache_dtype_str="fp8", attention_backend="tt")


# ==============================================================================
# 7. TPU-Only Kernel Parity & Outlier Injection Tests
# ==============================================================================


def _is_tpu():
    return get_tpu_version() > 0


def test_rpav3_and_mla_signatures_binding():
    """Validate that RPAv3 and MLA test calls strictly conform to kernel signatures (runs on CPU)."""
    import inspect

    from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention

    # 1. RPA signature bind
    sig_rpa = inspect.signature(ragged_paged_attention)
    bound_rpa = sig_rpa.bind(
        jnp.zeros((128, 8, 128), jnp.bfloat16),
        jnp.zeros((128, 2, 128), jnp.bfloat16),
        jnp.zeros((128, 2, 128), jnp.bfloat16),
        jnp.zeros((1, 128, 1, 4, 128), jnp.float8_e4m3fn),
        jnp.array([128], jnp.int32),
        jnp.array([0], jnp.int32),
        jnp.array([0, 128], jnp.int32),
        jnp.array([0, 128], jnp.int32),
        jnp.array([0, 1, 1], jnp.int32),
        None,
        sm_scale=0.125,
    )
    assert bound_rpa is not None

    # 2. MLA signature bind
    sig_mla = inspect.signature(mla_ragged_paged_attention)
    bound_mla = sig_mla.bind(
        jnp.zeros((1, 16, 512), jnp.bfloat16),
        jnp.zeros((1, 16, 64), jnp.bfloat16),
        jnp.zeros((1, 512), jnp.float8_e4m3fn),
        jnp.zeros((1, 64), jnp.float8_e4m3fn),
        jnp.zeros((1, 32, 4, 640), jnp.float8_e4m3fn),
        jnp.array([128], jnp.int32),
        jnp.array([0], jnp.int32),
        jnp.array([0, 1], jnp.int32),
        jnp.array([0, 128], jnp.int32),
        jnp.array([1, 1, 1], jnp.int32),
        sm_scale=0.125,
    )
    assert bound_mla is not None


@pytest.mark.skipif(not _is_tpu(), reason="TPU hardware required for RPAv3 kernel tests")
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_rpav3_parity_tpu(kv_heads):
    """TPU parity test: compare RPAv3 output between BF16 and FP8 (cosine >= 0.995)."""
    q_heads = 8
    head_dim = 128
    page_size = 128
    total_tokens = 128
    key = jax.random.PRNGKey(42)

    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    k = jax.random.normal(k2, (total_tokens, kv_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    v = jax.random.normal(k3, (total_tokens, kv_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )

    kv_lens = jnp.array([total_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, total_tokens], dtype=jnp.int32)
    cu_kv_lens = jnp.array([0, total_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    sm_scale = 1.0 / (head_dim**0.5)

    # 1. BF16 Reference
    cache_bf16 = jnp.zeros((1, page_size, kv_heads, 2, head_dim), dtype=jnp.bfloat16)
    out_bf16, _ = ragged_paged_attention(
        jnp.copy(q),
        jnp.copy(k),
        jnp.copy(v),
        cache_bf16,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        None,
        sm_scale=sm_scale,
    )

    # 2. FP8
    padded_heads_x2_fp8 = align_to(kv_heads * 2, 4)
    cache_fp8 = jnp.zeros(
        (1, page_size, padded_heads_x2_fp8 // 4, 4, head_dim), dtype=jnp.float8_e4m3fn
    )
    out_fp8, _ = ragged_paged_attention(
        jnp.copy(q),
        jnp.copy(k),
        jnp.copy(v),
        cache_fp8,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        None,
        sm_scale=sm_scale,
    )

    assert not jnp.isnan(out_fp8).any(), "FP8 kernel produced NaNs"

    # Compute cosine similarity
    u = out_bf16.astype(jnp.float32).flatten()
    w = out_fp8.astype(jnp.float32).flatten()
    cos_sim = float(jnp.dot(u, w) / (jnp.linalg.norm(u) * jnp.linalg.norm(w) + 1e-8))
    assert cos_sim >= 0.995, f"Cosine similarity {cos_sim} < 0.995 for kv_heads={kv_heads}"


@pytest.mark.skipif(not _is_tpu(), reason="TPU hardware required for RPAv3 outlier tests")
def test_rpav3_outlier_injection_tpu():
    """Verify large K outliers (+/-600) saturate to finite bounds and do not cause NaNs."""
    q_heads = 4
    kv_heads = 2
    head_dim = 128
    page_size = 128
    total_tokens = 64

    q = jnp.ones((total_tokens, q_heads, head_dim), dtype=jnp.bfloat16)
    k = jnp.ones((total_tokens, kv_heads, head_dim), dtype=jnp.bfloat16)
    # Inject outliers beyond float8_e4m3fn finite limit 448.0
    k = k.at[0, :, :10].set(600.0)
    k = k.at[1, :, :10].set(-600.0)
    v = jnp.ones((total_tokens, kv_heads, head_dim), dtype=jnp.bfloat16)

    k_quant = quantize_to_kv_dtype(k, jnp.float8_e4m3fn)
    assert not jnp.isnan(k_quant.astype(jnp.float32)).any()
    assert (k_quant.astype(jnp.float32) <= 448.0).all()
    assert (k_quant.astype(jnp.float32) >= -448.0).all()

    kv_lens = jnp.array([total_tokens], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, total_tokens], dtype=jnp.int32)
    cu_kv_lens = jnp.array([0, total_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    sm_scale = 1.0 / (head_dim**0.5)

    padded_heads_x2_fp8 = align_to(kv_heads * 2, 4)
    cache_fp8 = jnp.zeros(
        (1, page_size, padded_heads_x2_fp8 // 4, 4, head_dim), dtype=jnp.float8_e4m3fn
    )
    out_fp8, _ = ragged_paged_attention(
        jnp.copy(q),
        jnp.copy(k),
        jnp.copy(v),
        cache_fp8,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        None,
        sm_scale=sm_scale,
    )
    assert not jnp.isnan(out_fp8).any(), "Outlier injection produced NaNs in RPAv3 output"
    assert jnp.isfinite(out_fp8).all(), "Output contains infinite values"


@pytest.mark.skipif(not _is_tpu(), reason="TPU hardware required for MLA kernel tests")
def test_mla_parity_tpu():
    """TPU parity test: compare MLA v2 decode kernel output between BF16 and FP8 (cosine >= 0.995).

    Note: The MLA mixed-mode tuned entry (bq=128, bkvp=16, page_size=128) exceeds v7x scoped
    VMEM (32M) even for the BF16 reference (pre-existing upstream table issue, out of scope).
    This test validates the decode path (distribution=[total_tokens, 1, 1] with total_tokens=1).
    """
    from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention

    num_heads = 16
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    page_size = 128
    total_tokens = 1
    key = jax.random.PRNGKey(42)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    ql_nope = jax.random.normal(k1, (total_tokens, num_heads, kv_lora_rank), dtype=jnp.bfloat16)
    q_pe = jax.random.normal(k2, (total_tokens, num_heads, qk_rope_head_dim), dtype=jnp.bfloat16)
    new_kv_c = jax.random.normal(k3, (total_tokens, kv_lora_rank), dtype=jnp.bfloat16)
    new_k_pe = jax.random.normal(k4, (total_tokens, qk_rope_head_dim), dtype=jnp.bfloat16)

    kv_lens = jnp.array([128], dtype=jnp.int32)
    page_indices = jnp.array([0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, total_tokens], dtype=jnp.int32)
    cu_kv_lens = jnp.array([0, 128], dtype=jnp.int32)
    distribution = jnp.array([total_tokens, 1, 1], dtype=jnp.int32)
    sm_scale = 1.0 / ((128 + qk_rope_head_dim) ** 0.5)

    aligned_head_dim = align_to(kv_lora_rank, 128) + align_to(qk_rope_head_dim, 128)

    # 1. BF16 Reference
    cache_bf16 = jnp.zeros((1, page_size // 2, 2, aligned_head_dim), dtype=jnp.bfloat16)
    out_bf16, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_bf16,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale=sm_scale,
    )

    # 2. FP8
    cache_fp8 = jnp.zeros((1, page_size // 4, 4, aligned_head_dim), dtype=jnp.float8_e4m3fn)
    new_kv_c_fp8 = quantize_to_kv_dtype(new_kv_c, jnp.float8_e4m3fn)
    new_k_pe_fp8 = quantize_to_kv_dtype(new_k_pe, jnp.float8_e4m3fn)
    out_fp8, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c_fp8,
        new_k_pe_fp8,
        cache_fp8,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale=sm_scale,
    )

    assert not jnp.isnan(out_fp8).any(), "FP8 MLA kernel produced NaNs"

    u = out_bf16.astype(jnp.float32).flatten()
    w = out_fp8.astype(jnp.float32).flatten()
    cos_sim = float(jnp.dot(u, w) / (jnp.linalg.norm(u) * jnp.linalg.norm(w) + 1e-8))
    assert cos_sim >= 0.995, f"MLA Cosine similarity {cos_sim} < 0.995"
