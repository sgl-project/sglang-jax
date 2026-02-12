from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    MULTI_ITEM_MASK_MODE_DENSE,
    MULTI_ITEM_MASK_MODE_SEGMENT,
)
from sgl_jax.srt.managers.schedule_batch import global_server_args_dict
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _mask_from_segment_layout(prefix_end: int, row_seg_starts: np.ndarray, seq_len: int) -> np.ndarray:
    mask = np.zeros((seq_len, seq_len), dtype=np.int32)
    for row in range(seq_len):
        if row < prefix_end:
            mask[row, : row + 1] = 1
        else:
            mask[row, :prefix_end] = 1
            seg_start = int(row_seg_starts[row])
            mask[row, seg_start : row + 1] = 1
    return mask


def _build_flash_attn() -> FlashAttention:
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    return FlashAttention(
        num_attn_heads=2,
        num_kv_heads=1,
        head_dim=8,
        page_size=1,
        mesh=mesh,
    )


def _build_batch(tokens: list[int], delimiter: int) -> SimpleNamespace:
    seq_len = len(tokens)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        cache_loc=np.arange(seq_len, dtype=np.int32),
        extend_seq_lens=np.array([seq_len], dtype=np.int32),
        seq_lens=np.array([seq_len], dtype=np.int32),
        input_ids=np.array(tokens, dtype=np.int32),
        multi_item_scoring_delimiter=delimiter,
        multi_item_scoring_flags=np.array([True], dtype=np.bool_),
        real_bs=1,
    )


def test_segment_layout_reconstructs_dense_mask():
    delimiter = 99
    # query=[10, 11], then <d>item1(2 tokens)<d>item2(1 token)<d>
    tokens = np.array([10, 11, 99, 21, 22, 99, 31, 99], dtype=np.int32)

    dense_mask = FlashAttention._build_multi_item_attention_mask(tokens, delimiter)
    prefix_end, row_seg_starts = FlashAttention._build_multi_item_segment_layout(tokens, delimiter)
    rebuilt = _mask_from_segment_layout(prefix_end, row_seg_starts, tokens.shape[0])

    np.testing.assert_array_equal(rebuilt, dense_mask)


def test_get_forward_metadata_selects_segment_mode_in_auto():
    attn = _build_flash_attn()
    delimiter = 99
    tokens = [10, 11, 99, 21, 22, 99, 31, 99]
    batch = _build_batch(tokens, delimiter)

    old_impl = global_server_args_dict.get("multi_item_mask_impl")
    old_threshold = global_server_args_dict.get("multi_item_segment_fallback_threshold")
    try:
        global_server_args_dict["multi_item_mask_impl"] = "auto"
        global_server_args_dict["multi_item_segment_fallback_threshold"] = 4096
        metadata = attn.get_forward_metadata(batch)
    finally:
        global_server_args_dict["multi_item_mask_impl"] = old_impl
        global_server_args_dict["multi_item_segment_fallback_threshold"] = old_threshold

    assert metadata.multi_item_mask_mode == MULTI_ITEM_MASK_MODE_SEGMENT
    assert metadata.custom_mask is None

    prefix_end = np.asarray(jax.device_get(metadata.multi_item_prefix_end))[0]
    row_seg_starts = np.asarray(jax.device_get(metadata.multi_item_row_seg_starts))
    dense_mask = FlashAttention._build_multi_item_attention_mask(np.array(tokens, dtype=np.int32), delimiter)
    rebuilt = _mask_from_segment_layout(int(prefix_end), row_seg_starts, len(tokens))
    np.testing.assert_array_equal(rebuilt, dense_mask)


def test_get_forward_metadata_selects_dense_mode_when_forced():
    attn = _build_flash_attn()
    delimiter = 99
    tokens = [10, 11, 99, 21, 22, 99, 31, 99]
    batch = _build_batch(tokens, delimiter)

    old_impl = global_server_args_dict.get("multi_item_mask_impl")
    old_threshold = global_server_args_dict.get("multi_item_segment_fallback_threshold")
    try:
        global_server_args_dict["multi_item_mask_impl"] = "dense"
        global_server_args_dict["multi_item_segment_fallback_threshold"] = 1
        metadata = attn.get_forward_metadata(batch)
    finally:
        global_server_args_dict["multi_item_mask_impl"] = old_impl
        global_server_args_dict["multi_item_segment_fallback_threshold"] = old_threshold

    assert metadata.multi_item_mask_mode == MULTI_ITEM_MASK_MODE_DENSE
    assert metadata.custom_mask is not None
