"""Temporal (video) behavior of the Kimi-K2.5 vision tower.

These tests cover the host-side merge planning and the position embeddings, which
is everything that decides whether a video item (``t > 1``) is pooled correctly.
They deliberately avoid building the encoder so they stay CPU-only and fast.
"""

import numpy as np
import pytest

from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import (
    Learnable2DInterPosEmbDivided_fixed,
    Rope2DPosEmbRepeated,
    build_temporal_merge_plan,
)

MERGE_KERNEL = (2, 2)


def _reference_merge(grid_thws, patches, merge_kernel_size=MERGE_KERNEL):
    """Merge each item independently: the behavior the batched plan must match."""
    merge_h, merge_w = merge_kernel_size
    outputs = []
    offset = 0
    for t, h, w in grid_thws:
        new_h, new_w = h // merge_h, w // merge_w
        item = patches[offset : offset + t * h * w]
        item = item.reshape(t, new_h, merge_h, new_w, merge_w, -1)
        # (new_h, new_w, merge_h, merge_w, t, dim) -> mean over frames
        item = item.transpose(1, 3, 2, 4, 0, 5)
        item = item.reshape(new_h * new_w, merge_h * merge_w, t, -1)
        outputs.append(item.mean(axis=2))
        offset += t * h * w
    return np.concatenate(outputs, axis=0)


def _apply_plan(patches, merge_indices, merge_weights):
    """Mirror `VisionTower.compute_hidden_states`'s gather-and-average."""
    gathered = patches[merge_indices]
    return (gathered * merge_weights[:, None, :, None]).sum(axis=2)


def _patches(grid_thws, dim=3, seed=0):
    total = sum(t * h * w for t, h, w in grid_thws)
    return np.random.default_rng(seed).standard_normal((total, dim)).astype(np.float32)


@pytest.mark.parametrize(
    "grid_thws",
    [
        [(1, 2, 2)],  # single image
        [(1, 4, 6), (1, 2, 2)],  # images only
        [(4, 2, 2)],  # single video
        [(2, 2, 4), (4, 2, 2)],  # videos whose frame counts divide each other
        [(3, 2, 2), (4, 2, 2)],  # frame counts that do NOT divide each other
        [(1, 4, 4), (3, 2, 2), (2, 2, 4)],  # image + videos of differing lengths
    ],
)
def test_merge_plan_matches_per_item_temporal_mean(grid_thws):
    patches = _patches(grid_thws)
    merge_indices, merge_weights = build_temporal_merge_plan(grid_thws, MERGE_KERNEL)

    np.testing.assert_allclose(
        _apply_plan(patches, merge_indices, merge_weights),
        _reference_merge(grid_thws, patches),
        rtol=1e-6,
        atol=1e-6,
    )


def test_merge_plan_shapes_and_weights():
    grid_thws = [(1, 4, 4), (3, 2, 2)]
    merge_indices, merge_weights = build_temporal_merge_plan(grid_thws, MERGE_KERNEL)

    # Output tokens per item are frame-independent: h * w / (merge_h * merge_w).
    expected_tokens = 4 * 4 // 4 + 2 * 2 // 4
    max_t = 3
    assert merge_indices.shape == (expected_tokens, 4, max_t)
    assert merge_weights.shape == (expected_tokens, max_t)

    # The image only weights its single frame; the video splits across its three.
    np.testing.assert_allclose(merge_weights[:4], [[1.0, 0.0, 0.0]] * 4)
    np.testing.assert_allclose(merge_weights[4:], [[1 / 3, 1 / 3, 1 / 3]])
    # Every weight row sums to one, so merging preserves scale.
    np.testing.assert_allclose(merge_weights.sum(axis=1), 1.0)
    # Padded slots reuse in-bounds indices.
    total_patches = sum(t * h * w for t, h, w in grid_thws)
    assert merge_indices.min() >= 0
    assert merge_indices.max() < total_patches


def test_merge_plan_indices_cover_every_patch_once():
    grid_thws = [(2, 2, 4), (1, 2, 2)]
    merge_indices, merge_weights = build_temporal_merge_plan(grid_thws, MERGE_KERNEL)

    # A weight of 0 marks a padded temporal slot, so the non-zero positions must
    # name each real patch exactly once.
    real_slots = np.broadcast_to(merge_weights[:, None, :] > 0, merge_indices.shape)
    weighted = merge_indices[real_slots]
    total_patches = sum(t * h * w for t, h, w in grid_thws)
    np.testing.assert_array_equal(np.sort(weighted.reshape(-1)), np.arange(total_patches))


@pytest.mark.parametrize(
    ("grid_thws", "match"),
    [
        ([], "at least one"),
        ([(1, 3, 2)], "not divisible by merge kernel"),
    ],
)
def test_merge_plan_rejects_invalid_grids(grid_thws, match):
    with pytest.raises(ValueError, match=match):
        build_temporal_merge_plan(grid_thws, MERGE_KERNEL)


def test_pos_emb_allows_more_frames_than_the_table_depth():
    dim = 8
    pos_emb = Learnable2DInterPosEmbDivided_fixed(height=4, width=4, num_frames=4, dim=dim)

    # 16 frames is well past ``num_frames``; ``divided_fixed`` repeats the same 2D
    # table per frame, so a long video must still work.
    grid_thws = [(16, 4, 4)]
    embeddings = np.asarray(pos_emb(grid_thws))
    assert embeddings.shape == (16 * 4 * 4, dim)

    # Every frame gets an identical spatial embedding.
    per_frame = embeddings.reshape(16, 4 * 4, dim)
    np.testing.assert_allclose(per_frame, np.broadcast_to(per_frame[0], per_frame.shape))


def test_rope_repeats_per_frame():
    dim = 8
    rope = Rope2DPosEmbRepeated(dim=dim, max_height=8, max_width=8)

    freqs = np.asarray(rope._get_freqs_cis([(3, 2, 2)]))
    assert freqs.shape == (2, 3 * 2 * 2, dim // 2)

    # Frames share the same 2D frequencies (temporal order is carried by the
    # sequence layout, not by the RoPE table).
    per_frame = freqs.reshape(2, 3, 2 * 2, dim // 2)
    np.testing.assert_allclose(per_frame[:, 1], per_frame[:, 0])
    np.testing.assert_allclose(per_frame[:, 2], per_frame[:, 0])
