"""Tests for the ``mecord`` video-reader compatibility shim.

These validate the exact surface the Kimi-K2.5 HF remote code depends on:
metadata attributes consumed by ``get_video_meta`` (which feeds a pydantic
model with non-Optional fields) and ``video[indices]`` frame extraction used
by ``resampling``.
"""

import base64
import importlib
import os
import sys
import tempfile

import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.mecord_compat import VideoReader, install_mecord_shim

FPS = 10.0
NUM_FRAMES = 20
HEIGHT, WIDTH = 64, 96


def _frame_color(i: int) -> int:
    """Distinct, widely-spaced luma per frame so lossy codecs stay identifiable."""
    return 10 + i * 12


@pytest.fixture(scope="module")
def video_path() -> str:
    imageio = pytest.importorskip("imageio")
    import imageio.v3 as iio

    frames = np.stack(
        [np.full((HEIGHT, WIDTH, 3), _frame_color(i), dtype=np.uint8) for i in range(NUM_FRAMES)]
    )
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    iio.imwrite(path, frames, fps=FPS, codec="libx264")
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def video_bytes(video_path: str) -> bytes:
    with open(video_path, "rb") as f:
        return f.read()


def test_metadata_from_path(video_path):
    vr = VideoReader(video_path)
    assert vr.num_frames == NUM_FRAMES
    assert vr.original_width == WIDTH
    assert vr.original_height == HEIGHT
    assert vr.avg_fps == pytest.approx(FPS, rel=0.05)


def test_metadata_fields_are_never_none(video_path):
    """get_video_meta feeds these into pydantic fields typed list[int] / dict.

    Explicit ``None`` would raise a ValidationError, so the shim must always
    hand back concrete values.
    """
    vr = VideoReader(video_path)
    assert isinstance(vr.key_indices, list)
    assert vr.key_indices and all(isinstance(i, int) for i in vr.key_indices)
    assert isinstance(vr.frame_time_info, dict)


def test_metadata_from_bytes(video_bytes):
    vr = VideoReader(video_bytes)
    assert vr.num_frames == NUM_FRAMES
    assert vr.original_width == WIDTH
    assert vr.original_height == HEIGHT


def test_metadata_from_data_uri(video_bytes):
    uri = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode()
    vr = VideoReader(uri)
    assert vr.num_frames == NUM_FRAMES


def test_metadata_from_bare_base64(video_bytes):
    vr = VideoReader(base64.b64encode(video_bytes).decode())
    assert vr.num_frames == NUM_FRAMES


def test_reader_accepts_mecord_kwargs(video_path):
    """The remote code passes auto_init / num_threads / hint kwargs."""
    vr = VideoReader(video_path, auto_init=True, num_threads=4, frame_time_info={}, key_indices=[0])
    assert vr.num_frames == NUM_FRAMES


@pytest.mark.parametrize("indices", [[0], [0, 5, 19], [3, 3, 3], [19, 0, 10]])
def test_frames_match_requested_indices(video_path, indices):
    vr = VideoReader(video_path)
    frames = vr[indices]
    assert len(frames) == len(indices)
    for frame, idx in zip(frames, indices):
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.uint8
        # Order must be preserved exactly, including duplicates.
        assert abs(float(frame.mean()) - _frame_color(idx)) < 6.0, (
            f"frame {idx} decoded as luma {frame.mean():.1f}, expected ~{_frame_color(idx)}"
        )


def test_scalar_index_returns_single_frame(video_path):
    vr = VideoReader(video_path)
    frame = vr[7]
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert abs(float(frame.mean()) - _frame_color(7)) < 6.0


def test_len_matches_num_frames(video_path):
    assert len(VideoReader(video_path)) == NUM_FRAMES


def test_get_batch_stacks_frames(video_path):
    batch = VideoReader(video_path).get_batch([0, 1, 2])
    assert batch.shape == (3, HEIGHT, WIDTH, 3)


def test_out_of_range_index_clamps_instead_of_crashing(video_path):
    """Containers over-report frame counts; the sampler derives indices from
    that count, so an over-run must degrade rather than kill the request."""
    vr = VideoReader(video_path)
    frames = vr[[NUM_FRAMES + 5]]
    assert len(frames) == 1
    assert frames[0].shape == (HEIGHT, WIDTH, 3)
    # Must fall back to the genuine final frame, not an arbitrary one.
    assert abs(float(frames[0].mean()) - _frame_color(NUM_FRAMES - 1)) < 6.0


def test_partially_out_of_range_indices_keep_valid_frames(video_path):
    vr = VideoReader(video_path)
    frames = vr[[2, NUM_FRAMES + 3]]
    assert abs(float(frames[0].mean()) - _frame_color(2)) < 6.0
    assert abs(float(frames[1].mean()) - _frame_color(NUM_FRAMES - 1)) < 6.0


# The shim decodes via decord only. The property worth protecting is no longer
# "two internal backends agree" but "decord returns the frames we actually
# asked for", so check it against an independent decoder.


def test_frames_match_independent_decoder(video_path):
    """decord must return the same frames a separate decoder sees.

    Guards against silent off-by-one/seek errors: Kimi asks for exact indices
    from ``np.linspace`` and a wrong frame is a silent correctness bug, not a
    crash.
    """
    iio = pytest.importorskip("imageio.v3")

    indices = [0, 4, 11, 19]
    shim_frames = VideoReader(video_path)[indices]

    reference = {}
    for position, frame in enumerate(iio.imiter(video_path)):
        if position in set(indices):
            arr = np.asarray(frame)
            reference[position] = arr[:, :, :3] if arr.shape[2] == 4 else arr
        if position >= max(indices):
            break

    for frame, idx in zip(shim_frames, indices):
        assert frame.shape == reference[idx].shape
        assert abs(float(frame.mean()) - float(reference[idx].mean())) < 3.0


def test_negative_index_rejected(video_path):
    with pytest.raises(IndexError):
        VideoReader(video_path)[[-1]]


def test_unsupported_source_type_rejected():
    with pytest.raises(ValueError):
        VideoReader(12345)



def test_install_shim_registers_mecord_module(video_path):
    sys.modules.pop("mecord", None)
    try:
        assert install_mecord_shim() is True
        mecord = importlib.import_module("mecord")
        assert mecord.VideoReader is VideoReader
        # Second call is a no-op since the module is already present.
        assert install_mecord_shim() is False
    finally:
        sys.modules.pop("mecord", None)


def test_kimi_sampling_pattern_end_to_end(video_path):
    """Mirrors KimiK25VisionProcessor.split_video_chunks' sampling math."""
    vr = VideoReader(video_path)
    sample_fps = min(2.0, vr.avg_fps)
    sampled_nframes = max(round(vr.num_frames * sample_fps / vr.avg_fps), 1)
    frame_inds = np.linspace(0, vr.num_frames - 1, sampled_nframes).round().astype(int).tolist()

    frames = vr[frame_inds]
    assert len(frames) == sampled_nframes == 4
    for frame, idx in zip(frames, frame_inds):
        assert abs(float(frame.mean()) - _frame_color(idx)) < 6.0
