import pytest

from sgl_jax.srt.server_args import ServerArgs


def test_eagle_rejects_mixed_chunk_batches():
    server_args = ServerArgs(
        model_path="dummy-model",
        speculative_algorithm="EAGLE3",
        disable_overlap_schedule=True,
        enable_mixed_chunk=True,
    )

    with pytest.raises(ValueError, match="does not support mixed chunk batches"):
        server_args.check_server_args()


def test_eagle_allows_non_mixed_chunk_batches():
    server_args = ServerArgs(
        model_path="dummy-model",
        speculative_algorithm="EAGLE3",
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
    )

    server_args.check_server_args()
