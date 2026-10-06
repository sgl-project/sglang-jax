import pytest

from sgl_jax.srt.server_args import ServerArgs


def _eagle_args(**overrides):
    kwargs = dict(
        model_path="target",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        disable_overlap_schedule=True,
        grammar_backend="none",
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"])
def test_tree_drafting_is_rejected_by_name(algorithm):
    args = _eagle_args(
        speculative_algorithm=algorithm,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=8,
    )
    with pytest.raises(ValueError, match="--speculative-eagle-topk > 1"):
        args.check_server_args()


def test_tree_drafting_is_rejected_before_the_overlap_gate():
    # The overlap gate's advice is to pass --disable-overlap-schedule, which
    # does not make a topk > 1 config runnable.
    args = _eagle_args(speculative_eagle_topk=4, disable_overlap_schedule=False)
    with pytest.raises(ValueError, match="--speculative-eagle-topk > 1") as excinfo:
        args.check_server_args()
    assert "--disable-overlap-schedule" not in str(excinfo.value)


def test_default_tree_width_is_rejected_by_name():
    args = ServerArgs(
        model_path="target",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
        disable_overlap_schedule=True,
        grammar_backend="none",
    )
    assert args.speculative_eagle_topk > 1
    with pytest.raises(ValueError, match="--speculative-eagle-topk > 1"):
        args.check_server_args()


def test_chain_drafting_passes():
    _eagle_args().check_server_args()
