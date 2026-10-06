from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.scheduler import validate_eagle_tree_request
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


def _tree_args(**overrides):
    kwargs = dict(speculative_eagle_topk=4, speculative_num_draft_tokens=8, page_size=1)
    kwargs.update(overrides)
    return _eagle_args(**kwargs)


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3"])
def test_tree_drafting_passes(algorithm):
    _tree_args(speculative_algorithm=algorithm).check_server_args()


def test_chain_drafting_passes():
    _eagle_args().check_server_args()


@pytest.mark.parametrize("algorithm", ["NEXTN", "STANDALONE"])
def test_tree_drafting_is_eagle_only(algorithm):
    with pytest.raises(ValueError, match="EAGLE and EAGLE3 only"):
        _tree_args(speculative_algorithm=algorithm).check_server_args()


def test_tree_drafting_requires_non_overlap_before_the_overlap_gate():
    with pytest.raises(ValueError, match="> 1 requires --disable-overlap-schedule"):
        _tree_args(disable_overlap_schedule=False).check_server_args()


@pytest.mark.parametrize(
    "overrides, flag",
    [
        (dict(page_size=64), "--page-size 1"),
        (dict(dp_size=2, tp_size=2), "--dp-size 1"),
        (dict(attention_backend="native"), "--attention-backend fa"),
    ],
)
def test_tree_drafting_rejects_unsupported_layouts(overrides, flag):
    with pytest.raises(ValueError, match=flag):
        _tree_args(**overrides).check_server_args()


def test_tree_drafting_bounds_draft_tokens_by_drafted_candidates():
    # steps=3, topk=2 drafts 2 + 2*4 = 10 candidates, so at most 11 verify tokens.
    _tree_args(speculative_eagle_topk=2, speculative_num_draft_tokens=11).check_server_args()
    with pytest.raises(ValueError, match="at most 11"):
        _tree_args(speculative_eagle_topk=2, speculative_num_draft_tokens=12).check_server_args()


def test_default_tree_width_runs_without_overlap():
    args = ServerArgs(
        model_path="target",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
        disable_overlap_schedule=True,
        grammar_backend="none",
    )
    assert args.speculative_eagle_topk > 1
    args.check_server_args()


@pytest.mark.parametrize("top_k, rejected", [(1, False), (20, True)])
def test_tree_drafting_rejects_sampled_requests(top_k, rejected):
    req = SimpleNamespace(sampling_params=SimpleNamespace(top_k=top_k))
    err = validate_eagle_tree_request(req)
    assert (err is not None) == rejected
    if rejected:
        assert "greedy sampling only" in err
