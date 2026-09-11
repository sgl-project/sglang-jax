"""CPU checks for CircularEval integrity, response failures and VLM routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from eval.mmbench_utils import can_infer
from eval.simple_eval_mmbench import messages_for, run_mmbench, select_indices
from PIL import Image
from vlm_utils import image_content


@pytest.fixture
def mmbench_data():
    image = image_content(Image.new("RGB", (8, 8), "red"))["image_url"]["url"].split(",", 1)[1]
    common = {
        "category": "test",
        "split": "dev",
        "question": "Which color?",
        "hint": "Look.",
        "image": image,
        "C": None,
        "D": None,
    }
    return pd.DataFrame(
        [
            dict(common, index=0, A="red", B="blue", answer="A"),
            dict(common, index=1000000, A="blue", B="red", answer="B"),
        ]
    )


def test_mmbench_preserves_image_hint_and_options(mmbench_data):
    messages = messages_for(mmbench_data.iloc[0])
    assert [m["role"] for m in messages] == ["user"]
    content = messages[0]["content"]
    assert content[0] == image_content(Image.new("RGB", (8, 8), "red"))
    assert "Hint: Look." in content[1]["text"]
    assert "A. red" in content[1]["text"] and "B. blue" in content[1]["text"]
    assert "C." not in content[1]["text"]


def test_mmbench_selection_preserves_all_source_rotations(mmbench_data):
    data = mmbench_data
    assert len(select_indices(data, None)) == 2
    assert select_indices(data, None)["index"].tolist() == [0, 1000000]
    with pytest.raises(ValueError, match="Missing"):
        select_indices(data.iloc[0:0], None)
    assert select_indices(data, 1)["index"].tolist() == [0, 1000000]
    for limit in (0, -1, 200):
        with pytest.raises(ValueError, match="limit"):
            select_indices(data, limit)


def test_mmbench_sampling_is_repeatable_and_preserves_groups(mmbench_data):
    data = pd.concat(
        [mmbench_data.assign(index=mmbench_data["index"] + i) for i in range(4)],
        ignore_index=True,
    )
    expected = [1, 1000001, 3, 1000003]
    assert select_indices(data, 2)["index"].tolist() == expected
    assert select_indices(data, 2)["index"].tolist() == expected


def test_mmbench_four_options_can_have_three_source_rotations(mmbench_data):
    data = pd.concat([mmbench_data, mmbench_data.iloc[:1]], ignore_index=True)
    data["index"] = [0, 1000000, 2000000]
    data["C"] = "green"
    data["D"] = "All above are not right"
    assert len(select_indices(data, None)) == 3


def test_mmbench_rejects_corrupt_cached_dataset(monkeypatch, tmp_path):
    from eval.simple_eval_mmbench import DATASET, load_data

    monkeypatch.setenv("LMUData", str(tmp_path))
    (tmp_path / f"{DATASET}.tsv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_data()


@pytest.mark.parametrize(
    "answers,score",
    [(["A. red", "B. red"], 1.0), (["A", "A"], 0.0), (["A or B", "B"], 0.0)],
)
def test_circular_scoring_and_repeated_runs(monkeypatch, tmp_path, mmbench_data, answers, score):
    monkeypatch.setattr("eval.simple_eval_mmbench.load_data", lambda: mmbench_data)
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    client = MagicMock()
    client.__enter__.return_value = client
    create = client.chat.completions.create

    def response(text, reason="stop"):
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=text))]
        )

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    args = SimpleNamespace(
        base_url="http://localhost:30000",
        num_examples=None,
        num_threads=1,
        temperature=0,
        max_tokens=32,
        seed=42,
    )
    create.side_effect = [response(a) for a in answers]
    result = run_mmbench(args)
    assert result["score"] == score
    assert result["num_examples"] == 1 and result["num_requests"] == 2
    assert [s["id"] for s in result["samples"]] == [0, 1000000]
    # Repeating in the same artifact directory must grade the new responses.
    create.side_effect = [response("A"), response("A")]
    assert run_mmbench(args)["score"] == 0.0
    for score in (None, float("nan"), float("inf")):
        create.side_effect = [response("A"), response("B")]
        with monkeypatch.context() as patch:
            patch.setattr(pd.Series, "mean", lambda self, value=score: value)
            with pytest.raises(ValueError, match="no finite score"):
                run_mmbench(args)
    create.side_effect = [response("A", "length")]
    with pytest.raises(RuntimeError, match="Incomplete VLM response"):
        run_mmbench(args)
    create.side_effect = [response("  ")]
    with pytest.raises(RuntimeError, match="Incomplete VLM response"):
        run_mmbench(args)
    args.base_url = None
    args.host, args.port = "localhost", 30000
    create.side_effect = [response("A"), response("B")]
    assert run_mmbench(args)["score"] == 1.0
    create.side_effect = RuntimeError("server failed")
    with pytest.raises(RuntimeError, match="server failed"):
        run_mmbench(args)


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("B. dog", "B"),
        ("The answer is **B**.", "B"),
        ("dog", "B"),
        ("A or B", False),
        ("b", False),
        ("Cannot determine the answer", "Z"),
    ],
)
def test_mmbench_answer_extraction(answer, expected, monkeypatch):
    monkeypatch.delenv("VERBOSE", raising=False)
    assert can_infer(answer, {"A": "cat", "B": "dog"}) == expected


def test_mmbench_uses_standard_eval_entrypoint(monkeypatch):
    from run_eval import run_eval

    expected = {"score": 0.75}
    args = SimpleNamespace(eval_name="mmbench_v11", base_url=None, host="localhost", port=30000)
    monkeypatch.setattr("run_eval.set_ulimit", lambda: None)
    monkeypatch.setattr("eval.simple_eval_mmbench.run_mmbench", lambda received: expected)
    assert run_eval(args) == expected


@pytest.mark.parametrize("limit,lower", [(200, 0.70), (None, 0.74)])
@pytest.mark.parametrize(
    "point,expected",
    [("below", False), ("lower", True), ("upper", True), ("above", False)],
)
def test_mmbench_score_interval(limit, lower, point, expected):
    from nightly.results import build_accuracy_result
    from nightly.single_host.suite_runner import _gate_accuracy, _mmbench_case

    case = _mmbench_case("test", limit, lower)
    score = {"below": lower - 0.001, "lower": lower, "upper": 0.80, "above": 0.801}[point]
    result = build_accuracy_result(case, "test", "v7x-8", {"score": score}, 0, 1)
    assert result["score_upper_threshold"] == 0.80
    assert result["passed"] is expected
    assert (_gate_accuracy(case, result) is None) is expected
