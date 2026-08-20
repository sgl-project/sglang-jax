import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("validate_accuracy_metrics.py")
SPEC = spec_from_file_location("validate_accuracy_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
validate_metrics = MODULE.validate_metrics


def _write_metrics(root, *, score=0.95, num_examples=200, partial=False):
    run_dir = root / "sgl_eval_gsm8k_test"
    run_dir.mkdir()
    payload = {
        "name": "gsm8k",
        "num_examples": num_examples,
        "aggregate": {"score": score},
    }
    if partial:
        payload["partial"] = True
    (run_dir / "metrics.json").write_text(json.dumps(payload))


def test_validate_metrics_passes_complete_run(tmp_path):
    _write_metrics(tmp_path)
    report = validate_metrics(tmp_path, min_score=0.90, expected_examples=200)
    assert report["passed"] is True


def test_validate_metrics_rejects_low_score(tmp_path):
    _write_metrics(tmp_path, score=0.89)
    report = validate_metrics(tmp_path, min_score=0.90, expected_examples=200)
    assert report["passed"] is False


def test_validate_metrics_rejects_partial_run(tmp_path):
    _write_metrics(tmp_path, partial=True)
    report = validate_metrics(tmp_path, min_score=0.90, expected_examples=200)
    assert report["passed"] is False
