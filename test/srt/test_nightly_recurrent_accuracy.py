import sys
from pathlib import Path

NIGHTLY_DIR = Path(__file__).parent / "nightly"
sys.path[:0] = [str(NIGHTLY_DIR), str(NIGHTLY_DIR / "single_host")]

from profiles import load_profile  # noqa: E402
from suite_runner import SUITES  # noqa: E402


def test_recurrent_accuracy_cases_match_profiles():
    suite = SUITES["accuracy-text-models-v6e-4"]
    runs = {
        case.name: (run.launch_profile, case)
        for run in suite.runs
        for case in run.cases
        if case.name.startswith("recurrent-")
    }
    assert set(runs) == {"recurrent-gsm8k", "recurrent-mmlu-thinking"}

    gsm_profile_name, gsm_case = runs["recurrent-gsm8k"]
    mmlu_profile_name, mmlu_case = runs["recurrent-mmlu-thinking"]
    assert gsm_profile_name == "recurrent-qwen35-gsm8k-v6e-4.yaml"
    assert mmlu_profile_name == "recurrent-qwen35-mmlu-thinking-v6e-4.yaml"
    assert gsm_case.score_threshold == 0.95
    assert gsm_case.generation_config["chat_template_kwargs"]["enable_thinking"] is False
    assert mmlu_case.score_threshold == 0.88
    assert mmlu_case.generation_config["seed"] == 11
    assert mmlu_case.generation_config["chat_template_kwargs"]["enable_thinking"] is True


def test_recurrent_accuracy_profile_shapes():
    profiles_dir = NIGHTLY_DIR / "launch_profiles"
    gsm = load_profile(profiles_dir / "recurrent-qwen35-gsm8k-v6e-4.yaml")
    mmlu = load_profile(profiles_dir / "recurrent-qwen35-mmlu-thinking-v6e-4.yaml")

    assert (gsm.tp_size, gsm.dp_size, gsm.ep_size) == (4, 2, None)
    assert (mmlu.tp_size, mmlu.dp_size, mmlu.ep_size) == (4, 2, 4)
    assert gsm.check_cache_miss is True
    assert mmlu.check_cache_miss is False
    assert gsm.server_args[gsm.server_args.index("--context-length") + 1] == "8192"
    assert mmlu.server_args[mmlu.server_args.index("--context-length") + 1] == "32768"
