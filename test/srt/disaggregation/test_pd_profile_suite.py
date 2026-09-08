"""Profiling export budgets must not leak into normal serving benchmarks."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fail", [False, True])
def test_profile_timeout_override_is_scoped_and_restored(monkeypatch, tmp_path, fail):
    reference = tmp_path / "correctness.json"
    reference.write_text(json.dumps({"status": "passed_implemented_correctness_checks"}))
    monkeypatch.setenv("PD_REQUIRE_SUMMARY", str(reference))
    original = ["--page-size", "128"]
    launches = []
    driver = SimpleNamespace(OUT=tmp_path, MODEL="model", COMMON=original[:], stop=lambda: None)

    def boot(name, prefill, decode, chunk):
        launches.append((name, prefill, decode, chunk, driver.COMMON[:]))
        return 30020

    def profile(*_):
        if fail:
            raise RuntimeError("export failed")

    driver.boot = boot
    suite = SimpleNamespace(driver=driver, REPORT={}, save=lambda: None, profile=profile)
    monkeypatch.setitem(sys.modules, "steady_suite", suite)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: object())),
    )
    path = Path(__file__).resolve().parents[3] / "scripts/disaggregation/falcon/profile_suite.py"
    spec = importlib.util.spec_from_file_location("profile_suite_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if fail:
        with pytest.raises(RuntimeError, match="export failed"):
            module.main()
        assert suite.REPORT.get("status") != "passed"
    else:
        module.main()
        assert suite.REPORT["status"] == "passed"
        assert [launch[1:4] for launch in launches] == [(False, False, True), (True, True, True)]
    assert driver.COMMON == original
    for launch in launches:
        args = launch[-1]
        for flag in [
            "--disaggregation-pull-timeout-seconds",
            "--disaggregation-ack-timeout-seconds",
        ]:
            assert args[args.index(flag) + 1] == "300"
