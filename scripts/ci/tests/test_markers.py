"""Verify pytest markers stay in sync with run_suite.py suite dictionaries."""

import ast
import os
import re
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RUN_SUITE = os.path.join(REPO_ROOT, "test", "srt", "run_suite.py")


def _extract_suite_files(suite_name):
    with open(RUN_SUITE) as f:
        text = f.read()
    in_suite = False
    files = []
    for line in text.split("\n"):
        if f'"{suite_name}":' in line:
            in_suite = True
            continue
        if in_suite:
            if line.strip().startswith("],") or line.strip() == "]":
                break
            m = re.search(r'TestFile\("([^"]+)"', line)
            if m:
                files.append(m.group(1))
    return files


def _file_has_marker(filepath, marker_name):
    full_path = os.path.join(REPO_ROOT, filepath)
    if not os.path.exists(full_path):
        return False
    with open(full_path) as f:
        source = f.read()
    return f"pytest.mark.{marker_name}" in source


class TestMarkerConsistency(unittest.TestCase):
    def test_cpu_suite_files_have_cpu_only_marker(self):
        cpu_files = _extract_suite_files("unit-test-cpu")
        self.assertTrue(len(cpu_files) > 0, "No files found in unit-test-cpu suite")
        missing = [f for f in cpu_files if not _file_has_marker(f, "cpu_only")]
        self.assertEqual(
            missing,
            [],
            f"Files in unit-test-cpu suite missing cpu_only marker: {missing}",
        )

    def test_cpu_only_marker_not_in_tpu_suites(self):
        with open(RUN_SUITE) as f:
            text = f.read()
        tpu_suites = re.findall(r'"(unit-test-tpu-[^"]+)":', text)
        tpu_files = []
        for suite_name in tpu_suites:
            tpu_files.extend(_extract_suite_files(suite_name))
        wrong = [f for f in tpu_files if _file_has_marker(f, "cpu_only")]
        self.assertEqual(
            wrong,
            [],
            f"Files in TPU suites should not have cpu_only marker: {wrong}",
        )


if __name__ == "__main__":
    unittest.main()
