"""Contract tests for the frozen TPU-Inference GDN v3 vendor snapshot."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parents[2] / "srt" / "kernels" / "gdn" / "tpu_inference_v3"
PROVENANCE = VENDOR_DIR / "PROVENANCE.md"
MODULES = {
    "__init__.py",
    "compute_conv1d.py",
    "compute_gdn.py",
    "config.py",
    "memory_ref.py",
    "metadata.py",
    "vmem_ldst.py",
    "wrapper.py",
}
PACKAGE_NAME = "sgl_jax.srt.kernels.gdn.tpu_inference_v3"


def test_vendor_snapshot_has_only_the_approved_modules_and_headers():
    assert {path.name for path in VENDOR_DIR.glob("*.py")} == MODULES
    for module in MODULES:
        source = (VENDOR_DIR / module).read_text()
        assert "Google" in source
        assert "Licensed under the Apache License, Version 2.0" in source


def test_vendor_snapshot_has_closed_imports_and_frozen_provenance():
    for module in MODULES:
        source = (VENDOR_DIR / module).read_text()
        assert "tpu_inference." not in source
        assert "/tmp/" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names | {"jax"}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                assert node.module is not None
                assert (
                    node.module == PACKAGE_NAME
                    or node.module.startswith(f"{PACKAGE_NAME}.")
                    or node.module.split(".")[0] in sys.stdlib_module_names | {"jax"}
                )

    provenance = PROVENANCE.read_text()
    assert "repository: https://github.com/vllm-project/tpu-inference" in provenance
    assert "commit: a9072c881843622226efc101de1a62c731ab572f" in provenance
    assert "source: tpu_inference/kernels/gdn/v3" in provenance
    assert "license: Apache-2.0" in provenance


def test_vendor_snapshot_exports_the_upstream_entrypoint():
    package = importlib.import_module(PACKAGE_NAME)
    assert callable(package.fused_conv1d_gdn)
