"""Static contract check: every ``attn_backend.<method>(`` the speculative path calls
must be defined by the MLA/DSA backends (or their base), so an API rename in the
speculative code cannot surface only at server start on TPU.
"""

import pathlib
import re
import unittest

SRT = pathlib.Path(__file__).resolve().parents[3] / "python" / "sgl_jax" / "srt"
BACKENDS = [
    SRT / "layers" / "attention" / "mla_backend.py",
    SRT / "layers" / "attention" / "dsa_sparse_backend.py",
    SRT / "layers" / "attention" / "base_attn_backend.py",
]
# Only reachable from the topk>1 EAGLE tree path, which MLA does not support.
KNOWN_NON_MLA = {"get_eagle_multi_step_metadata"}


def _called_methods():
    names = set()
    for f in list((SRT / "speculative").glob("*.py")) + [
        SRT / "model_executor" / "model_runner.py"
    ]:
        names.update(re.findall(r"attn_backend\.([A-Za-z_][A-Za-z0-9_]*)\(", f.read_text()))
    return names


def _defined_methods():
    defs = set()
    for f in BACKENDS:
        if f.exists():
            defs.update(re.findall(r"^\s+def ([A-Za-z_][A-Za-z0-9_]*)\(", f.read_text(), re.M))
    return defs


class TestSpecAttnBackendContract(unittest.TestCase):
    def test_mla_backends_define_spec_methods(self):
        missing = sorted(_called_methods() - _defined_methods() - KNOWN_NON_MLA)
        self.assertEqual(
            missing, [], f"speculative path calls undefined MLA/DSA backend methods: {missing}"
        )

    def test_base_metadata_present(self):
        self.assertIn("get_eagle_base_metadata", _defined_methods())


if __name__ == "__main__":
    unittest.main()
