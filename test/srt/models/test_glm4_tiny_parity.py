"""CPU: GLM-4 dense (Glm4ForCausalLM) matches HF transformers on a tiny random checkpoint.

Exercises sandwich norms, fused gate_up_proj split, q/k/v bias and partial
interleaved RoPE. Requires network access to fetch the config/tokenizer.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class TestGlm4TinyParity(unittest.TestCase):
    def test_glm4_dense(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "glm4")
            subprocess.run(
                [sys.executable, os.path.join(HERE, "tiny_hf_parity.py"), "glm4", out],
                check=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )
            report = json.load(open(os.path.join(out, "parity.json")))
        self.assertTrue(report["all_match"], report)


if __name__ == "__main__":
    unittest.main()
