"""CPU: InternVL 3 matches HF transformers on a tiny random checkpoint.

1. vision tower + projector vs HF ``get_image_features`` (multi-crop image);
2. end-to-end greedy decode with images (multi-crop, single-crop, two images,
   text-only) through the Engine. Requires network access for config/processor.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = {**os.environ, "JAX_PLATFORMS": "cpu"}


class TestInternVLTinyParity(unittest.TestCase):
    def test_end_to_end_and_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "internvl")
            subprocess.run(
                [sys.executable, os.path.join(HERE, "tiny_hf_parity_internvl.py"), out],
                check=True,
                env=ENV,
            )
            report = json.load(open(os.path.join(out, "parity.json")))
            self.assertTrue(report["all_match"], report)
            enc = subprocess.run(
                [sys.executable, os.path.join(HERE, "internvl_encoder_check.py"), out],
                check=True,
                env=ENV,
                capture_output=True,
                text=True,
            )
            self.assertIn("ENCODER_MATCH", enc.stdout)


if __name__ == "__main__":
    unittest.main()
