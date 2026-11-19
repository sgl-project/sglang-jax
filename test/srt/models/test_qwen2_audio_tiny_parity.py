"""CPU: Qwen2-Audio matches HF transformers on tiny random checkpoints.

1. audio tower + projector vs HF at two clip lengths (padded-frame masking);
2. end-to-end greedy decode with audio (1 clip, 1 clip, 2 clips, text-only)
   through the Engine. Requires network access for config/processor.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = {**os.environ, "JAX_PLATFORMS": "cpu"}


class TestQwen2AudioTinyParity(unittest.TestCase):
    def test_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            enc = subprocess.run(
                [sys.executable, os.path.join(HERE, "qwen2_audio_encoder_check.py"), os.path.join(d, "a")],
                check=True, env=ENV, capture_output=True, text=True,
            )
        self.assertIn("ENCODER_MATCH", enc.stdout)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "qwen2_audio")
            subprocess.run(
                [sys.executable, os.path.join(HERE, "tiny_hf_parity_qwen2_audio.py"), out],
                check=True, env=ENV,
            )
            report = json.load(open(os.path.join(out, "parity.json")))
        self.assertTrue(report["all_match"], report)


if __name__ == "__main__":
    unittest.main()
