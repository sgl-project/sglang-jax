"""CPU regression tests for the scheduler's persistent compilation cache."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_COMPILE = """
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.scheduler import Scheduler

scheduler = Scheduler.__new__(Scheduler)
scheduler._setup_jit_cache(SimpleNamespace(device_indexes=None))

@jax.jit
def cache_relocation_probe(x):
    return jnp.tanh(x @ x.T).sum()

x = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
result = cache_relocation_probe(x).block_until_ready()
np.testing.assert_allclose(result, np.tanh(x @ x.T).sum(), rtol=1e-5)
"""


class TestJitCache(unittest.TestCase):
    def _compile(self, cache_dir):
        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "USE_DEVICE_TYPE": "cpu",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
            "JAX_LOG_COMPILES": "true",
            "JAX_EXPLAIN_CACHE_MISSES": "true",
        }
        result = subprocess.run(
            [sys.executable, "-c", _COMPILE],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stderr

    def test_cache_hits_after_directory_is_copied(self):
        hit = "Persistent compilation cache hit for 'jit_cache_relocation_probe'"
        miss = "PERSISTENT COMPILATION CACHE MISS for 'jit_cache_relocation_probe'"
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original"
            copied = Path(tmp) / "copied"

            cold_log = self._compile(original)
            self.assertIn(miss, cold_log)
            entries = {p.name for p in original.glob("*-cache")}
            self.assertTrue(entries, "The cold run must populate the persistent cache")

            warm_log = self._compile(original)
            self.assertIn(hit, warm_log)
            self.assertNotIn(miss, warm_log)

            shutil.copytree(original, copied)
            copied_log = self._compile(copied)
            self.assertIn(hit, copied_log)
            self.assertNotIn(miss, copied_log)
            self.assertEqual({p.name for p in copied.glob("*-cache")}, entries)


if __name__ == "__main__":
    unittest.main()
