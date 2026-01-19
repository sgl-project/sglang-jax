import queue
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.communication import QueueBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.manager.scheduler.vae_scheduler import VaeScheduler
from sgl_jax.srt.multimodal.models.wan2_1.vaes.wanvae import AutoencoderKLWan


class TestVaeScheduler(unittest.TestCase):
    """Test VaeScheduler full load and forward flow."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = jax.sharding.Mesh(jax.devices(), axis_names=("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            download_dir="/dev/shm",
        )
        cls.backend = QueueBackend(in_queue=queue.Queue(), out_queue=queue.Queue())
        with jax.default_device(jax.devices()[0]):
            cls.scheduler = VaeScheduler(
                server_args=cls.server_args,
                mesh=cls.mesh,
                communication_backend=cls.backend,
                model_class=AutoencoderKLWan,
            )

    def test_run_vae_step(self):
        """Test full scheduler forward pass."""
        x = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(1, 5, 3, 4, 16)
        req = Req(rid="test", latents=x)
        self.scheduler.run_vae_batch([req])
        result = self.backend._out_queue.get()
        self.assertEqual(result.output.shape, (1, 17, 24, 32, 3))


if __name__ == "__main__":
    unittest.main()
