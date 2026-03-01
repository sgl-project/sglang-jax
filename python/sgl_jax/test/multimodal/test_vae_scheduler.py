import queue
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.communication import QueueBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.manager.scheduler.vae_scheduler import VaeScheduler
from sgl_jax.srt.multimodal.models.wan.vaes.wanvae import AutoencoderKLWan


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
        req = Req(rid="test", latents=x, num_frames=17)
        self.scheduler.run_vae_batch([req])
        result = self.backend._out_queue.get()
        self.assertEqual(result.output.shape, (1, 17, 24, 32, 3))


class TestVaeSchedulerBatching(unittest.TestCase):
    def test_run_vae_batch_groups_same_shape_requests(self):
        scheduler = VaeScheduler.__new__(VaeScheduler)
        scheduler._comm_backend = QueueBackend(in_queue=queue.Queue(), out_queue=queue.Queue())
        scheduler.vae_worker = mock.Mock()
        scheduler.forward_ct = 0
        scheduler._profile_batch_predicate = mock.Mock()

        def fake_forward(latents):
            markers = latents[:, :1, :1, :1, :1]
            output = np.broadcast_to(markers, (latents.shape[0], 3, 2, 2, 1)).copy()
            return output, 0

        scheduler.vae_worker.forward_latents.side_effect = fake_forward

        reqs = []
        expected_rids = []
        for idx in range(10):
            latent_shape = (1, 2, 3, 4, 1) if idx != 5 else (1, 2, 2, 4, 1)
            req = Req(
                rid=f"req-{idx}",
                latents=np.full(latent_shape, idx, dtype=np.float32),
                num_frames=2,
            )
            reqs.append(req)
            expected_rids.append(req.rid)

        scheduler.run_vae_batch(reqs)

        self.assertEqual(scheduler.vae_worker.forward_latents.call_count, 2)
        self.assertIn(
            9,
            [
                call_args.args[0].shape[0]
                for call_args in scheduler.vae_worker.forward_latents.call_args_list
            ],
        )

        outputs = [scheduler._comm_backend._out_queue.get() for _ in reqs]
        self.assertEqual([req.rid for req in outputs], expected_rids)
        self.assertEqual(scheduler.forward_ct, len(reqs))
        for idx, req in enumerate(outputs):
            self.assertEqual(req.output.shape, (1, 2, 2, 2, 1))
            self.assertEqual(req.output[0, 0, 0, 0, 0], idx)


if __name__ == "__main__":
    unittest.main()
