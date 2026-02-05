"""Test VIT batch forward consistency.

Ensures that forward_batch produces the same results as calling forward
individually for each request.
"""

import unittest

import jax
import numpy as np

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vit.vit_model_runner import VitModelRunner
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vit import Qwen2_5_VL_VisionModel


class TestVitBatchConsistency(unittest.TestCase):
    """Test that forward_batch produces same results as individual forward calls."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], axis_names=("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="/models/Qwen/Qwen2.5-VL-7B-Instruct",
            download_dir="/dev/shm",
        )
        with jax.default_device(jax.devices("cpu")[0]):
            cls.model_runner = VitModelRunner(
                server_args=cls.server_args,
                mesh=cls.mesh,
                model_class=Qwen2_5_VL_VisionModel,
            )

    def _create_req_with_image(self, rid: str, grid_h: int, grid_w: int, seed: int = 0) -> Req:
        """Create a request with random image pixel values."""
        rng = np.random.default_rng(seed)
        t, h, w = 1, grid_h, grid_w
        # pixel_values shape: (num_patches, channels * patch_size * patch_size)
        # For Qwen2.5-VL: patch_size=14, so each patch has 14*14*3 = 588 values
        num_patches = t * h * w
        patch_dim = 14 * 14 * 3  # patch_size=14, channels=3
        pixel_values = rng.standard_normal((num_patches, patch_dim)).astype(np.float32)

        req = Req(
            rid=rid,
            pixel_values=pixel_values,
            image_grid_thw=((t, h, w),),
            video_grid_thw=None,
            input_ids=[151643, 151644, 151645],  # dummy input_ids
            vlm_inputs={
                "im_token_id": 151655,
                "video_token_id": 151656,
            },
        )
        return req

    def _create_req_with_video(
        self, rid: str, grid_t: int, grid_h: int, grid_w: int, seed: int = 0
    ) -> Req:
        """Create a request with random video pixel values."""
        rng = np.random.default_rng(seed)
        num_patches = grid_t * grid_h * grid_w
        patch_dim = 14 * 14 * 3
        pixel_values = rng.standard_normal((num_patches, patch_dim)).astype(np.float32)

        req = Req(
            rid=rid,
            pixel_values=pixel_values,
            image_grid_thw=None,
            video_grid_thw=((grid_t, grid_h, grid_w),),
            input_ids=[151643, 151644, 151645],
            vlm_inputs={
                "im_token_id": 151655,
                "video_token_id": 151656,
            },
        )
        return req

    def _create_req_with_image_and_video(
        self, rid: str, img_h: int, img_w: int, vid_t: int, vid_h: int, vid_w: int, seed: int = 0
    ) -> Req:
        """Create a request with both image and video."""
        rng = np.random.default_rng(seed)
        patch_dim = 14 * 14 * 3

        img_patches = 1 * img_h * img_w
        vid_patches = vid_t * vid_h * vid_w

        img_pixels = rng.standard_normal((img_patches, patch_dim)).astype(np.float32)
        vid_pixels = rng.standard_normal((vid_patches, patch_dim)).astype(np.float32)

        # pixel_values: image first, then video
        pixel_values = np.concatenate([img_pixels, vid_pixels], axis=0)

        req = Req(
            rid=rid,
            pixel_values=pixel_values,
            image_grid_thw=((1, img_h, img_w),),
            video_grid_thw=((vid_t, vid_h, vid_w),),
            input_ids=[151643, 151644, 151645],
            vlm_inputs={
                "im_token_id": 151655,
                "video_token_id": 151656,
            },
        )
        return req

    def _deep_copy_req(self, req: Req) -> Req:
        """Deep copy a request for independent processing."""
        new_req = Req(
            rid=req.rid,
            pixel_values=np.array(req.pixel_values) if req.pixel_values is not None else None,
            image_grid_thw=req.image_grid_thw,
            video_grid_thw=req.video_grid_thw,
            input_ids=list(req.input_ids) if req.input_ids else None,
            vlm_inputs=dict(req.vlm_inputs) if req.vlm_inputs else None,
        )
        return new_req

    def _get_embedding(self, req: Req) -> jax.Array | None:
        """Extract multimodal_embedding from request."""
        if req.vlm_inputs and "multimodal_embedding" in req.vlm_inputs:
            return req.vlm_inputs["multimodal_embedding"]
        return None

    def test_single_image_requests(self):
        """Test batch of requests each with a single image."""
        # Create requests with different image sizes
        req1 = self._create_req_with_image("req1", grid_h=28, grid_w=28, seed=1)
        req2 = self._create_req_with_image("req2", grid_h=14, grid_w=14, seed=2)
        req3 = self._create_req_with_image("req3", grid_h=28, grid_w=14, seed=3)

        # Deep copy for individual forward
        req1_single = self._deep_copy_req(req1)
        req2_single = self._deep_copy_req(req2)
        req3_single = self._deep_copy_req(req3)

        # Individual forward
        self.model_runner.forward(req1_single, self.mesh)
        self.model_runner.forward(req2_single, self.mesh)
        self.model_runner.forward(req3_single, self.mesh)

        # Batch forward
        batch_reqs = [req1, req2, req3]
        self.model_runner.forward_batch(batch_reqs, self.mesh)

        # Compare results
        for single_req, batch_req in [
            (req1_single, req1),
            (req2_single, req2),
            (req3_single, req3),
        ]:
            single_embed = self._get_embedding(single_req)
            batch_embed = self._get_embedding(batch_req)

            self.assertIsNotNone(
                single_embed, f"Single forward should produce embedding for {single_req.rid}"
            )
            self.assertIsNotNone(
                batch_embed, f"Batch forward should produce embedding for {batch_req.rid}"
            )
            self.assertEqual(
                single_embed.shape, batch_embed.shape, f"Shape mismatch for {single_req.rid}"
            )

            np.testing.assert_allclose(
                np.array(single_embed),
                np.array(batch_embed),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Embedding mismatch for {single_req.rid}",
            )

    def test_mixed_image_video_requests(self):
        """Test batch with mixed image and video requests."""
        # req1: image only
        req1 = self._create_req_with_image("req1", grid_h=14, grid_w=14, seed=10)
        # req2: video only
        req2 = self._create_req_with_video("req2", grid_t=2, grid_h=14, grid_w=14, seed=20)
        # req3: both image and video
        req3 = self._create_req_with_image_and_video(
            "req3", img_h=14, img_w=14, vid_t=2, vid_h=14, vid_w=14, seed=30
        )

        # Deep copy for individual forward
        req1_single = self._deep_copy_req(req1)
        req2_single = self._deep_copy_req(req2)
        req3_single = self._deep_copy_req(req3)

        # Individual forward
        self.model_runner.forward(req1_single, self.mesh)
        self.model_runner.forward(req2_single, self.mesh)
        self.model_runner.forward(req3_single, self.mesh)

        # Batch forward
        batch_reqs = [req1, req2, req3]
        self.model_runner.forward_batch(batch_reqs, self.mesh)

        # Compare results
        for single_req, batch_req in [
            (req1_single, req1),
            (req2_single, req2),
            (req3_single, req3),
        ]:
            single_embed = self._get_embedding(single_req)
            batch_embed = self._get_embedding(batch_req)

            self.assertIsNotNone(
                single_embed, f"Single forward should produce embedding for {single_req.rid}"
            )
            self.assertIsNotNone(
                batch_embed, f"Batch forward should produce embedding for {batch_req.rid}"
            )
            self.assertEqual(
                single_embed.shape, batch_embed.shape, f"Shape mismatch for {single_req.rid}"
            )

            np.testing.assert_allclose(
                np.array(single_embed),
                np.array(batch_embed),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Embedding mismatch for {single_req.rid}",
            )

    def test_batch_size_one_equals_forward(self):
        """Test that forward_batch with size 1 equals forward."""
        req = self._create_req_with_image("req_single", grid_h=14, grid_w=14, seed=42)
        req_copy = self._deep_copy_req(req)

        # Single forward
        self.model_runner.forward(req, self.mesh)

        # Batch forward with size 1
        self.model_runner.forward_batch([req_copy], self.mesh)

        single_embed = self._get_embedding(req)
        batch_embed = self._get_embedding(req_copy)

        self.assertIsNotNone(single_embed)
        self.assertIsNotNone(batch_embed)

        np.testing.assert_allclose(
            np.array(single_embed),
            np.array(batch_embed),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
