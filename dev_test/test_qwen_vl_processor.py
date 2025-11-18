import types
import jax.numpy as jnp

from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sgl_jax.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sgl_jax.srt.server_args import ServerArgs


class DummyProcessor:
    """Minimal stub to satisfy base class expectations."""
    tokenizer = None
    image_processor = None


class TestQwenVLProcessor(QwenVLImageProcessor):
    """Subclass to override heavy HF processing with a deterministic stub."""

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        # Create a simple sequence with one contiguous run of image_token_id
        img_id = self.mm_tokens.image_token_id
        # tokens: [11, 22, <img>, <img>, 33]
        input_ids = jnp.array([11, 22, img_id, img_id, 33], dtype=jnp.int32)
        # Return a dict emulating HF processor output keys that our collector understands.
        return {
            "input_ids": input_ids,
            # include a feature key so it is collected as an IMAGE modality item
            "pixel_values": jnp.zeros((1, 3, 8, 8), dtype=jnp.float32),
        }


def make_min_hf_config():
    # Build a minimal config object with required attributes
    vision_config = types.SimpleNamespace(spatial_merge_size=1, tokens_per_second=None)
    cfg = types.SimpleNamespace(
        model_type="qwen2_5_vl",
        vision_start_token_id=50000,
        vision_end_token_id=50001,
        image_token_id=50002,
        video_token_id=None,
        audio_token_id=None,
        vision_config=vision_config,
    )
    return cfg


def test_process_and_offsets():
    server_args = ServerArgs()

    hf_cfg = make_min_hf_config()
    proc = TestQwenVLProcessor(hf_cfg, server_args, DummyProcessor())

    # Build a base output indicating there is an image to trigger process_mm_data
    base_output = BaseMultiModalProcessorOutput(
        input_text="prompt <|vision_start|><|image_pad|><|vision_end|> end",
        images=[{"dummy": True}],
        videos=[],
        audios=[],
    )

    mm_items, input_ids, ret = proc.process_and_combine_mm_data(base_output, proc.mm_tokens)

    # Assertions
    assert isinstance(input_ids, jnp.ndarray)
    # From our stub: [11, 22, img_id, img_id, 33]
    assert input_ids.tolist() == [11, 22, proc.mm_tokens.image_token_id, proc.mm_tokens.image_token_id, 33]

    # One IMAGE item collected with a single contiguous run at indices (2, 3)
    assert len(mm_items) == 1, f"Expected 1 item, got {len(mm_items)}"
    item = mm_items[0]
    assert item.is_image(), "Collected item should be image modality"
    assert item.offsets == [(2, 3)], f"Unexpected offsets: {item.offsets}"


if __name__ == "__main__":
    test_process_and_offsets()
    print("QwenVLImageProcessor tests passed.")
