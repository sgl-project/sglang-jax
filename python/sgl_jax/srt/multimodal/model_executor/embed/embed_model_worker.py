import jax
import numpy as np

from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req


class EmbedModelWorker:
    """Worker shell for the Embed stage."""

    def __init__(
        self, server_args: MultimodalServerArgs, mesh: jax.sharding.Mesh = None, model_class=None
    ):
        self.mesh = mesh
        self.model_runner = None  # VitModelRunner(server_args, self.mesh, model_class=model_class)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, batch: Req, mesh: jax.sharding.Mesh):
        mm_inputs = batch.vlm_inputs if isinstance(batch.vlm_inputs, dict) else None
        if mm_inputs is not None:
            input_ids = batch.input_ids or batch.origin_input_ids
            if input_ids is not None:
                mm_inputs["multimodal_embedding"] = np.random.randn(1128, 2048)
                mm_inputs["deepstack_visual_pos_mask"] = (np.array(input_ids) == 151655).astype(
                    np.int32
                )
                mm_inputs["deepstack_visual_embedding"] = np.random.randn(3, 1107, 2048)
        # mock result
        return batch
        # return self.model_runner.forward(batch, mesh)
