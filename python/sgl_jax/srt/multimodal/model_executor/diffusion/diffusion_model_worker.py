from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)


class DiffusionModelWorker:
    def __init__(self, server_args: MultimodalServerArgs, mesh):
        self.mesh = mesh
        self.model_runner = DiffusionModelRunner(server_args, self.mesh)
        self.initialize()

    def initialize(self):
        self.model_loader.load_model()
        pass
        # init cache here if needed
        # init different attention backend if needed

    def forward(self, batch, mesh):
        # Implement the diffusion model inference logic here
        # latents: Array,
        # text_embeds: Array,
        # negative_embeds: Array,
        # num_steps: int = 30,
        # guidance_scale: float = 5.0,
        # scheduler: Optional[FlaxUniPCMultistepScheduler] = None,
        # scheduler_state: Optional[UniPCMultistepSchedulerState] = None,

        """
        Generate video from text embeddings using the diffusion model.

        Args:
            text_embeds: [B, seq_len, text_dim] text embeddings from UMT5
            num_frames: Number of frames to generate
            latent_size: Spatial size of latents
            num_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (5-6 recommended)

        Returns:
            latents: [B, T, H, W, C] generated video latents
        """

        self.model_runner.forward(batch, mesh)
