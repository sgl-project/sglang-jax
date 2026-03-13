import logging
import os

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.io_struct import AbortReq, ProfileReq
from sgl_jax.srt.managers.scheduler_profiler_mixing import SchedulerProfilerMixin
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.vae.vae_model_worker import VaeModelWorker
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


class VaeScheduler(SchedulerProfilerMixin):
    """Scheduler for VAE model inference within the multimodal pipeline.

    Responsibilities:
    - Receive batched requests via a communication backend and prepare inputs.
    - Preprocess latents according to model config (scaling/shift).
    - Move input arrays onto JAX devices using the provided `mesh` and a
      `NamedSharding`/`PartitionSpec` before forwarding to the VAE worker.
    - Run the VAE forward pass and return/send outputs via the communication
      backend.

    The scheduler assumes a `VaeModelWorker` handles model execution and that
    `communication_backend` provides `recv_requests()` and `send_pyobj()`.
    """

    def __init__(
        self,
        server_args: MultimodalServerArgs,
        communication_backend: CommunicationBackend,
        mesh: jax.sharding.Mesh,
        model_class,
        stage_sub_dir: str | None = None,
        precompile_params: dict | None = None,
        **kwargs,
    ):
        """Initialize the VaeScheduler.

        Args:
            server_args: Multimodal server configuration.
            communication_backend: Backend used to receive/send requests.
            mesh: JAX device mesh used for sharding inputs/outputs.
            model_class: The VAE model class; used to build `VaeModelWorker` and
                to obtain model-specific configuration values.
        """

        self._comm_backend = communication_backend
        self.mesh = mesh
        self.vae_worker = VaeModelWorker(
            model_class=model_class,
            mesh=self.mesh,
            server_args=server_args,
            stage_sub_dir=stage_sub_dir,
        )
        self.server_args = server_args
        self.forward_ct = 0
        self.init_profier()
        self.model_config = model_class.get_config_class()()
        # Track aborted request IDs to skip processing
        self.aborted_rids: set[str] = set()
        if not server_args.disable_precompile:
            logger.info("[VAE Scheduler] Begins to run vae worker precompile.")
            self.vae_worker.run_precompile()
            logger.info("[VAE Scheduler] Completes vae worker precompile.")

        # Initialize audio decoder + vocoder if available
        self.audio_decoder = None
        self.vocoder = None
        self._init_audio_models(mesh)

    def event_loop_normal(self):
        """Main blocking loop used in non-async environments.

        Repeatedly polls the `communication_backend` for requests, applies
        `preprocess`, shards `req.latents` onto `self.mesh` with a
        `NamedSharding(PartitionSpec())`, and then processes the batch via
        `run_vae_batch`. AbortReq messages are processed to track aborted
        request IDs, and any Req whose rid matches an aborted ID is skipped.
        """

        while True:
            reqs = self._comm_backend.recv_requests()
            if len(reqs) > 0:
                # Process abort requests first
                valid_reqs = []
                for req in reqs:
                    if isinstance(req, AbortReq):
                        logger.info("VaeScheduler received abort for rid=%s", req.rid)
                        self.aborted_rids.add(req.rid)
                    elif isinstance(req, ProfileReq):
                        result = self.profile(req)
                        self._comm_backend.send_pyobj(result)
                    elif isinstance(req, Req):
                        # Check if this request was aborted
                        if req.rid in self.aborted_rids:
                            logger.info("VaeScheduler skipping aborted request rid=%s", req.rid)
                            self.aborted_rids.discard(req.rid)
                            continue
                        assert req.latents is not None
                        self.preprocess(req)
                        req.latents = device_array(
                            req.latents, sharding=NamedSharding(self.mesh, PartitionSpec())
                        )
                        valid_reqs.append(req)
                    else:
                        logger.warning("VaeScheduler received unknown request type: %s", type(req))

                if valid_reqs:
                    self.run_vae_batch(valid_reqs)
            else:
                self._comm_backend.wait_for_new_requests(0.001)

    def preprocess(self, req):
        """Apply model-specific preprocessing to a single `Req`.

        Common operations: divide by `scaling_factor` and add `shift_factor`
        if those attributes exist on the model config. This prepares latents
        to match the expected value range for the VAE decoder.
        """

        if hasattr(self.model_config, "scaling_factor"):
            req.latents = req.latents / self.model_config.scaling_factor
        if hasattr(self.model_config, "shift_factor"):
            req.latents += self.model_config.shift_factor
        req.latents = jax.device_get(req.latents)
        latents_t_padding = 0
        if self.server_args.vae_decode_precompile_frame_paddings is not None and hasattr(
            self.model_config, "scale_factor_temporal"
        ):
            for n_frame in self.server_args.vae_decode_precompile_frame_paddings:
                latents_t = (n_frame - 1) // self.model_config.scale_factor_temporal + 1
                if latents_t >= req.latents.shape[1]:
                    latents_t_padding = latents_t - req.latents.shape[1]
                    break
        req.latents = np.pad(
            req.latents,
            pad_width=((0, 0), (0, latents_t_padding), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def run_vae_batch(self, batch: list[Req]):
        """Run the VAE forward pass for a batch of requests.

        For each `Req` in `batch`, invokes the `VaeModelWorker.forward`, moves
        the result back to host memory with `jax.device_get`, clears the
        latent to free memory, and sends the completed request through the
        communication backend.
        """

        for req in batch:
            output, cache_miss = self.vae_worker.forward(req)
            logger.info("VAE forward pass cache miss: %s", cache_miss)
            req.output = jax.device_get(output[:, : req.num_frames, :, :, :])

            # Decode audio latents if present
            if getattr(req, "audio_latents", None) is not None and self.audio_decoder is not None:
                self._decode_audio(req)
            # Clear all JAX array fields before sending to detokenizer process.
            # The detokenizer runs in a separate process without TPU access, so
            # any remaining jax.Array fields will fail to unpickle.
            req.latents = None
            req.audio_latents = None
            req.prompt_embeds = None
            req.negative_prompt_embeds = None
            req.audio_prompt_embeds = None
            req.audio_negative_prompt_embeds = None
            req.prompt_attention_mask = None
            req.negative_attention_mask = None
            req.clip_embedding_pos = None
            req.clip_embedding_neg = None
            req.image_embeds = []
            req.noise_pred = None
            req.timesteps = None
            req.timestep = None
            req.image_latent = None
            req.raw_latent_shape = None
            req.trajectory_latents = None
            req.trajectory_timesteps = None
            req.vision_embeds = None
            req.input_embeds = None
            req.audio_features = None
            req.pixel_values = None
            req.preprocessed_image = None
            req.pixel_values_images = None
            req.pixel_values_videos = None
            req.audio_input = None
            req.mel_input = None
            req.mel_input_lens = None
            req.codes = None
            req.audio_codes = None
            req.backbone_cache = None
            req.generated_text_tokens = None
            req.generated_audio_tokens = None
            req.text_logits = None
            req.pooled_embeds = []
            req.neg_pooled_embeds = []
            req.pil_image = None
            self.forward_ct += 1
            self._profile_batch_predicate(None)
            self._comm_backend.send_pyobj(req)

    def _init_audio_models(self, mesh):
        """Load AudioDecoder and Vocoder for audio latent decoding."""
        try:
            from sgl_jax.srt.multimodal.models.ltx2.audio_vae.ltx2_audio_vae import AudioDecoder
            from sgl_jax.srt.multimodal.models.ltx2.audio_vae.ltx2_audio_vae_config import (
                LTX2AudioVAEDecoderConfig, LTX2VocoderConfig,
            )
            from sgl_jax.srt.multimodal.models.ltx2.audio_vae.vocoder import Vocoder

            # Check if LTX-2 checkpoint exists (needed for weights)
            from sgl_jax.srt.multimodal.models.ltx2.utils import get_hf_snapshot_dir
            ltx_path = get_hf_snapshot_dir("Lightricks/LTX-2")
            if not ltx_path:
                logger.info("LTX-2 cache not found, skipping audio model init")
                return

            class _SimpleConfig:
                def __init__(self, path):
                    self.model_path = path

            model_cfg = _SimpleConfig(ltx_path)

            logger.info("Loading AudioDecoder...")
            with jax.set_mesh(mesh):
                decoder = nnx.eval_shape(
                    lambda: AudioDecoder(config=LTX2AudioVAEDecoderConfig(), mesh=mesh, dtype=jnp.float32)
                )
            decoder.load_weights(model_cfg)
            self.audio_decoder = decoder
            logger.info("AudioDecoder loaded")

            logger.info("Loading Vocoder...")
            with jax.set_mesh(mesh):
                vocoder = nnx.eval_shape(
                    lambda: Vocoder(config=LTX2VocoderConfig(), mesh=mesh, dtype=jnp.float32)
                )
            vocoder.load_weights(model_cfg)
            self.vocoder = vocoder
            logger.info("Vocoder loaded")
        except Exception as e:
            logger.warning("Failed to load audio models: %s", e)
            self.audio_decoder = None
            self.vocoder = None

    def _decode_audio(self, req: Req):
        """Decode audio latents to waveform and save as WAV alongside the video."""
        try:
            audio_latents = jnp.array(req.audio_latents, dtype=jnp.float32)  # (B, T, 128)
            # Unpatchify: (B, T, C*F) -> (B, T, F, C) where C=8, F=16
            # PyTorch: rearrange("b t (c f) -> b c t f", c=8, f=16) — c varies slower
            # So reshape to (B, T, C, F) first, then transpose to channel-last (B, T, F, C)
            b, t, cf = audio_latents.shape
            c, f = 8, 16
            audio_latents = audio_latents.reshape(b, t, c, f)  # (B, T, C=8, F=16)
            audio_latents = jnp.swapaxes(audio_latents, -1, -2)  # (B, T, F=16, C=8) JAX channel-last

            # Audio VAE decode: (B, T, F, C) -> (B, T', mel_bins, stereo)
            mel_spec = self.audio_decoder(audio_latents)
            logger.info("Audio VAE decoded: %s -> %s", audio_latents.shape, mel_spec.shape)

            # Vocoder: mel spec -> waveform
            waveform = self.vocoder(mel_spec)
            waveform = jax.device_get(waveform)  # (B, audio_len, 2)
            logger.info("Vocoder output: %s", waveform.shape)

            # Save as WAV
            import scipy.io.wavfile
            wav_data = np.array(waveform[0])  # (audio_len, 2)
            wav_data = np.clip(wav_data, -1.0, 1.0)
            wav_data = (wav_data * 32767).astype(np.int16)
            sample_rate = 24000  # Vocoder output rate
            output_path = getattr(req, "output_path", "outputs/")
            wav_path = os.path.join(output_path, f"{req.rid}.wav")
            os.makedirs(output_path, exist_ok=True)
            scipy.io.wavfile.write(wav_path, sample_rate, wav_data)
            logger.info("Saved audio to %s", wav_path)
        except Exception as e:
            logger.warning("Audio decode failed: %s", e)
