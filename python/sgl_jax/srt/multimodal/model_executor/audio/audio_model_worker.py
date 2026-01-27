from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_model_runner import AudioModelRunner
from sgl_jax.srt.server_args import ServerArgs


class AudioModelWorker:
    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioModelRunner(server_args, mesh, model_class=model_class)

    def forward(self, batch: Req, mode: str = "encode", **kwargs):
        if mode == "encode":
            # Use mel_input (preprocessed in tokenizer) instead of audio_input
            return self.model_runner.forward(
                batch.mel_input,
                batch.mel_input_lens,
                mode,
                **kwargs
            )
        elif mode == "decode":
            return self.model_runner.forward(batch.codes, None, mode)
