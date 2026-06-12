"""Neutral MiMo-Audio mel front-end processor.

``MiMoAudioProcessor`` converts a raw audio waveform into the log-mel
spectrogram the MiMo audio models expect. It is shared between the
generation-side ``MultimodalTokenizer`` (``multimodal/manager``) and the
understanding-side MiMo-V2.5 host codec processor (``srt/models/mimo_v2_5``), so
it lives here in the neutral ``mm_core`` family: the understanding processor can
import it without an ``srt -> multimodal`` import edge.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MiMoAudioProcessor:
    """Custom processor for MiMo Audio models."""

    def __init__(self):
        from transformers.audio_utils import mel_filter_bank, window_function

        sample_rate = 24000
        n_fft = 960
        hop_length = 240
        win_length = 960
        f_min = 0
        f_max = 12000
        n_mels = 128

        self.sampling_rate = sample_rate
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=n_mels,
            min_frequency=f_min,
            max_frequency=f_max,
            sampling_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        )
        self.window = window_function(win_length, "hann")
        self.mel_params = {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
        }

        logger.info(
            "Initialized MiMoAudioProcessor: sr=%d, n_fft=%d, hop=%d, n_mels=%d",
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
        )

    def __call__(self, audio_array: np.ndarray, sampling_rate: int = None) -> tuple:
        """Convert raw audio waveform to mel spectrogram.

        This matches the official MiMo Audio implementation:
        - Uses power=1.0 (amplitude spectrogram)
        - Applies natural log via log_mel="log"
        - Returns mel spectrogram in [batch, time, n_mels] format

        Args:
            audio_array: Raw audio waveform as numpy array, shape (samples,).
            sampling_rate: Input audio sample rate. If different from target rate, will resample.

        Returns:
            Tuple of (mel_spectrogram, input_lengths) as numpy arrays.
            mel_spectrogram shape: [batch, time, n_mels]
        """
        from transformers.audio_utils import spectrogram

        if audio_array.ndim == 2:
            audio_array = audio_array.squeeze(0)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            audio_array = self._resample_audio(audio_array, sampling_rate, self.sampling_rate)

        mels = spectrogram(
            waveform=audio_array,
            window=self.window,
            frame_length=self.mel_params["n_fft"],
            hop_length=self.mel_params["hop_length"],
            fft_length=self.mel_params["n_fft"],
            power=1.0,  # Amplitude spectrogram (matches official MiMo)
            center=True,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1e-7,
        )

        mels = mels.T[None, :, :]
        input_lens = np.array([mels.shape[1]])

        return mels, input_lens

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using torchaudio.

        Uses torchaudio.functional.resample to match official MiMo implementation.

        Args:
            audio: Input audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        import torch
        import torchaudio

        audio_tensor = torch.from_numpy(audio).float()
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        logger.info(
            "Resampled audio from %d Hz to %d Hz (%d -> %d samples)",
            orig_sr,
            target_sr,
            len(audio),
            len(resampled),
        )
        return resampled.numpy().astype(np.float32)
