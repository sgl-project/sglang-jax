"""Configuration classes for LTX-2 Audio VAE components."""


class LTX2AudioVAEEncoderConfig:
    """Config for LTX-2 Audio VAE Encoder."""

    model_type = "ltx2_audio_vae_encoder"

    def __init__(self):
        self.ch = 128
        self.ch_mult = (1, 2, 4)
        self.num_res_blocks = 2
        self.z_channels = 8
        self.double_z = True
        self.in_channels = 2  # stereo
        self.resolution = 256
        self.attn_resolutions = set()  # no attention at default resolutions


class LTX2AudioVAEDecoderConfig:
    """Config for LTX-2 Audio VAE Decoder."""

    model_type = "ltx2_audio_vae_decoder"

    def __init__(self):
        self.ch = 128
        self.out_ch = 2  # stereo
        self.ch_mult = (1, 2, 4)
        self.num_res_blocks = 2
        self.z_channels = 8
        self.resolution = 256
        self.attn_resolutions = set()
        self.mel_bins = 64


class LTX2VocoderConfig:
    """Config for LTX-2 HiFi-GAN Vocoder."""

    model_type = "ltx2_vocoder"

    def __init__(self):
        self.resblock_kernel_sizes = [3, 7, 11]
        self.upsample_rates = [6, 5, 2, 2, 2]
        self.upsample_kernel_sizes = [16, 15, 8, 4, 4]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_initial_channel = 1024
        self.stereo = True
        self.output_sample_rate = 24000
