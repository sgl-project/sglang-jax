"""
Weight mappings for loading LTX-2 Audio VAE and Vocoder weights from checkpoint.

Checkpoint key formats:
  Audio VAE Encoder: audio_vae.encoder.*
  Audio VAE Decoder: audio_vae.decoder.*
  Per-channel stats:  audio_vae.per_channel_statistics.*
  Vocoder:           vocoder.*

Conv2D weights: PyTorch [O, I, H, W] → JAX [H, W, I, O] via transpose (2, 3, 1, 0)
Conv1D weights: PyTorch [O, I, K] → JAX [K, I, O] via transpose (2, 1, 0)
ConvTranspose1D weights: PyTorch [I, O, K] → JAX [K, I, O] via transpose (2, 0, 1)
"""

from sgl_jax.srt.utils.weight_utils import WeightMapping

# Conv2D: PyTorch [O, I, H, W] → JAX [H, W, I, O]
_CONV2D_TRANSPOSE = (2, 3, 1, 0)

# Conv1D: PyTorch [O, I, K] → JAX [K, I, O]
_CONV1D_TRANSPOSE = (2, 1, 0)

# ConvTranspose1D: PyTorch [I, O, K] → JAX [K, I, O]
# Note: PyTorch ConvTranspose stores kernel as [in_ch, out_ch, kernel]
_CONV_TRANSPOSE_1D_TRANSPOSE = (2, 0, 1)


def create_audio_vae_encoder_weight_mappings() -> dict[str, WeightMapping]:
    """Create weight mappings for AudioEncoder.

    Checkpoint structure (from analysis):
      audio_vae.encoder.conv_in.conv.{weight,bias}
      audio_vae.encoder.down.{0,1,2}.block.{0,1}.conv{1,2}.conv.{weight,bias}
      audio_vae.encoder.down.{0,1,2}.block.{0}.nin_shortcut.conv.{weight,bias}  (levels 1,2 only)
      audio_vae.encoder.down.{0,1}.downsample.conv.{weight,bias}
      audio_vae.encoder.mid.block_{1,2}.conv{1,2}.conv.{weight,bias}
      audio_vae.encoder.conv_out.conv.{weight,bias}
      audio_vae.per_channel_statistics.{std-of-means,mean-of-means}
    """
    m = {}

    # Per-channel statistics
    m["audio_vae.per_channel_statistics.std-of-means"] = WeightMapping(
        target_path="per_channel_statistics.std_of_means", sharding=(None,),
    )
    m["audio_vae.per_channel_statistics.mean-of-means"] = WeightMapping(
        target_path="per_channel_statistics.mean_of_means", sharding=(None,),
    )

    # conv_in
    m["audio_vae.encoder.conv_in.conv.weight"] = WeightMapping(
        target_path="conv_in.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.conv_in.conv.bias"] = WeightMapping(
        target_path="conv_in.conv.bias", sharding=(None,),
    )

    # Down blocks: down[level].block[idx].conv{1,2} and nin_shortcut
    # Using wildcards for level and block index
    m["audio_vae.encoder.down.*.block.*.conv1.conv.weight"] = WeightMapping(
        target_path="down.*.0.*.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.down.*.block.*.conv1.conv.bias"] = WeightMapping(
        target_path="down.*.0.*.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.encoder.down.*.block.*.conv2.conv.weight"] = WeightMapping(
        target_path="down.*.0.*.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.down.*.block.*.conv2.conv.bias"] = WeightMapping(
        target_path="down.*.0.*.conv2.conv.bias", sharding=(None,),
    )
    # nin_shortcut (1x1 conv for channel mismatch)
    m["audio_vae.encoder.down.*.block.*.nin_shortcut.conv.weight"] = WeightMapping(
        target_path="down.*.0.*.nin_shortcut.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.down.*.block.*.nin_shortcut.conv.bias"] = WeightMapping(
        target_path="down.*.0.*.nin_shortcut.bias", sharding=(None,),
    )

    # Downsample convs (stride=2 conv)
    # Checkpoint: audio_vae.encoder.down.{0,1}.downsample.conv.{weight,bias}
    # Model: down[level][2] (third element of tuple) is Downsample2D with .conv
    m["audio_vae.encoder.down.*.downsample.conv.weight"] = WeightMapping(
        target_path="down.*.2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.down.*.downsample.conv.bias"] = WeightMapping(
        target_path="down.*.2.conv.bias", sharding=(None,),
    )

    # Mid blocks
    m["audio_vae.encoder.mid.block_1.conv1.conv.weight"] = WeightMapping(
        target_path="mid_block_1.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_1.conv1.conv.bias"] = WeightMapping(
        target_path="mid_block_1.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_1.conv2.conv.weight"] = WeightMapping(
        target_path="mid_block_1.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_1.conv2.conv.bias"] = WeightMapping(
        target_path="mid_block_1.conv2.conv.bias", sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_2.conv1.conv.weight"] = WeightMapping(
        target_path="mid_block_2.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_2.conv1.conv.bias"] = WeightMapping(
        target_path="mid_block_2.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_2.conv2.conv.weight"] = WeightMapping(
        target_path="mid_block_2.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.mid.block_2.conv2.conv.bias"] = WeightMapping(
        target_path="mid_block_2.conv2.conv.bias", sharding=(None,),
    )

    # conv_out
    m["audio_vae.encoder.conv_out.conv.weight"] = WeightMapping(
        target_path="conv_out.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.encoder.conv_out.conv.bias"] = WeightMapping(
        target_path="conv_out.conv.bias", sharding=(None,),
    )

    return m


def create_audio_vae_decoder_weight_mappings() -> dict[str, WeightMapping]:
    """Create weight mappings for AudioDecoder.

    Checkpoint structure (from analysis):
      audio_vae.decoder.conv_in.conv.{weight,bias}     [512, 8, 3, 3]
      audio_vae.decoder.mid.block_{1,2}.conv{1,2}.conv.{weight,bias}
      audio_vae.decoder.up.{0,1,2}.block.{0,1,2}.conv{1,2}.conv.{weight,bias}
      audio_vae.decoder.up.{0,1}.block.0.nin_shortcut.conv.{weight,bias}
      audio_vae.decoder.up.{1,2}.upsample.conv.conv.{weight,bias}
      audio_vae.decoder.conv_out.conv.{weight,bias}    [2, 128, 3, 3]
      audio_vae.per_channel_statistics.{std-of-means,mean-of-means}
    """
    m = {}

    # Per-channel statistics (shared with encoder in checkpoint)
    m["audio_vae.per_channel_statistics.std-of-means"] = WeightMapping(
        target_path="per_channel_statistics.std_of_means", sharding=(None,),
    )
    m["audio_vae.per_channel_statistics.mean-of-means"] = WeightMapping(
        target_path="per_channel_statistics.mean_of_means", sharding=(None,),
    )

    # conv_in
    m["audio_vae.decoder.conv_in.conv.weight"] = WeightMapping(
        target_path="conv_in.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.conv_in.conv.bias"] = WeightMapping(
        target_path="conv_in.conv.bias", sharding=(None,),
    )

    # Mid blocks
    m["audio_vae.decoder.mid.block_1.conv1.conv.weight"] = WeightMapping(
        target_path="mid_block_1.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_1.conv1.conv.bias"] = WeightMapping(
        target_path="mid_block_1.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_1.conv2.conv.weight"] = WeightMapping(
        target_path="mid_block_1.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_1.conv2.conv.bias"] = WeightMapping(
        target_path="mid_block_1.conv2.conv.bias", sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_2.conv1.conv.weight"] = WeightMapping(
        target_path="mid_block_2.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_2.conv1.conv.bias"] = WeightMapping(
        target_path="mid_block_2.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_2.conv2.conv.weight"] = WeightMapping(
        target_path="mid_block_2.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.mid.block_2.conv2.conv.bias"] = WeightMapping(
        target_path="mid_block_2.conv2.conv.bias", sharding=(None,),
    )

    # Up blocks: up[level].block[idx].conv{1,2} and nin_shortcut
    # Checkpoint uses up.{level}.block.{idx}
    # Model stores as self.up[level] = (blocks_list, attns_list, upsample)
    m["audio_vae.decoder.up.*.block.*.conv1.conv.weight"] = WeightMapping(
        target_path="up.*.0.*.conv1.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.up.*.block.*.conv1.conv.bias"] = WeightMapping(
        target_path="up.*.0.*.conv1.conv.bias", sharding=(None,),
    )
    m["audio_vae.decoder.up.*.block.*.conv2.conv.weight"] = WeightMapping(
        target_path="up.*.0.*.conv2.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.up.*.block.*.conv2.conv.bias"] = WeightMapping(
        target_path="up.*.0.*.conv2.conv.bias", sharding=(None,),
    )
    m["audio_vae.decoder.up.*.block.*.nin_shortcut.conv.weight"] = WeightMapping(
        target_path="up.*.0.*.nin_shortcut.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.up.*.block.*.nin_shortcut.conv.bias"] = WeightMapping(
        target_path="up.*.0.*.nin_shortcut.bias", sharding=(None,),
    )

    # Upsample convs
    m["audio_vae.decoder.up.*.upsample.conv.conv.weight"] = WeightMapping(
        target_path="up.*.2.conv.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.up.*.upsample.conv.conv.bias"] = WeightMapping(
        target_path="up.*.2.conv.conv.bias", sharding=(None,),
    )

    # conv_out
    m["audio_vae.decoder.conv_out.conv.weight"] = WeightMapping(
        target_path="conv_out.conv.kernel", transpose_axes=_CONV2D_TRANSPOSE, sharding=(None,),
    )
    m["audio_vae.decoder.conv_out.conv.bias"] = WeightMapping(
        target_path="conv_out.conv.bias", sharding=(None,),
    )

    return m


def create_vocoder_weight_mappings() -> dict[str, WeightMapping]:
    """Create weight mappings for Vocoder (HiFi-GAN).

    Checkpoint structure:
      vocoder.conv_pre.{weight,bias}          [1024, 128, 7]
      vocoder.ups.{0-4}.{weight,bias}         ConvTranspose1d
      vocoder.resblocks.{0-14}.convs1.{0-2}.{weight,bias}
      vocoder.resblocks.{0-14}.convs2.{0-2}.{weight,bias}
      vocoder.conv_post.{weight,bias}         [2, 32, 7]
    """
    m = {}

    # conv_pre
    m["vocoder.conv_pre.weight"] = WeightMapping(
        target_path="conv_pre.kernel", transpose_axes=_CONV1D_TRANSPOSE, sharding=(None,),
    )
    m["vocoder.conv_pre.bias"] = WeightMapping(
        target_path="conv_pre.bias", sharding=(None,),
    )

    # Upsample ConvTranspose1d layers
    m["vocoder.ups.*.weight"] = WeightMapping(
        target_path="ups.*.conv_transpose.kernel", transpose_axes=_CONV_TRANSPOSE_1D_TRANSPOSE, sharding=(None,),
    )
    m["vocoder.ups.*.bias"] = WeightMapping(
        target_path="ups.*.conv_transpose.bias", sharding=(None,),
    )

    # ResBlocks: convs1 and convs2
    m["vocoder.resblocks.*.convs1.*.weight"] = WeightMapping(
        target_path="resblocks.*.convs1.*.kernel", transpose_axes=_CONV1D_TRANSPOSE, sharding=(None,),
    )
    m["vocoder.resblocks.*.convs1.*.bias"] = WeightMapping(
        target_path="resblocks.*.convs1.*.bias", sharding=(None,),
    )
    m["vocoder.resblocks.*.convs2.*.weight"] = WeightMapping(
        target_path="resblocks.*.convs2.*.kernel", transpose_axes=_CONV1D_TRANSPOSE, sharding=(None,),
    )
    m["vocoder.resblocks.*.convs2.*.bias"] = WeightMapping(
        target_path="resblocks.*.convs2.*.bias", sharding=(None,),
    )

    # conv_post
    m["vocoder.conv_post.weight"] = WeightMapping(
        target_path="conv_post.kernel", transpose_axes=_CONV1D_TRANSPOSE, sharding=(None,),
    )
    m["vocoder.conv_post.bias"] = WeightMapping(
        target_path="conv_post.bias", sharding=(None,),
    )

    return m
