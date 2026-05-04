"""
Weight mapping utilities for loading LTX-2 VAE decoder weights from checkpoint.

Maps checkpoint keys (PyTorch format) to JAX model parameter paths using the
WeightLoader/WeightMapping infrastructure from sgl_jax.srt.utils.weight_utils.

Checkpoint key format: vae.decoder.* and vae.per_channel_statistics.*
Conv3D weights: PyTorch [O, I, T, H, W] -> JAX [T, H, W, I, O] via transpose_axes=(2,3,4,1,0)
"""

from sgl_jax.srt.utils.weight_utils import WeightMapping

# Conv3D transpose from PyTorch [O, I, T, H, W] to JAX [T, H, W, I, O]
_CONV3D_TRANSPOSE = (2, 3, 4, 1, 0)


def create_vae_decoder_weight_mappings() -> dict[str, WeightMapping]:
    """
    Create weight mappings from LTX-2 checkpoint keys to JAX VideoDecoder param paths.

    The LTX-2 checkpoint stores VAE decoder weights under:
    - vae.per_channel_statistics.{std-of-means, mean-of-means}
    - vae.decoder.conv_in.conv.{weight, bias}
    - vae.decoder.up_blocks.{0,2,4,6}.res_blocks.{0-4}.conv{1,2}.conv.{weight, bias}
    - vae.decoder.up_blocks.{1,3,5}.conv.conv.{weight, bias}
    - vae.decoder.conv_out.conv.{weight, bias}

    Returns:
        Dict mapping checkpoint keys to WeightMapping objects.
        Uses wildcards (*) for repeated block/layer patterns.
    """
    mappings = {}

    # Per-channel statistics (1D, no transpose needed)
    mappings["vae.per_channel_statistics.std-of-means"] = WeightMapping(
        target_path="per_channel_statistics.std_of_means",
        sharding=(None,),
    )
    mappings["vae.per_channel_statistics.mean-of-means"] = WeightMapping(
        target_path="per_channel_statistics.mean_of_means",
        sharding=(None,),
    )

    # Input convolution
    mappings["vae.decoder.conv_in.conv.weight"] = WeightMapping(
        target_path="conv_in.conv.kernel",
        transpose_axes=_CONV3D_TRANSPOSE,
        sharding=(None,),
    )
    mappings["vae.decoder.conv_in.conv.bias"] = WeightMapping(
        target_path="conv_in.conv.bias",
        sharding=(None,),
    )

    # Up blocks - residual blocks (UNetMidBlock3D at indices 0,2,4,6)
    # Checkpoint: res_blocks -> Model: resnets
    # Each res block has conv1 and conv2 (both CausalConv3d)
    mappings["vae.decoder.up_blocks.*.res_blocks.*.conv1.conv.weight"] = WeightMapping(
        target_path="up_blocks.*.resnets.*.conv1.conv.kernel",
        transpose_axes=_CONV3D_TRANSPOSE,
        sharding=(None,),
    )
    mappings["vae.decoder.up_blocks.*.res_blocks.*.conv1.conv.bias"] = WeightMapping(
        target_path="up_blocks.*.resnets.*.conv1.conv.bias",
        sharding=(None,),
    )
    mappings["vae.decoder.up_blocks.*.res_blocks.*.conv2.conv.weight"] = WeightMapping(
        target_path="up_blocks.*.resnets.*.conv2.conv.kernel",
        transpose_axes=_CONV3D_TRANSPOSE,
        sharding=(None,),
    )
    mappings["vae.decoder.up_blocks.*.res_blocks.*.conv2.conv.bias"] = WeightMapping(
        target_path="up_blocks.*.resnets.*.conv2.conv.bias",
        sharding=(None,),
    )

    # Up blocks - upsample blocks (DepthToSpaceUpsample at indices 1,3,5)
    # Each has a single conv (CausalConv3d)
    mappings["vae.decoder.up_blocks.*.conv.conv.weight"] = WeightMapping(
        target_path="up_blocks.*.conv.conv.kernel",
        transpose_axes=_CONV3D_TRANSPOSE,
        sharding=(None,),
    )
    mappings["vae.decoder.up_blocks.*.conv.conv.bias"] = WeightMapping(
        target_path="up_blocks.*.conv.conv.bias",
        sharding=(None,),
    )

    # Output convolution
    mappings["vae.decoder.conv_out.conv.weight"] = WeightMapping(
        target_path="conv_out.conv.kernel",
        transpose_axes=_CONV3D_TRANSPOSE,
        sharding=(None,),
    )
    mappings["vae.decoder.conv_out.conv.bias"] = WeightMapping(
        target_path="conv_out.conv.bias",
        sharding=(None,),
    )

    return mappings
