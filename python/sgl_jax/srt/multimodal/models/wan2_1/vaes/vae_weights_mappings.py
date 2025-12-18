from sgl_jax.srt.utils.weight_utils import WeightMapping

# Multi-dimensional transpose axes for different weight types
TRANSPOSE_2D_CONV = (2, 3, 1, 0)  # PyTorch OIHW -> JAX HWIO
TRANSPOSE_3D_CONV = (2, 3, 4, 1, 0)  # PyTorch OIDHW -> JAX DHWIO
TRANSPOSE_2D_SCALE = (1, 2, 0)  # (1, 1, C) -> different layout
TRANSPOSE_3D_SCALE = (1, 2, 3, 0)  # (1, 1, 1, C) -> different layout


def to_mappings() -> dict[str, WeightMapping]:
    return {
        # Decoder conv_in/conv_out
        "decoder.conv_in.weight": WeightMapping(
            target_path="decoder.conv_in.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.conv_in.bias": WeightMapping(
            target_path="decoder.conv_in.conv.bias",
            sharding=(None,),
        ),
        "decoder.conv_out.weight": WeightMapping(
            target_path="decoder.conv_out.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.conv_out.bias": WeightMapping(
            target_path="decoder.conv_out.conv.bias",
            sharding=(None,),
        ),
        # Decoder mid_block attentions
        "decoder.mid_block.attentions.*.norm.gamma": WeightMapping(
            target_path="decoder.mid_block.attentions.*.norm.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_2D_SCALE,
        ),
        "decoder.mid_block.attentions.*.proj.bias": WeightMapping(
            target_path="decoder.mid_block.attentions.*.proj.bias",
            sharding=(None,),
        ),
        "decoder.mid_block.attentions.*.proj.weight": WeightMapping(
            target_path="decoder.mid_block.attentions.*.proj.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        "decoder.mid_block.attentions.*.to_qkv.bias": WeightMapping(
            target_path="decoder.mid_block.attentions.*.qkv.bias",
            sharding=(None,),
        ),
        "decoder.mid_block.attentions.*.to_qkv.weight": WeightMapping(
            target_path="decoder.mid_block.attentions.*.qkv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        # Decoder mid_block resnets
        "decoder.mid_block.resnets.*.conv1.bias": WeightMapping(
            target_path="decoder.mid_block.resnets.*.conv1.conv.bias",
            sharding=(None,),
        ),
        "decoder.mid_block.resnets.*.conv1.weight": WeightMapping(
            target_path="decoder.mid_block.resnets.*.conv1.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.mid_block.resnets.*.conv2.bias": WeightMapping(
            target_path="decoder.mid_block.resnets.*.conv2.conv.bias",
            sharding=(None,),
        ),
        "decoder.mid_block.resnets.*.conv2.weight": WeightMapping(
            target_path="decoder.mid_block.resnets.*.conv2.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.mid_block.resnets.*.norm1.gamma": WeightMapping(
            target_path="decoder.mid_block.resnets.*.norm1.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        "decoder.mid_block.resnets.*.norm2.gamma": WeightMapping(
            target_path="decoder.mid_block.resnets.*.norm2.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        # Decoder norm_out
        "decoder.norm_out.gamma": WeightMapping(
            target_path="decoder.norm_out.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        # Decoder up_blocks resnets
        "decoder.up_blocks.*.resnets.*.conv1.bias": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.conv1.conv.bias",
            sharding=(None,),
        ),
        "decoder.up_blocks.*.resnets.*.conv1.weight": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.conv1.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.up_blocks.*.resnets.*.conv2.bias": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.conv2.conv.bias",
            sharding=(None,),
        ),
        "decoder.up_blocks.*.resnets.*.conv2.weight": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.conv2.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "decoder.up_blocks.*.resnets.*.norm1.gamma": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.norm1.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        "decoder.up_blocks.*.resnets.*.norm2.gamma": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.norm2.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        # Decoder up_blocks upsamplers
        "decoder.up_blocks.*.upsamplers.*.resample.1.bias": WeightMapping(
            target_path="decoder.up_blocks.*.upsamplers.*.spatial_conv.bias",
            sharding=(None,),
        ),
        "decoder.up_blocks.*.upsamplers.*.resample.1.weight": WeightMapping(
            target_path="decoder.up_blocks.*.upsamplers.*.spatial_conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.bias": WeightMapping(
            target_path="decoder.up_blocks.*.upsamplers.*.time_conv.conv.bias",
            sharding=(None,),
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.weight": WeightMapping(
            target_path="decoder.up_blocks.*.upsamplers.*.time_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        # Decoder up_blocks resnets skip_conv
        "decoder.up_blocks.*.resnets.*.conv_shortcut.bias": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.skip_conv.conv.bias",
            sharding=(None,),
        ),
        "decoder.up_blocks.*.resnets.*.conv_shortcut.weight": WeightMapping(
            target_path="decoder.up_blocks.*.resnets.*.skip_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        # Encoder conv_in/conv_out
        "encoder.conv_in.weight": WeightMapping(
            target_path="encoder.conv_in.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.conv_in.bias": WeightMapping(
            target_path="encoder.conv_in.conv.bias",
            sharding=(None,),
        ),
        "encoder.conv_out.weight": WeightMapping(
            target_path="encoder.conv_out.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.conv_out.bias": WeightMapping(
            target_path="encoder.conv_out.conv.bias",
            sharding=(None,),
        ),
        # Encoder down_blocks
        "encoder.down_blocks.*.conv1.bias": WeightMapping(
            target_path="encoder.down_blocks.*.conv1.conv.bias",
            sharding=(None,),
        ),
        "encoder.down_blocks.*.conv1.weight": WeightMapping(
            target_path="encoder.down_blocks.*.conv1.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.down_blocks.*.conv2.bias": WeightMapping(
            target_path="encoder.down_blocks.*.conv2.conv.bias",
            sharding=(None,),
        ),
        "encoder.down_blocks.*.conv2.weight": WeightMapping(
            target_path="encoder.down_blocks.*.conv2.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.down_blocks.*.norm1.gamma": WeightMapping(
            target_path="encoder.down_blocks.*.norm1.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        "encoder.down_blocks.*.norm2.gamma": WeightMapping(
            target_path="encoder.down_blocks.*.norm2.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        "encoder.down_blocks.*.resample.1.bias": WeightMapping(
            target_path="encoder.down_blocks.*.spatial_conv.bias",
            sharding=(None,),
        ),
        "encoder.down_blocks.*.resample.1.weight": WeightMapping(
            target_path="encoder.down_blocks.*.spatial_conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        "encoder.down_blocks.*.time_conv.bias": WeightMapping(
            target_path="encoder.down_blocks.*.time_conv.conv.bias",
            sharding=(None,),
        ),
        "encoder.down_blocks.*.time_conv.weight": WeightMapping(
            target_path="encoder.down_blocks.*.time_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.down_blocks.*.conv_shortcut.bias": WeightMapping(
            target_path="encoder.down_blocks.*.skip_conv.conv.bias",
            sharding=(None,),
        ),
        "encoder.down_blocks.*.conv_shortcut.weight": WeightMapping(
            target_path="encoder.down_blocks.*.skip_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        # Encoder mid_block attentions
        "encoder.mid_block.attentions.*.norm.gamma": WeightMapping(
            target_path="encoder.mid_block.attentions.*.norm.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_2D_SCALE,
        ),
        "encoder.mid_block.attentions.*.proj.bias": WeightMapping(
            target_path="encoder.mid_block.attentions.*.proj.bias",
            sharding=(None,),
        ),
        "encoder.mid_block.attentions.*.proj.weight": WeightMapping(
            target_path="encoder.mid_block.attentions.*.proj.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        "encoder.mid_block.attentions.*.to_qkv.bias": WeightMapping(
            target_path="encoder.mid_block.attentions.*.qkv.bias",
            sharding=(None,),
        ),
        "encoder.mid_block.attentions.*.to_qkv.weight": WeightMapping(
            target_path="encoder.mid_block.attentions.*.qkv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_2D_CONV,
        ),
        # Encoder mid_block resnets
        "encoder.mid_block.resnets.*.conv1.bias": WeightMapping(
            target_path="encoder.mid_block.resnets.*.conv1.conv.bias",
            sharding=(None,),
        ),
        "encoder.mid_block.resnets.*.conv1.weight": WeightMapping(
            target_path="encoder.mid_block.resnets.*.conv1.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.mid_block.resnets.*.conv2.bias": WeightMapping(
            target_path="encoder.mid_block.resnets.*.conv2.conv.bias",
            sharding=(None,),
        ),
        "encoder.mid_block.resnets.*.conv2.weight": WeightMapping(
            target_path="encoder.mid_block.resnets.*.conv2.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "encoder.mid_block.resnets.*.norm1.gamma": WeightMapping(
            target_path="encoder.mid_block.resnets.*.norm1.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        "encoder.mid_block.resnets.*.norm2.gamma": WeightMapping(
            target_path="encoder.mid_block.resnets.*.norm2.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        # Encoder norm_out
        "encoder.norm_out.gamma": WeightMapping(
            target_path="encoder.norm_out.scale",
            sharding=(None,),
            transpose_axes=TRANSPOSE_3D_SCALE,
        ),
        # Quant conv (not used in encoder and decoder)
        "post_quant_conv.bias": WeightMapping(
            target_path="post_quant_conv.conv.bias",
            sharding=(None,),
        ),
        "post_quant_conv.weight": WeightMapping(
            target_path="post_quant_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
        "quant_conv.bias": WeightMapping(
            target_path="quant_conv.conv.bias",
            sharding=(None,),
        ),
        "quant_conv.weight": WeightMapping(
            target_path="quant_conv.conv.kernel",
            sharding=(None, None),
            transpose_axes=TRANSPOSE_3D_CONV,
        ),
    }
