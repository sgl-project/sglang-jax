import enum

TransposeAndSharding = tuple[tuple[int, ...] | None, tuple[str | None, ...]]
MappingEntry = tuple[str, TransposeAndSharding]


class TRANSPOSE(enum.Enum):
    TRANSPOSE_2D_CONV = (2, 3, 1, 0)
    TRANSPOSE_3D_CONV = (2, 3, 4, 1, 0)
    TRANSPOSE_2D_SCALE = (1, 2, 0)
    TRANSPOSE_3D_SCALE = (1, 2, 3, 0)


def to_mappings() -> dict[str, MappingEntry]:
    return {
        "decoder.conv_in.weight": (
            "decoder.conv_in.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.conv_in.bias": ("decoder.conv_in.conv.bias", (None, (None, "model"))),
        "decoder.conv_out.weight": (
            "decoder.conv_out.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.conv_out.bias": ("decoder.conv_out.conv.bias", (None, (None, "model"))),
        "decoder.mid_block.attentions.*.norm.gamma": (
            "decoder.mid_block.attentions.*.norm.scale",
            (TRANSPOSE.TRANSPOSE_2D_SCALE.value, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.proj.bias": (
            "decoder.mid_block.attentions.*.proj.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.proj.weight": (
            "decoder.mid_block.attentions.*.proj.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.to_qkv.bias": (
            "decoder.mid_block.attentions.*.qkv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.to_qkv.weight": (
            "decoder.mid_block.attentions.*.qkv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv1.bias": (
            "decoder.mid_block.resnets.*.conv1.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv1.weight": (
            "decoder.mid_block.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv2.bias": (
            "decoder.mid_block.resnets.*.conv2.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv2.weight": (
            "decoder.mid_block.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.norm1.gamma": (
            "decoder.mid_block.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.norm2.gamma": (
            "decoder.mid_block.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.norm_out.gamma": (
            "decoder.norm_out.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv1.bias": (
            "decoder.up_blocks.*.resnets.*.conv1.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.resnets.*.conv1.weight": (
            "decoder.up_blocks.*.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv2.bias": (
            "decoder.up_blocks.*.resnets.*.conv2.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.resnets.*.conv2.weight": (
            "decoder.up_blocks.*.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.norm1.gamma": (
            "decoder.up_blocks.*.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.norm2.gamma": (
            "decoder.up_blocks.*.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.upsamplers.*.resample.1.bias": (
            "decoder.up_blocks.*.upsamplers.*.spatial_conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.upsamplers.*.resample.1.weight": (
            "decoder.up_blocks.*.upsamplers.*.spatial_conv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.bias": (
            "decoder.up_blocks.*.upsamplers.*.time_conv.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.weight": (
            "decoder.up_blocks.*.upsamplers.*.time_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv_shortcut.bias": (
            "decoder.up_blocks.*.resnets.*.skip_conv.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv_shortcut.weight": (
            "decoder.up_blocks.*.resnets.*.skip_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_in.weight": (
            "encoder.conv_in.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_in.bias": ("encoder.conv_in.conv.bias", (None, (None, "model"))),
        "encoder.conv_out.weight": (
            "encoder.conv_out.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_out.bias": ("encoder.conv_out.conv.bias", (None, (None, "model"))),
        "encoder.down_blocks.*.conv1.bias": (
            "encoder.down_blocks.*.conv1.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.down_blocks.*.conv1.weight": (
            "encoder.down_blocks.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.conv2.bias": (
            "encoder.down_blocks.*.conv2.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.down_blocks.*.conv2.weight": (
            "encoder.down_blocks.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.norm1.gamma": (
            "encoder.down_blocks.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.down_blocks.*.norm2.gamma": (
            "encoder.down_blocks.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.down_blocks.*.resample.1.bias": (
            "encoder.down_blocks.*.spatial_conv.bias",
            (None, "model"),
        ),
        "encoder.down_blocks.*.resample.1.weight": (
            "encoder.down_blocks.*.spatial_conv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.time_conv.bias": (
            "encoder.down_blocks.*.time_conv.conv.bias",
            (None, "model"),
        ),
        "encoder.down_blocks.*.time_conv.weight": (
            "encoder.down_blocks.*.time_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.conv_shortcut.bias": (
            "encoder.down_blocks.*.skip_conv.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.down_blocks.*.conv_shortcut.weight": (
            "encoder.down_blocks.*.skip_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.norm.gamma": (
            "encoder.mid_block.attentions.*.norm.scale",
            (TRANSPOSE.TRANSPOSE_2D_SCALE.value, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.proj.bias": (
            "encoder.mid_block.attentions.*.proj.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.proj.weight": (
            "encoder.mid_block.attentions.*.proj.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.to_qkv.bias": (
            "encoder.mid_block.attentions.*.qkv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.to_qkv.weight": (
            "encoder.mid_block.attentions.*.qkv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv1.bias": (
            "encoder.mid_block.resnets.*.conv1.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv1.weight": (
            "encoder.mid_block.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv2.bias": (
            "encoder.mid_block.resnets.*.conv2.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv2.weight": (
            "encoder.mid_block.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.norm1.gamma": (
            "encoder.mid_block.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.norm2.gamma": (
            "encoder.mid_block.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.norm_out.gamma": (
            "encoder.norm_out.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        # not used in encoder and decoder
        "post_quant_conv.bias": (
            "post_quant_conv.conv.bias",
            (None, (None, "model")),
        ),
        "post_quant_conv.weight": (
            "post_quant_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "quant_conv.bias": (
            "quant_conv.conv.bias",
            (None, (None, "model")),
        ),
        "quant_conv.weight": (
            "quant_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
    }
