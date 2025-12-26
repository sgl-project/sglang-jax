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
            "conv_in.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.conv_in.bias": ("conv_in.conv.bias", (None, (None, "model"))),
        "decoder.conv_out.weight": (
            "conv_out.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.conv_out.bias": ("conv_out.conv.bias", (None, (None, "model"))),
        "decoder.mid_block.attentions.*.norm.gamma": (
            "mid_block.attentions.*.norm.scale",
            (TRANSPOSE.TRANSPOSE_2D_SCALE.value, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.proj.bias": (
            "mid_block.attentions.*.proj.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.proj.weight": (
            "mid_block.attentions.*.proj.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.to_qkv.bias": (
            "mid_block.attentions.*.qkv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.attentions.*.to_qkv.weight": (
            "mid_block.attentions.*.qkv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv1.bias": (
            "mid_block.resnets.*.conv1.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv1.weight": (
            "mid_block.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv2.bias": (
            "mid_block.resnets.*.conv2.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.conv2.weight": (
            "mid_block.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.norm1.gamma": (
            "mid_block.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.mid_block.resnets.*.norm2.gamma": (
            "mid_block.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.norm_out.gamma": (
            "norm_out.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv1.bias": (
            "up_blocks.*.resnets.*.conv1.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.resnets.*.conv1.weight": (
            "up_blocks.*.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv2.bias": (
            "up_blocks.*.resnets.*.conv2.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.resnets.*.conv2.weight": (
            "up_blocks.*.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.norm1.gamma": (
            "up_blocks.*.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.norm2.gamma": (
            "up_blocks.*.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "decoder.up_blocks.*.upsamplers.*.resample.1.bias": (
            "up_blocks.*.upsamplers.*.spatial_conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.upsamplers.*.resample.1.weight": (
            "up_blocks.*.upsamplers.*.spatial_conv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.bias": (
            "up_blocks.*.upsamplers.*.time_conv.conv.bias",
            (None, "model"),
        ),
        "decoder.up_blocks.*.upsamplers.*.time_conv.weight": (
            "up_blocks.*.upsamplers.*.time_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv_shortcut.bias": (
            "up_blocks.*.resnets.*.skip_conv.conv.bias",
            (None, (None, "model")),
        ),
        "decoder.up_blocks.*.resnets.*.conv_shortcut.weight": (
            "up_blocks.*.resnets.*.skip_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_in.weight": (
            "conv_in.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_in.bias": ("conv_in.conv.bias", (None, (None, "model"))),
        "encoder.conv_out.weight": (
            "conv_out.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.conv_out.bias": ("conv_out.conv.bias", (None, (None, "model"))),

        "encoder.down_blocks.*.conv1.bias": ("down_blocks.*.conv1.conv.bias", (None, (None, "model"))),
        "encoder.down_blocks.*.conv1.weight": ("down_blocks.*.conv1.conv.kernel", (None, (TRANSPOSE.TRANSPOSE_3D_CONV.value, "model"))),
        "encoder.down_blocks.*.conv2.bias": ("down_blocks.*.conv2.conv.bias", (None, (None, "model"))),
        "encoder.down_blocks.*.conv2.weight": ("down_blocks.*.conv2.conv.kernel", (None, (TRANSPOSE.TRANSPOSE_3D_CONV.value, "model"))),
        "encoder.down_blocks.*.norm1.gamma": ("down_blocks.*.norm1.scale", (None, (TRANSPOSE.TRANSPOSE_3D_SCALE.value, "model"))),
        "encoder.down_blocks.*.norm2.gamma": ("down_blocks.*.norm2.scale", (None, (TRANSPOSE.TRANSPOSE_3D_SCALE.value, "model"))),
        "encoder.down_blocks.*.resample.1.bias": (
            "down_blocks.*.spatial_conv.bias",
            (None, "model"),
        ),
        "encoder.down_blocks.*.resample.1.weight": (
            "down_blocks.*.spatial_conv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.time_conv.bias": (
            "down_blocks.*.time_conv.conv.bias",
            (None, "model"),
        ),
        "encoder.down_blocks.*.time_conv.weight": (
            "down_blocks.*.time_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.down_blocks.*.conv_shortcut.bias": (
            "down_blocks.*.skip_conv.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.down_blocks.*.conv_shortcut.weight": (
            "down_blocks.*.skip_conv.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),

        "encoder.mid_block.attentions.*.norm.gamma": (
            "mid_block.attentions.*.norm.scale",
            (TRANSPOSE.TRANSPOSE_2D_SCALE.value, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.proj.bias": (
            "mid_block.attentions.*.proj.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.proj.weight": (
            "mid_block.attentions.*.proj.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.to_qkv.bias": (
            "mid_block.attentions.*.qkv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.attentions.*.to_qkv.weight": (
            "mid_block.attentions.*.qkv.kernel",
            (TRANSPOSE.TRANSPOSE_2D_CONV.value, (None, "model")),
        ),

        "encoder.mid_block.resnets.*.conv1.bias": (
            "mid_block.resnets.*.conv1.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv1.weight": (
            "mid_block.resnets.*.conv1.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv2.bias": (
            "mid_block.resnets.*.conv2.conv.bias",
            (None, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.conv2.weight": (
            "mid_block.resnets.*.conv2.conv.kernel",
            (TRANSPOSE.TRANSPOSE_3D_CONV.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.norm1.gamma": (
            "mid_block.resnets.*.norm1.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.mid_block.resnets.*.norm2.gamma": (
            "mid_block.resnets.*.norm2.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
        "encoder.norm_out.gamma": (
            "norm_out.scale",
            (TRANSPOSE.TRANSPOSE_3D_SCALE.value, (None, "model")),
        ),
    }
