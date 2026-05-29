from __future__ import annotations

from sgl_jax.srt.multimodal.configs.vaes.flux_vae_config import FluxVAEConfig
from sgl_jax.srt.utils.weight_utils import WeightMapping

TRANSPOSE_2D_CONV = (2, 3, 1, 0)


def _replicated(path: str, *, transpose: bool = False, transpose_axes=None):
    return WeightMapping(
        target_path=path,
        sharding=(),
        transpose=transpose,
        transpose_axes=transpose_axes,
    )


def to_mappings(config: FluxVAEConfig | None = None):
    mappings = {
        "encoder.conv_in.weight": _replicated(
            "encoder.conv_in.kernel", transpose_axes=TRANSPOSE_2D_CONV
        ),
        "encoder.conv_in.bias": _replicated("encoder.conv_in.bias"),
        "encoder.conv_norm_out.weight": _replicated("encoder.conv_norm_out.scale"),
        "encoder.conv_norm_out.bias": _replicated("encoder.conv_norm_out.bias"),
        "encoder.conv_out.weight": _replicated(
            "encoder.conv_out.kernel", transpose_axes=TRANSPOSE_2D_CONV
        ),
        "encoder.conv_out.bias": _replicated("encoder.conv_out.bias"),
        "decoder.conv_in.weight": _replicated(
            "decoder.conv_in.kernel", transpose_axes=TRANSPOSE_2D_CONV
        ),
        "decoder.conv_in.bias": _replicated("decoder.conv_in.bias"),
        "decoder.conv_norm_out.weight": _replicated("decoder.conv_norm_out.scale"),
        "decoder.conv_norm_out.bias": _replicated("decoder.conv_norm_out.bias"),
        "decoder.conv_out.weight": _replicated(
            "decoder.conv_out.kernel", transpose_axes=TRANSPOSE_2D_CONV
        ),
        "decoder.conv_out.bias": _replicated("decoder.conv_out.bias"),
    }

    for prefix in ("encoder", "decoder"):
        mappings.update(
            {
                f"{prefix}.mid_block.resnets.*.conv1.weight": _replicated(
                    f"{prefix}.mid_block.resnets.*.conv1.kernel",
                    transpose_axes=TRANSPOSE_2D_CONV,
                ),
                f"{prefix}.mid_block.resnets.*.conv1.bias": _replicated(
                    f"{prefix}.mid_block.resnets.*.conv1.bias"
                ),
                f"{prefix}.mid_block.resnets.*.conv2.weight": _replicated(
                    f"{prefix}.mid_block.resnets.*.conv2.kernel",
                    transpose_axes=TRANSPOSE_2D_CONV,
                ),
                f"{prefix}.mid_block.resnets.*.conv2.bias": _replicated(
                    f"{prefix}.mid_block.resnets.*.conv2.bias"
                ),
                f"{prefix}.mid_block.resnets.*.norm1.weight": _replicated(
                    f"{prefix}.mid_block.resnets.*.norm1.scale"
                ),
                f"{prefix}.mid_block.resnets.*.norm1.bias": _replicated(
                    f"{prefix}.mid_block.resnets.*.norm1.bias"
                ),
                f"{prefix}.mid_block.resnets.*.norm2.weight": _replicated(
                    f"{prefix}.mid_block.resnets.*.norm2.scale"
                ),
                f"{prefix}.mid_block.resnets.*.norm2.bias": _replicated(
                    f"{prefix}.mid_block.resnets.*.norm2.bias"
                ),
                f"{prefix}.mid_block.attentions.*.group_norm.weight": _replicated(
                    f"{prefix}.mid_block.attentions.*.group_norm.scale"
                ),
                f"{prefix}.mid_block.attentions.*.group_norm.bias": _replicated(
                    f"{prefix}.mid_block.attentions.*.group_norm.bias"
                ),
                f"{prefix}.mid_block.attentions.*.to_q.weight": _replicated(
                    f"{prefix}.mid_block.attentions.*.query.kernel",
                    transpose=True,
                ),
                f"{prefix}.mid_block.attentions.*.to_q.bias": _replicated(
                    f"{prefix}.mid_block.attentions.*.query.bias"
                ),
                f"{prefix}.mid_block.attentions.*.to_k.weight": _replicated(
                    f"{prefix}.mid_block.attentions.*.key.kernel",
                    transpose=True,
                ),
                f"{prefix}.mid_block.attentions.*.to_k.bias": _replicated(
                    f"{prefix}.mid_block.attentions.*.key.bias"
                ),
                f"{prefix}.mid_block.attentions.*.to_v.weight": _replicated(
                    f"{prefix}.mid_block.attentions.*.value.kernel",
                    transpose=True,
                ),
                f"{prefix}.mid_block.attentions.*.to_v.bias": _replicated(
                    f"{prefix}.mid_block.attentions.*.value.bias"
                ),
                f"{prefix}.mid_block.attentions.*.to_out.0.weight": _replicated(
                    f"{prefix}.mid_block.attentions.*.proj_attn.kernel",
                    transpose=True,
                ),
                f"{prefix}.mid_block.attentions.*.to_out.0.bias": _replicated(
                    f"{prefix}.mid_block.attentions.*.proj_attn.bias"
                ),
            }
        )

    mappings.update(
        {
            "encoder.down_blocks.*.resnets.*.conv1.weight": _replicated(
                "encoder.down_blocks.*.resnets.*.conv1.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "encoder.down_blocks.*.resnets.*.conv1.bias": _replicated(
                "encoder.down_blocks.*.resnets.*.conv1.bias"
            ),
            "encoder.down_blocks.*.resnets.*.conv2.weight": _replicated(
                "encoder.down_blocks.*.resnets.*.conv2.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "encoder.down_blocks.*.resnets.*.conv2.bias": _replicated(
                "encoder.down_blocks.*.resnets.*.conv2.bias"
            ),
            "encoder.down_blocks.*.resnets.*.conv_shortcut.weight": _replicated(
                "encoder.down_blocks.*.resnets.*.conv_shortcut.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "encoder.down_blocks.*.resnets.*.conv_shortcut.bias": _replicated(
                "encoder.down_blocks.*.resnets.*.conv_shortcut.bias"
            ),
            "encoder.down_blocks.*.resnets.*.norm1.weight": _replicated(
                "encoder.down_blocks.*.resnets.*.norm1.scale"
            ),
            "encoder.down_blocks.*.resnets.*.norm1.bias": _replicated(
                "encoder.down_blocks.*.resnets.*.norm1.bias"
            ),
            "encoder.down_blocks.*.resnets.*.norm2.weight": _replicated(
                "encoder.down_blocks.*.resnets.*.norm2.scale"
            ),
            "encoder.down_blocks.*.resnets.*.norm2.bias": _replicated(
                "encoder.down_blocks.*.resnets.*.norm2.bias"
            ),
            "encoder.down_blocks.*.downsamplers.*.conv.weight": _replicated(
                "encoder.down_blocks.*.downsamplers.*.conv.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "encoder.down_blocks.*.downsamplers.*.conv.bias": _replicated(
                "encoder.down_blocks.*.downsamplers.*.conv.bias"
            ),
            "decoder.up_blocks.*.resnets.*.conv1.weight": _replicated(
                "decoder.up_blocks.*.resnets.*.conv1.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "decoder.up_blocks.*.resnets.*.conv1.bias": _replicated(
                "decoder.up_blocks.*.resnets.*.conv1.bias"
            ),
            "decoder.up_blocks.*.resnets.*.conv2.weight": _replicated(
                "decoder.up_blocks.*.resnets.*.conv2.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "decoder.up_blocks.*.resnets.*.conv2.bias": _replicated(
                "decoder.up_blocks.*.resnets.*.conv2.bias"
            ),
            "decoder.up_blocks.*.resnets.*.conv_shortcut.weight": _replicated(
                "decoder.up_blocks.*.resnets.*.conv_shortcut.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "decoder.up_blocks.*.resnets.*.conv_shortcut.bias": _replicated(
                "decoder.up_blocks.*.resnets.*.conv_shortcut.bias"
            ),
            "decoder.up_blocks.*.resnets.*.norm1.weight": _replicated(
                "decoder.up_blocks.*.resnets.*.norm1.scale"
            ),
            "decoder.up_blocks.*.resnets.*.norm1.bias": _replicated(
                "decoder.up_blocks.*.resnets.*.norm1.bias"
            ),
            "decoder.up_blocks.*.resnets.*.norm2.weight": _replicated(
                "decoder.up_blocks.*.resnets.*.norm2.scale"
            ),
            "decoder.up_blocks.*.resnets.*.norm2.bias": _replicated(
                "decoder.up_blocks.*.resnets.*.norm2.bias"
            ),
            "decoder.up_blocks.*.upsamplers.*.conv.weight": _replicated(
                "decoder.up_blocks.*.upsamplers.*.conv.kernel",
                transpose_axes=TRANSPOSE_2D_CONV,
            ),
            "decoder.up_blocks.*.upsamplers.*.conv.bias": _replicated(
                "decoder.up_blocks.*.upsamplers.*.conv.bias"
            ),
        }
    )

    if config is not None and config.use_quant_conv:
        mappings.update(
            {
                "quant_conv.weight": _replicated(
                    "quant_conv.kernel", transpose_axes=TRANSPOSE_2D_CONV
                ),
                "quant_conv.bias": _replicated("quant_conv.bias"),
            }
        )

    if config is not None and config.use_post_quant_conv:
        mappings.update(
            {
                "post_quant_conv.weight": _replicated(
                    "post_quant_conv.kernel", transpose_axes=TRANSPOSE_2D_CONV
                ),
                "post_quant_conv.bias": _replicated("post_quant_conv.bias"),
            }
        )

    return mappings
