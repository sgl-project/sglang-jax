import argparse
import dataclasses
from dataclasses import field

from sgl_jax.srt.server_args import ServerArgs


@dataclasses.dataclass
class MultimodalServerArgs(ServerArgs):
    embedded_cfg_scale: float = 6.0
    flow_shift: float | None = None

    dit_precision: str | None = None

    vae_precision: str = "bf16"
    vae_tiling: bool = True
    vae_sp: bool = True
    DEFAULT_TEXT_ENCODER_PRECISIONS = ("fp32",)
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))
    image_encoder_precision: str = "bf16"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        prefix_with_dot = ""
        parser.add_argument(
            f"--{prefix_with_dot}embedded-cfg-scale",
            type=float,
            dest=f"{prefix_with_dot.replace('-', '_')}embedded_cfg_scale",
            default=MultimodalServerArgs.embedded_cfg_scale,
            help="Embedded CFG scale",
        )
        parser.add_argument(
            f"--{prefix_with_dot}flow-shift",
            type=float,
            dest=f"{prefix_with_dot.replace('-', '_')}flow_shift",
            default=MultimodalServerArgs.flow_shift,
            help="Flow shift parameter",
        )

        # DiT configuration
        parser.add_argument(
            f"--{prefix_with_dot}dit-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}dit_precision",
            default=MultimodalServerArgs.dit_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for the DiT model",
        )

        # VAE configuration
        parser.add_argument(
            f"--{prefix_with_dot}vae-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}vae_precision",
            default=MultimodalServerArgs.vae_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for VAE",
        )
        parser.add_argument(
            f"--{prefix_with_dot}vae-tiling",
            action="store_true",
            dest=f"{prefix_with_dot.replace('-', '_')}vae_tiling",
            default=MultimodalServerArgs.vae_tiling,
            help="Enable VAE tiling",
        )
        parser.add_argument(
            f"--{prefix_with_dot}vae-sp",
            action="store_true",
            dest=f"{prefix_with_dot.replace('-', '_')}vae_sp",
            help="Enable VAE spatial parallelism",
        )

        # Text encoder configuration
        parser.add_argument(
            f"--{prefix_with_dot}text-encoder-precisions",
            nargs="+",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}text_encoder_precisions",
            default=MultimodalServerArgs.DEFAULT_TEXT_ENCODER_PRECISIONS,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for each text encoder",
        )

        # Image encoder configuration
        parser.add_argument(
            f"--{prefix_with_dot}image-encoder-precision",
            type=str,
            dest=f"{prefix_with_dot.replace('-', '_')}image_encoder_precision",
            default=MultimodalServerArgs.image_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for image encoder",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})
