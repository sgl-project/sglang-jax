"""Measure one V4 expert pair's load-time conversion in a fresh process.

With --weight-file, read a real safetensors pair. Otherwise generate packed
MXFP4 rows on demand without materializing the complete synthetic source.
Run this script in a fresh Python process for a meaningful RSS high-water delta.
"""

import argparse
import json
import resource
import sys

import numpy as np

from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    convert_mxfp4_pair_from_reader,
    convert_mxfp4_pair_from_safetensors,
)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-file")
    parser.add_argument("--scale-file")
    parser.add_argument("--weight-name", default="layers.0.ffn.experts.0.w1.weight")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0 or args.columns % 32:
        parser.error("synthetic rows must be positive and columns a positive multiple of 32")
    if not args.weight_name.endswith(".weight"):
        parser.error("--weight-name must end in .weight")
    scale_name = args.weight_name[: -len(".weight")] + ".scale"

    baseline = peak_rss_bytes()
    if args.weight_file:
        converted = convert_mxfp4_pair_from_safetensors(
            args.weight_file,
            args.weight_name,
            scale_name,
            scale_file=args.scale_file,
            row_chunk_size=args.row_chunk_size,
            strict=True,
        )
        source = args.weight_file
    else:
        converted = convert_mxfp4_pair_from_reader(
            weight_name=args.weight_name,
            scale_name=scale_name,
            weight_shape=(args.rows, args.columns // 2),
            scale_shape=(args.rows, args.columns // 32),
            weight_dtype="I8",
            scale_dtype="F8_E8M0",
            read_weight_rows=lambda part: np.full(
                (part.stop - part.start, args.columns // 2), 0x22, np.uint8
            ),
            read_scale_rows=lambda part: np.full(
                (part.stop - part.start, args.columns // 32), 127, np.uint8
            ),
            row_chunk_size=args.row_chunk_size,
            strict=True,
        )
        source = "synthetic constant MXFP4 rows"
    peak = peak_rss_bytes()
    print(
        json.dumps(
            {
                "source": source,
                "logical_shape": converted.logical_shape,
                "row_chunk_size": converted.report.row_chunk_size,
                "exact": converted.report.exact,
                "max_abs_error": converted.report.max_abs_error,
                "baseline_rss_high_water_bytes": baseline,
                "peak_rss_high_water_bytes": peak,
                "rss_high_water_delta_bytes": peak - baseline,
                "calculated_peak_converter_host_bytes": (
                    converted.report.calculated_peak_host_bytes
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
