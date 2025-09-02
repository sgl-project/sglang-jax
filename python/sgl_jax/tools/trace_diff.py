#!/usr/bin/env python3
"""
JSONL Trace Comparison Tool

Compares two JSONL trace files by content_hash and displays differences
with color-coded output similar to difflib.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Set, Tuple


class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file and return list of JSON objects"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def group_by_content_hash(traces: List[Dict]) -> Dict[str, List[Dict]]:
    """Group traces by content_hash"""
    groups = {}
    for trace in traces:
        content_hash = trace.get("content_hash")
        if content_hash:
            if content_hash not in groups:
                groups[content_hash] = []
            groups[content_hash].append(trace)
    return groups


def compare_precision_records(
    records1: List[Dict], records2: List[Dict], tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """Compare precision records between two traces with detailed hierarchical output"""
    differences = []

    if len(records1) != len(records2):
        differences.append(
            f"❌ Record count mismatch: {len(records1)} vs {len(records2)}"
        )
        return False, differences

    all_match = True

    # Group records by layer_id and module_type for better organization
    layer_groups1 = {}
    layer_groups2 = {}

    for rec in records1:
        layer_id = rec.get("layer_id", "unknown")
        module_type = rec.get("module_type", "unknown")
        key = f"{layer_id}_{module_type}"
        if key not in layer_groups1:
            layer_groups1[key] = []
        layer_groups1[key].append(rec)

    for rec in records2:
        layer_id = rec.get("layer_id", "unknown")
        module_type = rec.get("module_type", "unknown")
        key = f"{layer_id}_{module_type}"
        if key not in layer_groups2:
            layer_groups2[key] = []
        layer_groups2[key].append(rec)

    # Compare by groups
    all_keys = set(layer_groups1.keys()) | set(layer_groups2.keys())

    for key in sorted(all_keys):
        layer_id, module_type = key.split("_", 1)
        layer_display = (
            f"Layer {layer_id}" if layer_id != "unknown" else "Unknown Layer"
        )
        module_display = (
            module_type.title() if module_type != "unknown" else "Unknown Module"
        )

        group1 = layer_groups1.get(key, [])
        group2 = layer_groups2.get(key, [])

        if len(group1) != len(group2):
            differences.append(
                f"❌ {layer_display} | {module_display} | Record count: {len(group1)} vs {len(group2)}"
            )
            all_match = False
            continue

        # Compare records within the group
        for i, (rec1, rec2) in enumerate(zip(group1, group2)):
            record_prefix = f"{layer_display} | {module_type.title()} | {rec1.get('name', 'unnamed')}"

            # Compare key fields
            for field in ["stage", "name", "shape", "dtype"]:
                val1, val2 = rec1.get(field), rec2.get(field)
                if val1 != val2:
                    differences.append(
                        f"❌ {record_prefix} | {field}: {val1} vs {val2}"
                    )
                    all_match = False

            # Compare numerical values with tolerance
            for field in ["min", "max", "mean", "std"]:
                val1, val2 = rec1.get(field), rec2.get(field)
                if val1 is not None and val2 is not None:
                    if isinstance(val1, (int, float)) and isinstance(
                        val2, (int, float)
                    ):
                        diff = abs(val1 - val2)
                        if diff > tolerance:
                            differences.append(
                                f"❌ {record_prefix} | {field}: {val1:.6f} vs {val2:.6f} (diff: {diff:.6f})"
                            )
                            all_match = False
                    elif val1 != val2:
                        differences.append(
                            f"❌ {record_prefix} | {field}: {val1} vs {val2}"
                        )
                        all_match = False

            # Compare boolean flags
            for field in ["has_nan", "has_inf"]:
                val1, val2 = rec1.get(field), rec2.get(field)
                if val1 != val2:
                    differences.append(
                        f"❌ {record_prefix} | {field}: {val1} vs {val2}"
                    )
                    all_match = False

            # Compare token-level statistics if available
            token_stats1 = rec1.get("token_stats", [])
            token_stats2 = rec2.get("token_stats", [])

            if token_stats1 and token_stats2:
                if len(token_stats1) != len(token_stats2):
                    differences.append(
                        f"❌ {record_prefix} | token_stats count: {len(token_stats1)} vs {len(token_stats2)}"
                    )
                    all_match = False
                else:
                    # Compare token-level stats
                    for token_stat1, token_stat2 in zip(token_stats1, token_stats2):
                        token_idx = token_stat1.get("token_idx", "unknown")
                        token_prefix = f"{record_prefix} | Token[{token_idx}]"

                        # Compare token-level numerical values
                        for token_field in ["min", "max", "mean", "std", "value"]:
                            tval1, tval2 = token_stat1.get(
                                token_field
                            ), token_stat2.get(token_field)
                            if tval1 is not None and tval2 is not None:
                                if isinstance(tval1, (int, float)) and isinstance(
                                    tval2, (int, float)
                                ):
                                    diff = abs(tval1 - tval2)
                                    if diff > tolerance:
                                        differences.append(
                                            f"❌ {token_prefix} | {token_field}: {tval1:.6f} vs {tval2:.6f} (diff: {diff:.6f})"
                                        )
                                        all_match = False
                                elif tval1 != tval2:
                                    differences.append(
                                        f"❌ {token_prefix} | {token_field}: {tval1} vs {tval2}"
                                    )
                                    all_match = False

                        # Compare token-level boolean flags
                        for token_field in ["has_nan", "has_inf"]:
                            tval1, tval2 = token_stat1.get(
                                token_field
                            ), token_stat2.get(token_field)
                            if tval1 != tval2:
                                differences.append(
                                    f"❌ {token_prefix} | {token_field}: {tval1} vs {tval2}"
                                )
                                all_match = False
            elif token_stats1 or token_stats2:
                # One has token stats, the other doesn't
                differences.append(
                    f"❌ {record_prefix} | token_stats availability: {bool(token_stats1)} vs {bool(token_stats2)}"
                )
                all_match = False

            # Add tensor statistics summary for context
            if not all_match and i == 0:  # Only for first record in group to avoid spam
                shape1 = rec1.get("shape", "unknown")
                shape2 = rec2.get("shape", "unknown")
                seq_len1 = rec1.get("sequence_length")
                seq_len2 = rec2.get("sequence_length")

                if shape1 == shape2:
                    shape_info = f"📊 Tensor shape: {shape1}"
                    if seq_len1 is not None:
                        shape_info += f" (seq_len: {seq_len1})"
                    differences.append(f"    {shape_info}")
                else:
                    differences.append(
                        f"    📊 Tensor shapes differ: {shape1} vs {shape2}"
                    )
                    if seq_len1 is not None and seq_len2 is not None:
                        differences.append(
                            f"    📊 Sequence lengths: {seq_len1} vs {seq_len2}"
                        )

    return all_match, differences


def print_diff_header(content_hash: str, trace1: Dict, trace2: Dict):
    """Print a header for the diff section"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Content Hash: {content_hash}{Colors.RESET}")
    print(
        f"{Colors.BOLD}Request 1:{Colors.RESET} ID={trace1.get('request_id', 'N/A')}, "
        f"Duration={trace1.get('duration', 'N/A'):.3f}s"
    )
    print(
        f"{Colors.BOLD}Request 2:{Colors.RESET} ID={trace2.get('request_id', 'N/A')}, "
        f"Duration={trace2.get('duration', 'N/A'):.3f}s"
    )
    print("=" * 80)


def print_match_status(is_match: bool, differences: List[str]):
    """Print match status with colors and detailed hierarchy"""
    if is_match:
        print(f"{Colors.GREEN}✓ MATCH{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ DIFFERENCES FOUND:{Colors.RESET}")

        # Group differences by layer for better readability
        layer_groups = {}
        other_diffs = []

        for diff in differences:
            if "Layer " in diff and "|" in diff:
                # Extract layer info
                parts = diff.split("|", 1)
                if len(parts) >= 1:
                    layer_info = parts[0].strip()
                    if layer_info not in layer_groups:
                        layer_groups[layer_info] = []
                    layer_groups[layer_info].append(diff)
                else:
                    other_diffs.append(diff)
            else:
                other_diffs.append(diff)

        # Print layer-grouped differences
        for layer_info in sorted(layer_groups.keys()):
            print(f"\n  {Colors.BOLD}{Colors.BLUE}{layer_info}:{Colors.RESET}")
            layer_diffs = layer_groups[layer_info]

            for diff in layer_diffs[:20]:  # Limit per layer
                if diff.startswith("❌"):
                    print(f"    {Colors.RED}{diff}{Colors.RESET}")
                elif diff.startswith("    📊"):
                    print(f"      {Colors.YELLOW}{diff[4:]}{Colors.RESET}")
                else:
                    print(f"    {diff}")

        # Print other differences
        if other_diffs:
            print(f"\n  {Colors.BOLD}Other Differences:{Colors.RESET}")
            for diff in other_diffs[:10]:
                print(f"    {Colors.RED}{diff}{Colors.RESET}")
            if len(other_diffs) > 10:
                print(
                    f"    {Colors.YELLOW}... and {len(other_diffs) - 10} more general differences{Colors.RESET}"
                )


def compare_trace_files(
    file1: str, file2: str, tolerance: float = 1e-6, show_matches: bool = False
) -> bool:
    """
    Compare two JSONL trace files by content_hash with detailed hierarchical output

    Args:
        file1, file2: Paths to JSONL trace files
        tolerance: Numerical tolerance for floating point comparisons
        show_matches: Whether to show matching traces (default: only show differences)

    Returns:
        True if all traces match, False otherwise
    """
    print(f"{Colors.BOLD}Loading trace files...{Colors.RESET}")
    traces1 = load_jsonl(file1)
    traces2 = load_jsonl(file2)

    if not traces1:
        print(f"{Colors.RED}Error: No traces loaded from {file1}{Colors.RESET}")
        return False
    if not traces2:
        print(f"{Colors.RED}Error: No traces loaded from {file2}{Colors.RESET}")
        return False

    print(f"File 1: {len(traces1)} traces")
    print(f"File 2: {len(traces2)} traces")

    # Group by content_hash
    groups1 = group_by_content_hash(traces1)
    groups2 = group_by_content_hash(traces2)

    print(f"File 1: {len(groups1)} unique content hashes")
    print(f"File 2: {len(groups2)} unique content hashes")

    # Find common and unique hashes
    hashes1 = set(groups1.keys())
    hashes2 = set(groups2.keys())
    common_hashes = hashes1 & hashes2
    only_in_1 = hashes1 - hashes2
    only_in_2 = hashes2 - hashes1

    print(f"\nCommon hashes: {len(common_hashes)}")
    print(f"Only in file 1: {len(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")

    all_match = True
    match_count = 0
    diff_count = 0

    # Compare common hashes
    for content_hash in sorted(common_hashes):
        traces_1 = groups1[content_hash]
        traces_2 = groups2[content_hash]

        # For now, compare the first trace of each group
        trace1 = traces_1[0]
        trace2 = traces_2[0]

        records1 = trace1.get("precision_records", [])
        records2 = trace2.get("precision_records", [])

        is_match, differences = compare_precision_records(records1, records2, tolerance)

        if is_match:
            match_count += 1
            if show_matches:
                print_diff_header(content_hash, trace1, trace2)
                print_match_status(True, [])
        else:
            diff_count += 1
            all_match = False
            print_diff_header(content_hash, trace1, trace2)
            print_match_status(False, differences)

    # Show unique hashes with more detail
    if only_in_1:
        print(f"\n{Colors.YELLOW}Content hashes only in file 1:{Colors.RESET}")
        for content_hash in sorted(only_in_1):
            traces = groups1[content_hash]
            trace = traces[0]
            records = trace.get("precision_records", [])
            print(
                f"  {Colors.YELLOW}−{Colors.RESET} {content_hash} (Request ID: {trace.get('request_id', 'N/A')})"
            )
            print(f"    📊 {len(records)} precision records")
            if records:
                print(
                    f"    🔍 Layers: {', '.join(set(str(r.get('layer_id', 'unknown')) for r in records[:5]))}"
                )

    if only_in_2:
        print(f"\n{Colors.YELLOW}Content hashes only in file 2:{Colors.RESET}")
        for content_hash in sorted(only_in_2):
            traces = groups2[content_hash]
            trace = traces[0]
            records = trace.get("precision_records", [])
            print(
                f"  {Colors.YELLOW}+{Colors.RESET} {content_hash} (Request ID: {trace.get('request_id', 'N/A')})"
            )
            print(f"    📊 {len(records)} precision records")
            if records:
                print(
                    f"    🔍 Layers: {', '.join(set(str(r.get('layer_id', 'unknown')) for r in records[:5]))}"
                )

    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  {Colors.GREEN}Matches: {match_count}{Colors.RESET}")
    print(f"  {Colors.RED}Differences: {diff_count}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Only in file 1: {len(only_in_1)}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Only in file 2: {len(only_in_2)}{Colors.RESET}")

    if all_match and len(only_in_1) == 0 and len(only_in_2) == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All traces match!{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Traces have differences{Colors.RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSONL trace files by content_hash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trace_diff.py trace1.jsonl trace2.jsonl
  python trace_diff.py trace1.jsonl trace2.jsonl --tolerance 1e-4
  python trace_diff.py trace1.jsonl trace2.jsonl --show-matches
        """,
    )

    parser.add_argument("file1", help="First JSONL trace file")
    parser.add_argument("file2", help="Second JSONL trace file")
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-6,
        help="Numerical tolerance for floating point comparisons (default: 1e-6)",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Show matching traces (default: only show differences)",
    )

    args = parser.parse_args()

    # Check if files exist
    import os

    for file_path in [args.file1, args.file2]:
        if not os.path.exists(file_path):
            print(f"{Colors.RED}Error: File {file_path} does not exist{Colors.RESET}")
            return 1

    print(f"{Colors.BOLD}Comparing trace files:{Colors.RESET}")
    print(f"  File 1: {args.file1}")
    print(f"  File 2: {args.file2}")
    print(f"  Tolerance: {args.tolerance}")

    success = compare_trace_files(
        args.file1, args.file2, tolerance=args.tolerance, show_matches=args.show_matches
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
