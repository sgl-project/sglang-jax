import argparse
import json
import sys
from typing import Dict, List, Optional, Set, Tuple


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    BG_RED = "\033[48;5;124m"
    BG_GREEN = "\033[48;5;28m"
    BG_YELLOW = "\033[48;5;178m"


def load_jsonl(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def group_by_content_hash(traces: List[Dict]) -> Dict[str, List[Dict]]:
    groups = {}
    for trace in traces:
        content_hash = trace.get("content_hash")
        if content_hash:
            if content_hash not in groups:
                groups[content_hash] = []
            groups[content_hash].append(trace)
    return groups


def compare_precision_records(
    records1: Dict,
    records2: Dict,
    tolerance: float = 1e-6,
    max_decode_tokens: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    differences = []
    all_match = True

    if isinstance(records1, list) or isinstance(records2, list):
        differences.append("Legacy format detected - basic comparison only")
        return len(records1) == len(records2), differences

    # Compare structure: prefill and decode categories
    categories1 = set(records1.keys()) if isinstance(records1, dict) else set()
    categories2 = set(records2.keys()) if isinstance(records2, dict) else set()

    if categories1 != categories2:
        differences.append(f"Category mismatch: {categories1} vs {categories2}")
        return False, differences

    # Compare each category (prefill, decode) - ensure prefill comes first
    def category_sort_key(category):
        if category == "prefill":
            return 0
        elif category == "decode":
            return 1
        else:
            return 2

    for category in sorted(categories1, key=category_sort_key):
        tokens1 = records1.get(category, [])
        tokens2 = records2.get(category, [])

        category_match, category_diffs = compare_token_groups(
            category, tokens1, tokens2, tolerance, max_decode_tokens
        )

        if not category_match:
            all_match = False
            differences.extend(category_diffs)

    return all_match, differences


def compare_token_groups(
    category: str,
    tokens1: List[Dict],
    tokens2: List[Dict],
    tolerance: float = 1e-6,
    max_decode_tokens: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """Compare token groups within a category (prefill/decode)"""
    differences = []
    all_match = True

    # Apply max_decode_tokens limit for decode category
    if category == "decode" and max_decode_tokens is not None:
        original_len1, original_len2 = len(tokens1), len(tokens2)
        tokens1 = tokens1[:max_decode_tokens]
        tokens2 = tokens2[:max_decode_tokens]
        if original_len1 > max_decode_tokens or original_len2 > max_decode_tokens:
            differences.append(
                f"  {category}: Limited comparison to first {max_decode_tokens} tokens (original: {original_len1} vs {original_len2})"
            )

    if len(tokens1) != len(tokens2):
        differences.append(
            f"  {category}: Token count mismatch: {len(tokens1)} vs {len(tokens2)}"
        )
        all_match = False
        # Continue comparing up to the shorter length
        min_length = min(len(tokens1), len(tokens2))
    else:
        min_length = len(tokens1)

    # Compare each token group up to the shorter length
    for i in range(min_length):
        token1 = tokens1[i]
        token2 = tokens2[i]
        token_idx1 = token1.get("token_idx", i)
        token_idx2 = token2.get("token_idx", i)

        if token_idx1 != token_idx2:
            differences.append(
                f"    {category}[{i}]: Token index mismatch: {token_idx1} vs {token_idx2}"
            )
            all_match = False
            continue

        # Compare records within this token group
        records1 = token1.get("records", [])
        records2 = token2.get("records", [])

        token_match, token_diffs = compare_token_records(
            category, token_idx1, records1, records2, tolerance
        )

        if not token_match:
            all_match = False
            differences.extend(token_diffs)

    # Add information about skipped tokens if lengths differ
    if len(tokens1) != len(tokens2):
        skipped_count = abs(len(tokens1) - len(tokens2))
        longer_file = "file1" if len(tokens1) > len(tokens2) else "file2"
        differences.append(
            f"  {category}: {skipped_count} tokens skipped from {longer_file} (compared first {min_length} tokens)"
        )

    return all_match, differences


def compare_token_records(
    category: str,
    token_idx: int,
    records1: List[Dict],
    records2: List[Dict],
    tolerance: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """Compare records within a single token group"""
    differences = []
    all_match = True

    if len(records1) != len(records2):
        differences.append(
            f"    {category}[{token_idx}]: Record count mismatch: {len(records1)} vs {len(records2)}"
        )
        return False, differences

    # Group records by layer and module for alignment
    def group_records(records):
        groups = {}
        for rec in records:
            layer_id = rec.get("layer_id", "unknown")
            module_type = rec.get("module_type", "unknown")
            name = rec.get("name", "unnamed")
            key = f"{layer_id}_{module_type}_{name}"
            groups[key] = rec
        return groups

    groups1 = group_records(records1)
    groups2 = group_records(records2)

    all_keys = set(groups1.keys()) | set(groups2.keys())

    for key in sorted(all_keys):
        layer_id, module_type, name = key.split("_", 2)

        if key not in groups1:
            differences.append(
                f"      {category}[{token_idx}] {layer_id}.{module_type}.{name}: Missing in trace 1"
            )
            all_match = False
            continue

        if key not in groups2:
            differences.append(
                f"      {category}[{token_idx}] {layer_id}.{module_type}.{name}: Missing in trace 2"
            )
            all_match = False
            continue

        rec1, rec2 = groups1[key], groups2[key]

        # Compare basic fields
        for field in ["stage", "shape", "dtype"]:
            val1, val2 = rec1.get(field), rec2.get(field)
            if val1 != val2:
                differences.append(
                    f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MISMATCH|{val1}|{val2}"
                )
                all_match = False
            else:
                differences.append(
                    f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MATCH|{val1}"
                )

        # Compare numerical values
        for field in ["min", "max", "mean", "std"]:
            val1, val2 = rec1.get(field), rec2.get(field)
            if val1 is not None and val2 is not None:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = abs(val1 - val2)
                    if diff > tolerance:
                        differences.append(
                            f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MISMATCH|{val1:.6f}|{val2:.6f}|{diff:.6f}"
                        )
                        all_match = False
                    else:
                        differences.append(
                            f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MATCH|{val1:.6f}|{val2:.6f}|{diff:.6f}"
                        )
                elif val1 != val2:
                    differences.append(
                        f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MISMATCH|{val1}|{val2}"
                    )
                    all_match = False
                else:
                    differences.append(
                        f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MATCH|{val1}"
                    )

        # Compare boolean flags
        for field in ["has_nan", "has_inf"]:
            val1, val2 = rec1.get(field), rec2.get(field)
            if val1 != val2:
                differences.append(
                    f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MISMATCH|{val1}|{val2}"
                )
                all_match = False
            else:
                differences.append(
                    f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.{field}: MATCH|{val1}"
                )

        # For decode, compare token_stats if present
        if category == "decode":
            token_stats1 = rec1.get("token_stats", [])
            token_stats2 = rec2.get("token_stats", [])

            if token_stats1 and token_stats2:
                if len(token_stats1) != len(token_stats2):
                    differences.append(
                        f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.token_stats: Length mismatch {len(token_stats1)} vs {len(token_stats2)}"
                    )
                    all_match = False
                else:
                    # Compare token-level stats
                    for ts_idx, (ts1, ts2) in enumerate(
                        zip(token_stats1, token_stats2)
                    ):
                        for ts_field in ["min", "max", "mean", "std", "value"]:
                            ts_val1, ts_val2 = ts1.get(ts_field), ts2.get(ts_field)
                            if ts_val1 is not None and ts_val2 is not None:
                                if isinstance(ts_val1, (int, float)) and isinstance(
                                    ts_val2, (int, float)
                                ):
                                    ts_diff = abs(ts_val1 - ts_val2)
                                    if ts_diff > tolerance:
                                        differences.append(
                                            f"          {category}[{token_idx}] {layer_id}.{module_type}.{name}.token[{ts_idx}].{ts_field}: MISMATCH|{ts_val1:.6f}|{ts_val2:.6f}|{ts_diff:.6f}"
                                        )
                                        all_match = False
                                    else:
                                        differences.append(
                                            f"          {category}[{token_idx}] {layer_id}.{module_type}.{name}.token[{ts_idx}].{ts_field}: MATCH|{ts_val1:.6f}|{ts_val2:.6f}|{ts_diff:.6f}"
                                        )
                                elif ts_val1 != ts_val2:
                                    differences.append(
                                        f"          {category}[{token_idx}] {layer_id}.{module_type}.{name}.token[{ts_idx}].{ts_field}: MISMATCH|{ts_val1}|{ts_val2}"
                                    )
                                    all_match = False
                                else:
                                    differences.append(
                                        f"          {category}[{token_idx}] {layer_id}.{module_type}.{name}.token[{ts_idx}].{ts_field}: MATCH|{ts_val1}"
                                    )
            elif token_stats1 or token_stats2:
                differences.append(
                    f"        {category}[{token_idx}] {layer_id}.{module_type}.{name}.token_stats: Availability mismatch {bool(token_stats1)} vs {bool(token_stats2)}"
                )
                all_match = False

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


def format_comparison_result(message: str) -> str:
    """Format comparison result with appropriate colors"""
    if "|" not in message:
        return message

    parts = message.split("|")
    if len(parts) < 2:
        return message

    status = parts[0]
    if status == "MATCH":
        if len(parts) == 2:
            # Simple match: MATCH|value
            return f"{Colors.BG_GREEN}{Colors.WHITE} ✓ {parts[1]} {Colors.RESET}"
        elif len(parts) == 4:
            # Numerical match with diff: MATCH|val1|val2|diff
            return f"{Colors.BG_GREEN}{Colors.WHITE} ✓ {parts[1]} ≈ {parts[2]} (diff: {parts[3]}) {Colors.RESET}"
        else:
            return f"{Colors.BG_GREEN}{Colors.WHITE} ✓ {parts[1]} {Colors.RESET}"
    elif status == "MISMATCH":
        if len(parts) == 3:
            # Simple mismatch: MISMATCH|val1|val2
            return f"{Colors.BG_RED}{Colors.WHITE} ✗ {parts[1]} ≠ {parts[2]} {Colors.RESET}"
        elif len(parts) == 4:
            # Numerical mismatch: MISMATCH|val1|val2|diff
            return f"{Colors.BG_RED}{Colors.WHITE} ✗ {parts[1]} ≠ {parts[2]} (diff: {parts[3]}) {Colors.RESET}"
        else:
            return f"{Colors.BG_RED}{Colors.WHITE} ✗ {parts[1]} {Colors.RESET}"

    return message


def print_tree_differences(differences: List[str]):
    """Print differences in a tree-like structure with color coding"""
    if not differences:
        print(f"{Colors.BG_GREEN}{Colors.WHITE} ALL MATCH {Colors.RESET}")
        return

    has_mismatches = any("MISMATCH" in diff for diff in differences)
    if has_mismatches:
        print(f"{Colors.RED}DIFFERENCES FOUND:{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}ALL VALUES MATCH:{Colors.RESET}")

    # Group differences by category and token for tree display
    tree = {}

    for diff in differences:
        parts = diff.split(": ", 1)
        if len(parts) != 2:
            continue

        path = parts[0].strip()
        message = parts[1].strip()

        # Parse the hierarchical path
        if "[" in path and "]" in path:
            # Extract category and token info
            if "prefill[" in path or "decode[" in path:
                category_part = path.split("[")[0].strip()
                rest = path.split("]", 1)[1].strip() if "]" in path else ""
                token_part = (
                    path.split("[")[1].split("]")[0]
                    if "[" in path and "]" in path
                    else "0"
                )

                if category_part not in tree:
                    tree[category_part] = {}
                if token_part not in tree[category_part]:
                    tree[category_part][token_part] = []

                tree[category_part][token_part].append(f"{rest}: {message}")
            else:
                # Other format
                if "other" not in tree:
                    tree["other"] = {"0": []}
                tree["other"]["0"].append(diff)
        else:
            # Top-level difference
            if "root" not in tree:
                tree["root"] = {"0": []}
            tree["root"]["0"].append(diff)

    # Print tree structure - ensure prefill comes first
    def display_category_sort_key(category):
        if category == "prefill":
            return 0
        elif category == "decode":
            return 1
        elif category == "root":
            return 999  # root always last
        else:
            return 2

    for category in sorted(tree.keys(), key=display_category_sort_key):
        if category == "root":
            continue
        print(f"\n{Colors.BOLD}{category.upper()}:{Colors.RESET}")

        tokens = tree[category]
        for token_idx in sorted(
            tokens.keys(), key=lambda x: int(x) if x.isdigit() else 999
        ):
            token_diffs = tokens[token_idx]
            if len(token_diffs) > 0:
                print(f"  Token[{token_idx}]:")

                # Group by layer/module
                layer_groups = {}
                for diff in token_diffs:
                    if " " in diff:
                        layer_part = diff.split(" ")[0] if " " in diff else diff
                        if layer_part not in layer_groups:
                            layer_groups[layer_part] = []
                        layer_groups[layer_part].append(diff)
                    else:
                        if "misc" not in layer_groups:
                            layer_groups["misc"] = []
                        layer_groups["misc"].append(diff)

                # Sort and group by layer number for better organization
                def parse_layer_module(layer_key):
                    """Parse layer key like '1.SELF.ATTN_self_attn_output.std' into (layer_num, module_type)"""
                    if layer_key == "misc":
                        return (999, "misc")

                    # Find the first dot to extract layer number
                    first_dot = layer_key.find(".")
                    if first_dot > 0:
                        layer_part = layer_key[:first_dot]
                        if layer_part.isdigit():
                            # Extract module type after layer number (e.g., "SELF.ATTN", "INPUT.LAYERNORM", "MLP")
                            remaining = layer_key[first_dot + 1 :]
                            # Get the module type part before the first underscore
                            if "_" in remaining:
                                before_underscore = remaining.split("_")[0]
                                # Special handling for MLP pattern
                                if before_underscore.startswith("MLP."):
                                    module_type = "MLP"
                                else:
                                    module_type = before_underscore
                            else:
                                # If no underscore, take the remaining as module type
                                module_type = remaining
                            return (int(layer_part), module_type)

                    return (999, layer_key)

                # Group by layer number first
                layers_by_num = {}
                for layer_key in layer_groups.keys():
                    layer_num, module_type = parse_layer_module(layer_key)
                    if layer_num not in layers_by_num:
                        layers_by_num[layer_num] = {}
                    layers_by_num[layer_num][layer_key] = layer_groups[layer_key]

                # Display with separators
                for layer_num in sorted(layers_by_num.keys()):
                    if layer_num != 999:  # Skip misc for layer separator
                        print(f"    {Colors.BLUE}{'=' * 50}{Colors.RESET}")
                        print(f"    {Colors.BOLD}Layer {layer_num}{Colors.RESET}")
                        print(f"    {Colors.BLUE}{'=' * 50}{Colors.RESET}")

                    layer_modules = layers_by_num[layer_num]
                    module_keys = sorted(
                        layer_modules.keys(), key=lambda x: parse_layer_module(x)[1]
                    )

                    for i, layer_key in enumerate(module_keys):
                        # Add module separator between different modules in the same layer
                        if i > 0:
                            print(f"    {Colors.CYAN}{'-' * 30}{Colors.RESET}")

                        if layer_key != "misc":
                            print(f"    {layer_key}:")

                        # Separate matches and mismatches for better organization
                        matches = []
                        mismatches = []
                        others = []

                        for diff in layer_modules[layer_key]:
                            if ": " in diff:
                                field_msg = diff.split(": ", 1)[1]
                                if "MATCH" in field_msg:
                                    matches.append(field_msg)
                                elif "MISMATCH" in field_msg:
                                    mismatches.append(field_msg)
                                else:
                                    others.append(field_msg)
                            else:
                                others.append(diff)

                        # Show mismatches first (more important)
                        for msg in mismatches[:3]:  # Limit output
                            formatted = format_comparison_result(msg)
                            print(f"      {formatted}")

                        # Then show matches
                        for msg in matches[:3]:  # Limit output
                            formatted = format_comparison_result(msg)
                            print(f"      {formatted}")

                        # Then others
                        for msg in others[:2]:
                            print(f"      {Colors.YELLOW}{msg}{Colors.RESET}")

                        total_items = len(mismatches) + len(matches) + len(others)
                        shown_items = (
                            min(3, len(mismatches))
                            + min(3, len(matches))
                            + min(2, len(others))
                        )
                        if total_items > shown_items:
                            print(
                                f"      {Colors.YELLOW}... and {total_items - shown_items} more{Colors.RESET}"
                            )

    # Print root-level differences
    if "root" in tree:
        print(f"\n{Colors.BOLD}GENERAL:{Colors.RESET}")
        for diff in tree["root"]["0"][:10]:
            formatted = format_comparison_result(diff)
            print(f"  {formatted}")


def print_match_status(is_match: bool, differences: List[str]):
    """Print match status with tree-like hierarchy"""
    print_tree_differences(differences)


def compare_trace_files(
    file1: str,
    file2: str,
    tolerance: float = 1e-6,
    show_matches: bool = False,
    max_decode_tokens: Optional[int] = None,
) -> bool:
    """
    Compare two JSONL trace files by content_hash with tree-structured output

    Args:
        file1, file2: Paths to JSONL trace files
        tolerance: Numerical tolerance for floating point comparisons
        show_matches: Whether to show matching traces (default: only show differences)
        max_decode_tokens: Maximum number of decode tokens to compare (default: compare all)

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

        is_match, differences = compare_precision_records(
            records1, records2, tolerance, max_decode_tokens
        )

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
                f"  {Colors.YELLOW}-{Colors.RESET} {content_hash} (Request ID: {trace.get('request_id', 'N/A')})"
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

    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  {Colors.GREEN}Matches: {match_count}{Colors.RESET}")
    print(f"  {Colors.RED}Differences: {diff_count}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Only in file 1: {len(only_in_1)}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Only in file 2: {len(only_in_2)}{Colors.RESET}")

    if all_match and len(only_in_1) == 0 and len(only_in_2) == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All traces match!{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Traces have differences{Colors.RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSONL trace files by content_hash with token-level tree display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trace_diff.py trace1.jsonl trace2.jsonl
  python trace_diff.py trace1.jsonl trace2.jsonl --tolerance 1e-4
  python trace_diff.py trace1.jsonl trace2.jsonl --show-matches
  python trace_diff.py trace1.jsonl trace2.jsonl --max-decode-tokens 5
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
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=None,
        help="Maximum number of decode tokens to compare (default: compare all)",
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
    if args.max_decode_tokens:
        print(f"  Max decode tokens: {args.max_decode_tokens}")

    success = compare_trace_files(
        args.file1,
        args.file2,
        tolerance=args.tolerance,
        show_matches=args.show_matches,
        max_decode_tokens=args.max_decode_tokens,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
