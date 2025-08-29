#!/usr/bin/env python3
"""
JSON file comparison tool that ignores the order of entries.
Supports both single JSON objects and JSONL (one JSON object per line) format.
"""

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file, supporting both JSON and JSONL formats.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of JSON objects
    """
    objects = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try to parse as a single JSON object first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            objects = data
        else:
            objects = [data]
    except json.JSONDecodeError:
        # If that fails, try parsing as JSONL (one JSON object per line)
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                sys.exit(1)

    return objects


def get_object_key(obj: Dict[str, Any], key_field: str = "id") -> str:
    """
    Get the key for an object for comparison purposes.

    Args:
        obj: JSON object
        key_field: Field name to use as key (default: "id")

    Returns:
        String key for the object
    """
    if key_field in obj:
        return str(obj[key_field])
    else:
        # If no key field, use the entire object as key (converted to string)
        return json.dumps(obj, sort_keys=True)


class Colors:
    """ANSI color codes for terminal output"""

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Git diff style colors
    REMOVED = "\033[31m"  # Red
    ADDED = "\033[32m"  # Green
    CONTEXT = "\033[37m"  # White/default
    HEADER = "\033[36m"  # Cyan

    @classmethod
    def disable_if_no_tty(cls):
        """Disable colors if not outputting to a terminal"""
        if not sys.stdout.isatty():
            for attr in dir(cls):
                if not attr.startswith("_") and isinstance(getattr(cls, attr), str):
                    setattr(cls, attr, "")


def format_diff_line(line: str, line_type: str, use_colors: bool = True) -> str:
    """Format a diff line with appropriate colors and prefixes"""
    if not use_colors:
        return line

    if line_type == "removed":
        return f"{Colors.REMOVED}-{line}{Colors.RESET}"
    elif line_type == "added":
        return f"{Colors.ADDED}+{line}{Colors.RESET}"
    elif line_type == "context":
        return f" {line}"
    else:
        return line


def get_word_diff(text1: str, text2: str, use_colors: bool = True) -> str:
    """Generate word-level diff highlighting"""
    words1 = re.split(r"(\s+)", text1)
    words2 = re.split(r"(\s+)", text2)

    result = []
    matcher = difflib.SequenceMatcher(None, words1, words2)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.extend(words1[i1:i2])
        elif tag == "delete":
            if use_colors:
                for word in words1[i1:i2]:
                    if word.strip():
                        result.append(f"{Colors.REMOVED}{word}{Colors.RESET}")
                    else:
                        result.append(word)
            else:
                result.extend([f"[-{word}-]" for word in words1[i1:i2]])
        elif tag == "insert":
            if use_colors:
                for word in words2[j1:j2]:
                    if word.strip():
                        result.append(f"{Colors.ADDED}{word}{Colors.RESET}")
                    else:
                        result.append(word)
            else:
                result.extend([f"[+{word}+]" for word in words2[j1:j2]])
        elif tag == "replace":
            if use_colors:
                for word in words1[i1:i2]:
                    if word.strip():
                        result.append(f"{Colors.REMOVED}{word}{Colors.RESET}")
                    else:
                        result.append(word)
                for word in words2[j1:j2]:
                    if word.strip():
                        result.append(f"{Colors.ADDED}{word}{Colors.RESET}")
                    else:
                        result.append(word)
            else:
                result.extend([f"[-{word}-]" for word in words1[i1:i2]])
                result.extend([f"[+{word}+]" for word in words2[j1:j2]])

    return "".join(result)


def get_line_diff(
    text1: str, text2: str, context_lines: int = 3, use_colors: bool = True
) -> str:
    """Generate line-by-line unified diff"""
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines1, lines2, fromfile="file1", tofile="file2", n=context_lines, lineterm=""
    )

    result = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            if use_colors:
                result.append(f"{Colors.HEADER}{line}{Colors.RESET}")
            else:
                result.append(line)
        elif line.startswith("@@"):
            if use_colors:
                result.append(f"{Colors.CYAN}{line}{Colors.RESET}")
            else:
                result.append(line)
        elif line.startswith("-"):
            result.append(format_diff_line(line[1:], "removed", use_colors))
        elif line.startswith("+"):
            result.append(format_diff_line(line[1:], "added", use_colors))
        else:
            result.append(
                format_diff_line(
                    line[1:] if line.startswith(" ") else line, "context", use_colors
                )
            )

    return "\n".join(result)


def format_json_value(value: Any, max_length: int = 100) -> str:
    """Format a JSON value for display, truncating if too long"""
    if isinstance(value, str):
        # Handle long strings specially
        if len(value) > max_length:
            return repr(value[:max_length] + "...")
        return repr(value)
    else:
        json_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(json_str) > max_length:
            return json_str[:max_length] + "..."
        return json_str


class DiffResult:
    """Container for diff results with detailed formatting"""

    def __init__(self, path: str, value1: Any, value2: Any, diff_type: str):
        self.path = path
        self.value1 = value1
        self.value2 = value2
        self.diff_type = diff_type

    def format_diff(self, use_colors: bool = True, diff_mode: str = "word") -> str:
        """Format the diff for display"""
        header = f"{Colors.BOLD if use_colors else ''}{self.path}{Colors.RESET if use_colors else ''}"

        if self.diff_type == "missing_in_first":
            return f"{header}\n  {format_diff_line(format_json_value(self.value2), 'added', use_colors)}"
        elif self.diff_type == "missing_in_second":
            return f"{header}\n  {format_diff_line(format_json_value(self.value1), 'removed', use_colors)}"
        elif self.diff_type == "type_mismatch":
            v1_str = f"{type(self.value1).__name__}: {format_json_value(self.value1)}"
            v2_str = f"{type(self.value2).__name__}: {format_json_value(self.value2)}"
            return f"{header}\n  {format_diff_line(v1_str, 'removed', use_colors)}\n  {format_diff_line(v2_str, 'added', use_colors)}"
        elif self.diff_type == "value_mismatch":
            # Handle string values specially for better diff
            if isinstance(self.value1, str) and isinstance(self.value2, str):
                if diff_mode == "line" and ("\n" in self.value1 or "\n" in self.value2):
                    diff_output = get_line_diff(
                        self.value1, self.value2, use_colors=use_colors
                    )
                    return f"{header}\n{diff_output}"
                elif diff_mode == "word":
                    diff_output = get_word_diff(
                        self.value1, self.value2, use_colors=use_colors
                    )
                    return f"{header}\n  {diff_output}"

            # Fallback to simple before/after format
            return f"{header}\n  {format_diff_line(format_json_value(self.value1), 'removed', use_colors)}\n  {format_diff_line(format_json_value(self.value2), 'added', use_colors)}"

        return f"{header}\n  Unknown diff type: {self.diff_type}"


def compare_objects(
    obj1: Dict[str, Any], obj2: Dict[str, Any]
) -> Tuple[bool, List[DiffResult]]:
    """Compare two JSON objects and return detailed diff results"""
    differences = []

    def compare_values(v1, v2, path=""):
        if type(v1) != type(v2):
            differences.append(DiffResult(path, v1, v2, "type_mismatch"))
            return False

        if isinstance(v1, dict):
            all_keys = set(v1.keys()) | set(v2.keys())
            equal = True
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in v1:
                    differences.append(
                        DiffResult(new_path, None, v2[key], "missing_in_first")
                    )
                    equal = False
                elif key not in v2:
                    differences.append(
                        DiffResult(new_path, v1[key], None, "missing_in_second")
                    )
                    equal = False
                else:
                    equal &= compare_values(v1[key], v2[key], new_path)
            return equal
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                differences.append(
                    DiffResult(f"{path}[length]", len(v1), len(v2), "value_mismatch")
                )
                return False
            equal = True
            for i, (item1, item2) in enumerate(zip(v1, v2)):
                equal &= compare_values(item1, item2, f"{path}[{i}]")
            return equal
        else:
            if v1 != v2:
                differences.append(DiffResult(path, v1, v2, "value_mismatch"))
                return False
            return True

    is_equal = compare_values(obj1, obj2)
    return is_equal, differences


def compare_json_files(
    file1_path: str,
    file2_path: str,
    key_field: str = "id",
    verbose: bool = False,
    use_colors: bool = True,
    diff_mode: str = "word",
) -> bool:
    """
    Compare two JSON files ignoring the order of entries.

    Args:
        file1_path: Path to first JSON file
        file2_path: Path to second JSON file
        key_field: Field name to use as key for matching objects
        verbose: Whether to show detailed differences

    Returns:
        True if files are equivalent, False otherwise
    """
    # Load data from both files
    objects1 = load_json_file(file1_path)
    objects2 = load_json_file(file2_path)

    print(f"File 1: {len(objects1)} objects")
    print(f"File 2: {len(objects2)} objects")

    # Create dictionaries with keys for easy lookup
    dict1 = {}
    dict2 = {}

    for obj in objects1:
        key = get_object_key(obj, key_field)
        if key in dict1:
            print(f"Warning: Duplicate key '{key}' found in file 1")
        dict1[key] = obj

    for obj in objects2:
        key = get_object_key(obj, key_field)
        if key in dict2:
            print(f"Warning: Duplicate key '{key}' found in file 2")
        dict2[key] = obj

    # Compare the dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())
    only_in_file1 = set(dict1.keys()) - set(dict2.keys())
    only_in_file2 = set(dict2.keys()) - set(dict1.keys())
    common_keys = set(dict1.keys()) & set(dict2.keys())

    files_equal = True

    # Report keys only in file1
    if only_in_file1:
        files_equal = False
        print(f"\nObjects only in file 1 ({len(only_in_file1)}):")
        for key in sorted(only_in_file1):
            print(f"  {key}")

    # Report keys only in file2
    if only_in_file2:
        files_equal = False
        print(f"\nObjects only in file 2 ({len(only_in_file2)}):")
        for key in sorted(only_in_file2):
            print(f"  {key}")

    # Compare common objects
    different_objects = []
    for key in common_keys:
        obj1 = dict1[key]
        obj2 = dict2[key]
        is_equal, differences = compare_objects(obj1, obj2)
        if not is_equal:
            different_objects.append((key, differences))

    if different_objects:
        files_equal = False
        header = f"\nObjects with differences ({len(different_objects)}):"
        if use_colors:
            print(f"{Colors.BOLD}{header}{Colors.RESET}")
        else:
            print(header)

        for key, differences in different_objects:
            key_header = f"\n{Colors.YELLOW if use_colors else ''}=== {key} ==={Colors.RESET if use_colors else ''}"
            print(key_header)

            if verbose:
                max_diffs = 20  # Show more diffs in verbose mode
                for i, diff in enumerate(differences[:max_diffs]):
                    print(diff.format_diff(use_colors=use_colors, diff_mode=diff_mode))
                    if (
                        i < len(differences[:max_diffs]) - 1
                    ):  # Add separator between diffs
                        print()

                if len(differences) > max_diffs:
                    remaining = len(differences) - max_diffs
                    print(
                        f"\n{Colors.DIM if use_colors else ''}... and {remaining} more difference(s){Colors.RESET if use_colors else ''}"
                    )
            else:
                diff_summary = f"{len(differences)} difference(s)"
                if use_colors:
                    print(f"  {Colors.DIM}{diff_summary}{Colors.RESET}")
                else:
                    print(f"  {diff_summary}")
                print(
                    f"  {Colors.DIM if use_colors else ''}(use --verbose to see details){Colors.RESET if use_colors else ''}"
                )

    if files_equal:
        success_msg = "\n✅ Files are identical (ignoring order)"
        if use_colors:
            print(f"{Colors.GREEN}{success_msg}{Colors.RESET}")
        else:
            print(success_msg)
    else:
        error_msg = "\n❌ Files have differences"
        if use_colors:
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
        else:
            print(error_msg)

    return files_equal


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSON files ignoring the order of entries"
    )
    parser.add_argument("file1", help="First JSON file to compare")
    parser.add_argument("file2", help="Second JSON file to compare")
    parser.add_argument(
        "--key",
        "-k",
        default="id",
        help="Field name to use as key for matching objects (default: id)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed differences"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--diff-mode",
        choices=["word", "line"],
        default="word",
        help="Diff mode for text content (default: word)",
    )

    args = parser.parse_args()

    try:
        use_colors = not args.no_color
        if use_colors:
            Colors.disable_if_no_tty()

        files_equal = compare_json_files(
            args.file1,
            args.file2,
            key_field=args.key,
            verbose=args.verbose,
            use_colors=use_colors,
            diff_mode=args.diff_mode,
        )
        sys.exit(0 if files_equal else 1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
