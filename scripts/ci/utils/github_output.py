"""Shared helper for writing GitHub Actions step outputs."""

import os


def write_github_output(name, value):
    """Write a name=value pair to GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"  output: {name}={value}")
