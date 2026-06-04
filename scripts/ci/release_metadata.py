"""Compute release metadata for GitHub Actions workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from packaging.version import InvalidVersion, Version


def canonical_version(raw: str) -> str:
    value = raw.strip()
    try:
        return str(Version(value))
    except InvalidVersion as exc:
        raise SystemExit(f"::error::Invalid PEP 440 version {value!r}: {exc}") from exc


def write_github_outputs(outputs: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")


def compute_from_version(args: argparse.Namespace) -> None:
    version = canonical_version(args.version)
    write_github_outputs(
        {
            "version": version,
            "tag": f"v{version}",
        }
    )


def compute_from_tag(args: argparse.Namespace) -> None:
    tag = args.tag.strip()
    if not tag.startswith("v") or len(tag) == 1:
        raise SystemExit(f"::error::Tag {tag!r} does not start with 'v'")

    version = canonical_version(tag[1:])
    canonical_tag = f"v{version}"
    if tag != canonical_tag:
        raise SystemExit(f"::error::Tag {tag!r} is not canonical; use {canonical_tag!r}")

    parsed = Version(version)
    is_prerelease = parsed.is_prerelease
    write_github_outputs(
        {
            "tag": canonical_tag,
            "version": version,
            "is_prerelease": str(is_prerelease).lower(),
            "update_latest": str(not is_prerelease).lower(),
            "dry_run": str(args.dry_run).lower(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    version_parser = subparsers.add_parser("from-version")
    version_parser.add_argument("--version", required=True)
    version_parser.set_defaults(func=compute_from_version)

    tag_parser = subparsers.add_parser("from-tag")
    tag_parser.add_argument("--tag", required=True)
    tag_parser.add_argument("--dry-run", action="store_true")
    tag_parser.set_defaults(func=compute_from_tag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
