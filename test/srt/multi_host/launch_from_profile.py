"""CLI: print the launch_server command implied by a LaunchProfile (dry only)."""

import argparse
import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from profile_loader import load_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the sgl_jax.launch_server command for a launch profile"
    )
    parser.add_argument("profile", help="Path to launch profile YAML")
    args = parser.parse_args()

    profile = load_profile(args.profile)

    argv = [
        "python",
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        profile.model_path,
        "--tp-size",
        str(profile.tp_size),
        "--dp-size",
        str(profile.dp_size),
    ]
    if profile.ep_size is not None:
        argv.extend(["--ep-size", str(profile.ep_size)])
    argv.extend(["--host", "0.0.0.0", "--port", str(profile.port)])
    argv.extend(profile.server_args)

    print(" ".join(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
