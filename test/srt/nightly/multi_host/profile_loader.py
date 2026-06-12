"""Back-compat shim: the launch-profile contract now lives in the host-neutral
``test/srt/nightly/profiles.py`` (shared by single- and multi-host runners).

Kept so existing ``from profile_loader import LaunchProfile, load_profile,
build_other_server_args`` imports inside the multi_host package keep working.
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profiles import (  # noqa: E402,F401
    LaunchProfile,
    RuntimeConfig,
    build_other_server_args,
    load_profile,
)
