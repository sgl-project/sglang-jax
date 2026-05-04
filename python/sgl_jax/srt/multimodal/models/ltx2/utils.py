"""Utility functions for LTX-2 model loading."""

import os
import shutil
import tempfile


def get_hf_snapshot_dir(repo_id: str) -> str | None:
    """Find the HuggingFace snapshot directory for a given repo.

    Searches for cached model snapshots using the HF_HOME environment variable,
    falling back to the default HuggingFace cache location.

    Args:
        repo_id: HuggingFace repository ID (e.g., "Lightricks/LTX-2").

    Returns:
        Absolute path to the snapshot directory, or None if not found.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub") if not hf_home.endswith("hub") else hf_home
    # HF cache uses "models--org--name" format
    cache_name = f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = os.path.join(hub_dir, cache_name, "snapshots")

    # Also check if HF_HOME itself contains the models-- directory (non-standard layout)
    alt_snapshot_dir = os.path.join(hf_home, cache_name, "snapshots")

    # Prefer the snapshot that contains actual .safetensors files
    # (some HF cache layouts have snapshots with only symlinks that may not resolve)
    best = None
    for candidate in [snapshot_dir, alt_snapshot_dir]:
        if os.path.exists(candidate):
            snapshots = os.listdir(candidate)
            if snapshots:
                snap_path = os.path.join(candidate, snapshots[0])
                has_weights = any(f.endswith(".safetensors") for f in os.listdir(snap_path))
                if has_weights:
                    return snap_path
                if best is None:
                    best = snap_path

    return best


def get_ltx2_checkpoint_dir(checkpoint_name: str = "ltx-2-19b-dev.safetensors") -> str | None:
    """Get a directory containing only the main LTX-2 checkpoint file.

    The LTX-2 HF repo contains multiple safetensors variants (FP4, FP8,
    distilled). WeightLoader scans all *.safetensors in a directory, which
    causes conflicts with quantization scale keys from the variant files.
    This function returns a temp directory with a symlink to only the main
    checkpoint, suitable for use as model_config.model_path.

    The caller should call cleanup_ltx2_checkpoint_dir() after loading.

    Args:
        checkpoint_name: Name of the main checkpoint file.

    Returns:
        Path to a temp directory containing only the main checkpoint, or
        the snapshot dir if the specific file isn't found.
    """
    ltx_path = get_hf_snapshot_dir("Lightricks/LTX-2")
    if ltx_path is None:
        return None

    main_ckpt = os.path.join(ltx_path, checkpoint_name)
    if os.path.exists(main_ckpt):
        tmpdir = tempfile.mkdtemp(prefix="ltx2_ckpt_")
        os.symlink(main_ckpt, os.path.join(tmpdir, checkpoint_name))
        return tmpdir

    return ltx_path


def cleanup_ltx2_checkpoint_dir(path: str | None) -> None:
    """Clean up a temp directory created by get_ltx2_checkpoint_dir."""
    if path is not None and path.startswith(tempfile.gettempdir()):
        shutil.rmtree(path, ignore_errors=True)
