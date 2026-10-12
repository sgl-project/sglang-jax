"""RunAI byte-range I/O for the shared JAX safetensors weight loader.

Unlike the PyTorch loaders' sequential tensor iterator, JAX requests local
shards in mapping order, including expert groups spanning multiple files.
FileStreamer preserves that contract without staging the checkpoint on disk.
"""

import hashlib
import os
from pathlib import Path

from filelock import FileLock


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def _sdk():
    try:
        import runai_model_streamer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "RunAI loading requires the optional Linux dependency: "
            "pip install 'sglang-jax[runai]'"
        ) from exc
    return runai_model_streamer


def configure_runai(extra_config: dict) -> None:
    """Validate all options before applying the SDK's process-wide settings."""
    if not isinstance(extra_config, dict):
        raise ValueError("RunAI model-loader-extra-config must be a JSON object")
    unknown = set(extra_config) - {"concurrency", "memory_limit", "distributed"}
    if unknown:
        raise ValueError(f"Unexpected runai_streamer options: {sorted(unknown)}")
    if extra_config.get("distributed", False) is not False:
        raise ValueError(
            "RunAI distributed streaming requires torch.distributed and is not supported "
            "by the JAX loader; each JAX process reads its addressable shards."
        )
    updates = {}
    for key in ("concurrency", "memory_limit"):
        if key in extra_config:
            value = extra_config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"RunAI {key} must be a positive integer, got {value!r}")
            updates[f"RUNAI_STREAMER_{key.upper()}"] = str(value)
    os.environ.update(updates)


def download_metadata(uri: str, download_dir: str | None = None) -> str:
    """Stage config/tokenizer files once, leaving weight objects in GCS."""
    if not is_gcs_path(uri) or not uri[5:].strip("/"):
        raise ValueError(f"Expected a GCS model directory, got {uri!r}")
    root = Path(download_dir or os.getenv("HF_HOME") or Path.home() / ".cache/huggingface")
    root = root / "sglang-jax-runai"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / hashlib.sha256(uri.rstrip("/").encode()).hexdigest()
    _sdk()
    # The SDK's GCS metadata helpers call get_bucket(), which unnecessarily
    # requires storage.buckets.get. Object readers may only have list/get.
    # Keep the same source-specific, process-safe cache without that extra RPC.
    sentinel = destination / ".sglang_complete"
    with FileLock(str(destination) + ".lock"):
        if sentinel.exists():
            return str(destination)
        destination.mkdir(parents=True, exist_ok=True)
        _, prefix = _gcs_location(uri)
        count = 0
        for blob in _gcs_blobs(uri):
            if blob.name.endswith(("/", ".safetensors", ".bin", ".pt", ".pth", ".tensors")):
                continue
            if not blob.name.startswith(prefix):
                raise ValueError(f"Object is outside model prefix: {blob.name!r}")
            target = (destination / blob.name[len(prefix) :]).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Object escapes metadata directory: {blob.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            count += 1
        if not count:
            raise ValueError(f"No metadata files found at {uri!r}")
        sentinel.touch()
    return str(destination)


def _gcs_location(uri: str):
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/") + "/" if prefix.strip("/") else ""


def _gcs_blobs(uri: str, *, delimiter=None):
    from google.cloud import storage
    from runai_model_streamer_gcs.credentials.credentials import get_credentials

    bucket, prefix = _gcs_location(uri)
    # Reuse the SDK's credential selection; bucket() constructs a resource
    # without fetching bucket metadata, unlike get_bucket().
    client = storage.Client(credentials=get_credentials().gcp_credentials())
    return client.list_blobs(client.bucket(bucket), prefix=prefix, delimiter=delimiter)


def _list_safetensors(path: str):
    if not is_gcs_path(path):
        return _sdk().list_safetensors(path)
    bucket, _ = _gcs_location(path)
    return [
        f"gs://{bucket}/{blob.name}"
        for blob in _gcs_blobs(path, delimiter="/")
        if blob.name.endswith(".safetensors")
    ]
