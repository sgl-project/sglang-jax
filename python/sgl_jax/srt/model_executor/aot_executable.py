"""Persist model executables and load them after checking the serving lowering."""

import hashlib
import importlib.metadata
import json
import logging
import os
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
from jax.experimental import serialize_executable

logger = logging.getLogger(__name__)


def _runtime_versions():
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for package in ("jax", "jaxlib", "flax", "libtpu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _compile_flags():
    # Export owns its dump paths. They do not change the executable program.
    return {
        name: [flag for flag in shlex.split(os.environ.get(name, "")) if "dump" not in flag]
        for name in ("XLA_FLAGS", "LIBTPU_INIT_ARGS")
    }


def _kept_inputs(lowered):
    return sorted(lowered._lowering.compile_args["kept_var_idx"])


def _signature(lowered, mesh, compiler_options):
    # Use the same location-stripping canonicalization as JAX's disk cache.
    # Executable serialization is version-specific; require the exact JAX stack.
    from jax._src import cache_key

    digest = hashlib.sha256()
    # JAX 0.11 uses a boolean here; 0.10 used an enum. Preserve callbacks.
    callbacks = cache_key.IgnoreCallbacks.NO if hasattr(cache_key, "IgnoreCallbacks") else False
    cache_key._hash_computation(digest, lowered.compiler_ir(), callbacks)
    infos = jax.tree_util.tree_leaves(lowered.args_info)
    return {
        "ir_sha256": digest.hexdigest(),
        "inputs": [
            {
                "shape": list(info.shape),
                "dtype": str(info.dtype),
                "weak_type": info._aval.weak_type,
                "donated": info.donated,
            }
            for info in (infos[i] for i in _kept_inputs(lowered))
        ],
        "mesh": dict(mesh.shape),
        "devices": [
            {
                "id": device.id,
                "kind": device.device_kind,
                "coords": list(getattr(device, "coords", ())),
                "core": getattr(device, "core_on_chip", None),
                "process": device.process_index,
            }
            for device in mesh.devices.flat
        ],
        "versions": _runtime_versions(),
        "flags": _compile_flags(),
        "compiler_options": compiler_options or {},
    }


def _key(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def save_executable(compiled, lowered, mesh, compiler_options, directory):
    """Save the binary, keeping Python model objects out of the artifact."""
    if compiled._params.const_args:
        raise ValueError("Executable export requires weights/constants to be explicit inputs")
    # Give the binary a flat ABI containing only inputs that survive JAX DCE.
    # Serving may carry extra unused metadata or empty weight placeholders;
    # those must not shift the executable's retained-input indices.
    kept = _kept_inputs(lowered)
    infos = jax.tree_util.tree_leaves(compiled.args_info)
    unloaded = replace(
        compiled._executable._unloaded_executable,
        kept_var_idx=set(range(len(kept))),
        all_args_info=None,
    )
    flat_compiled = jax.stages.Compiled(
        unloaded.load(), [], (tuple(infos[i] for i in kept), {}), compiled.out_tree
    )
    payload, _, _ = serialize_executable.serialize(flat_compiled)
    signature = _signature(lowered, mesh, compiler_options)
    metadata = {
        "schema_version": 1,
        "signature": signature,
        "key": _key(signature),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    directory = Path(directory)
    (directory / "executable.bin").write_bytes(payload)
    (directory / "executable.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class ExecutableStore:
    """Load matching trusted binaries; never compile on a missing artifact.

    A fresh lowering supplies current Python input/output trees and checks the
    complete program, including static values and constants. It does not invoke
    the backend compiler. The checked JAX callable enforces runtime layouts.
    """

    def __init__(self, directory, mesh):
        self.mesh = mesh
        self._entries = {}
        root = Path(directory)
        for path in sorted(root.rglob("executable.json")):
            metadata = json.loads(path.read_text())
            if metadata.get("schema_version") != 1:
                raise ValueError(f"Unsupported executable schema: {path}")
            if metadata["key"] != _key(metadata["signature"]):
                raise ValueError(f"Invalid executable signature: {path}")
            # Repeated exports can differ in debug metadata while sharing the
            # same program/ABI. Choose deterministically, without loading both.
            self._entries.setdefault(metadata["key"], (path.parent, metadata))
        if not self._entries:
            raise ValueError(f"No executables found in {directory}; export with --save-executable")

    def load(self, lowered, compiler_options=None):
        signature = _signature(lowered, self.mesh, compiler_options)
        key = _key(signature)
        entry = self._entries.get(key)
        if entry is None:
            nearest = min(
                self._entries.values(),
                key=lambda item: sum(
                    signature[k] != item[1]["signature"].get(k) for k in signature
                ),
            )[1]["signature"]
            differences = [k for k in signature if signature[k] != nearest.get(k)]
            raise ValueError(
                f"No matching AOT model executable ({key}); differing fields: {differences}. "
                "Export this serving workload/configuration with --save-executable. "
                "No backend compilation was attempted."
            )
        directory, metadata = entry
        payload = (directory / "executable.bin").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Executable checksum mismatch: {directory}")
        devices = list(self.mesh.devices.flat)
        started = time.monotonic()
        compiled = serialize_executable.deserialize_and_load(
            payload,
            jax.tree_util.tree_structure((tuple(range(len(_kept_inputs(lowered)))), {})),
            lowered.out_tree,
            backend=devices[0].client,
            execution_devices=devices,
        )
        logger.info("[aot-model] loaded %s in %.3fs", directory, time.monotonic() - started)
        return compiled
