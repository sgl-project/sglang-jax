#!/usr/bin/env python3
"""Launcher for running scripts on multi-process GKE TPU (e.g. TPU v7x-8).

Handles JAX distributed initialization and runs the target script on all
processes. ALL processes execute the same code path — this is required by JAX
for sharded computations.

Usage (from local machine, after pod is ready):

    # Copy to pod
    for C in <WORKLOAD>-1 <WORKLOAD>-2; do
      kubectl cp scripts/gke_tpu7x/launcher.py <POD>:/tmp/launcher.py -c $C
    done

    # Run on both containers simultaneously
    kubectl exec <POD> -c <WORKLOAD>-2 -- python3 -u /tmp/launcher.py <script> [args...] &
    kubectl exec <POD> -c <WORKLOAD>-1 -- python3 -u /tmp/launcher.py <script> [args...]

Examples:
    python3 -u /tmp/launcher.py benchmark/moe/bench_fused_moe.py \\
        --num-experts 8 --top-k 2 --hidden-size 2048 --intermediate-size 512 \\
        --num-tokens 64 128 --iters 3 --warmup-iters 1

    python3 -u /tmp/launcher.py python/sgl_jax/test/kernels/some_test.py
"""
import os
import runpy
import sys

# Set up import paths: repo root (for benchmark/) and python/ (for sgl_jax)
REPO_ROOT = "/tmp/sglang-jax"
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import jax

jax.distributed.initialize()
proc = jax.process_index()
print(
    f"[Process {proc}] JAX {jax.__version__}, "
    f"{jax.device_count()} devices, local {jax.local_device_count()}",
    flush=True,
)

if len(sys.argv) < 2:
    if proc == 0:
        print("Usage: python3 launcher.py <script.py> [args...]", flush=True)
    sys.exit(1)

# Set sys.argv to the target script + its args (all processes get the same argv)
script_path = os.path.join(REPO_ROOT, sys.argv[1])
sys.argv = [sys.argv[1]] + sys.argv[2:]

runpy.run_path(script_path, run_name="__main__")
