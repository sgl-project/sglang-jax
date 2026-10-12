"""Real two-controller CPU loads and coordinated failures; no TPU or checkpoint download."""

import os
import socket
import subprocess
import sys
from pathlib import Path


def test_two_controllers(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="2")
    processes = []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank-{rank}.log").open("w")
            process = subprocess.Popen(
                [sys.executable, __file__, str(rank), str(port), str(tmp_path)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log))
        for process, _ in processes:
            process.wait(timeout=60)
        for rank, (process, log) in enumerate(processes):
            log.close()
            output = (tmp_path / f"rank-{rank}.log").read_text()
            assert process.returncode == 0, output
            assert "DISTRIBUTED_LOAD_PASS" in output, output
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()


def _worker(rank, port, root):
    from types import SimpleNamespace

    import jax
    import numpy as np
    from flax import nnx
    from jax.experimental import multihost_utils
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from safetensors.numpy import save_file

    from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec

    if sys.platform == "darwin":
        # Gloo's default hostname may be an unresolved macOS .local name.
        # Configure its real TCP transport explicitly for this loopback test.
        from functools import partial

        from jax._src.lib import _jax

        _jax.make_gloo_tcp_collectives = partial(
            _jax.make_gloo_tcp_collectives, hostname="127.0.0.1"
        )
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{port}", num_processes=2, process_id=rank
    )
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",), axis_types=(AxisType.Explicit,))
    local = root / f"checkpoint-{rank}"
    local.mkdir()
    weight = np.arange(128, dtype=np.float32).reshape(16, 8)
    filename = local / "model.safetensors"
    try:
        for case in ("success", "plan_mismatch", "missing_file", "bad_header"):
            save_file({"w": weight}, filename)
            if case == "bad_header" and rank == 0:
                filename.write_bytes(b"broken")
            model = nnx.Module()
            model.w = nnx.Param(
                jax.ShapeDtypeStruct(
                    (8, 16), np.float32, sharding=NamedSharding(mesh, P(None, "tensor"))
                )
            )
            loader = WeightLoader(model, SimpleNamespace(model_path=str(local)), mesh)
            if case == "missing_file":
                assert "w" in loader.metadata
                if rank == 1:
                    filename.unlink()
            multihost_utils.sync_global_devices(case)
            error = None
            try:
                with jax.set_mesh(mesh):
                    loader.load(
                        {
                            "w": WeightSpec(
                                "w",
                                transpose=True,
                                sharding=(
                                    ("tensor", None)
                                    if case == "plan_mismatch" and rank == 1
                                    else (None, "tensor")
                                ),
                            )
                        }
                    )
            except (RuntimeError, ValueError) as exc:
                error = str(exc)
            if case == "success":
                assert error is None, error
                for shard in model.w.value.addressable_shards:
                    np.testing.assert_array_equal(np.asarray(shard.data), weight.T[shard.index])
            else:
                assert error is not None, case
                assert (
                    "plans differ" if case == "plan_mismatch" else "failed on ranks"
                ) in error, error
        # One rank loses its P/D cache while the other retains it. Both must
        # follow the read path together; a local-only cache hit would deadlock.
        from sgl_jax.srt.model_loader.weights.loader import _PD_WEIGHT_CACHE

        os.environ["SGLANG_PD_WEIGHT_CACHE"] = "1"
        save_file({"e0": weight, "e1": weight + 1}, filename)
        for attempt in range(2):
            if attempt and rank == 1:
                _PD_WEIGHT_CACHE.clear()
            model = nnx.Module()
            model.w = nnx.Param(jax.ShapeDtypeStruct((2, 8, 16), np.float32))
            loader = WeightLoader(model, SimpleNamespace(model_path=str(local)), mesh)
            with jax.set_mesh(mesh):
                loader.load(
                    {
                        "experts": WeightSpec(
                            "w",
                            sources=("e0", "e1"),
                            transpose=True,
                            sharding=(None, None, "tensor"),
                        )
                    }
                )
            expected = np.stack((weight.T, (weight + 1).T))
            for shard in model.w.value.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
        print("DISTRIBUTED_LOAD_PASS", rank, flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
