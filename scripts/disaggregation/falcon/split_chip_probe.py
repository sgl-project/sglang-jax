"""Cross-process Raiden page integrity probe on disjoint chip groups."""

import argparse
import json
import os
import time
from pathlib import Path

from sgl_jax.raiden import preload_raiden

preload_raiden()  # The native extension must load before JAX.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper  # noqa: E402


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(path)


def read(path):
    deadline = time.monotonic() + 120
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.05)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["producer", "consumer"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    assert len(devices) == 4 and jax.process_count() == 1, devices
    write(
        out / f"{args.role}-devices.json",
        {
            "devices": list(map(str, devices)),
            "visible_chips": os.environ["TPU_VISIBLE_CHIPS"],
            "jax": jax.__version__,
        },
    )
    shape = (16, 128, 4, 2, 128)
    mesh = Mesh(np.array(devices), ("tensor",))
    sharding = NamedSharding(mesh, PartitionSpec(None, None, "tensor", None, None))
    # BF16 exactly representable values, changing across page and position.
    raw = np.arange(np.prod(shape)).reshape(shape) % 97 + np.arange(16).reshape(16, 1, 1, 1, 1)
    patterns = [jnp.asarray(raw + layer * 32, dtype=jnp.bfloat16) for layer in range(2)]
    caches = [
        jax.device_put(x if args.role == "producer" else jnp.zeros(shape, jnp.bfloat16), sharding)
        for x in patterns
    ]
    for cache in caches:
        cache.block_until_ready()
    wrapper = RaidenTransferWrapper("127.0.0.1")  # native endpoint chooses routable host itself
    wrapper.start(caches, max_blocks=16, num_slots=2, timeout_s=60)
    expected = [np.zeros(shape, dtype=np.float32) for _ in caches]
    cases = [([0, 1], [2, 3]), ([5, 2, 8], [1, 7, 4]), ([15], [0]), ([6, 9], [2, 3])] * 3
    for i, (src, dst) in enumerate(cases):
        name, uid = f"probe-{i}", 1000 + i
        if args.role == "producer":
            assert wrapper.register_read(name, uid, src)
            write(out / f"ready-{i}.json", wrapper.endpoints)
        else:
            endpoints = read(out / f"ready-{i}.json")
            remote = endpoints[0]["endpoint"] if len(endpoints) == 1 else endpoints
            wrapper.start_read(name, uid, remote, src, dst)
        deadline = time.monotonic() + 90
        while True:
            sent, received, failed = wrapper.poll_stats()
            assert not failed, failed
            if name in (sent if args.role == "producer" else received):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(name)
            time.sleep(0.01)
        if args.role == "consumer":
            for layer, cache in enumerate(caches):
                expected[layer][dst] = np.asarray(patterns[layer], dtype=np.float32)[src]
                # Force a fresh device computation; native writes are external to JAX.
                actual = np.asarray(jax.jit(lambda x: x.astype(jnp.float32) + 0)(cache))
                np.testing.assert_array_equal(actual, expected[layer])
            write(out / f"ack-{i}.json", {"passed": True})
        else:
            read(out / f"ack-{i}.json")
    write(
        out / f"{args.role}-result.json",
        {
            "status": "passed",
            "rounds": len(cases),
            "layers": 2,
            "shape": shape,
            "scope": "disjoint-chip cross-process BF16 full-page integrity and slot reuse",
        },
    )
    print(f"PROBE_PASSED role={args.role} rounds={len(cases)}", flush=True)


if __name__ == "__main__":
    main()
