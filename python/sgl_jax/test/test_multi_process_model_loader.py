import os

import jax
from flax import nnx
from flax.nnx.statelib import State
from jax.sharding import Mesh

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.loader import get_model_loader

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"


def print_sharding(params, prefix=""):
    if isinstance(params, (dict, State)):
        for k, v in params.items():
            print_sharding(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(params, (list, tuple)):
        for idx, v in enumerate(params):
            print_sharding(v, f"{prefix}[{idx}]")
    elif hasattr(params, "value"):
        v = params.value
        if isinstance(v, (dict, State, list, tuple)):
            print_sharding(v, prefix)
        else:
            if hasattr(v, "sharding"):
                print(f"[SHARDING] {prefix}: sharding={v.sharding}")
                for i, shard in enumerate(getattr(v, "addressable_shards", [])):
                    print(
                        f"    [SHARD] idx={i}, device={shard.device}, index={getattr(shard, 'index', None)}, shape={getattr(shard.data, 'shape', None)}"
                    )
            else:
                print(f"[SHARDING] {prefix}: not sharded, type={type(v)}")


def main():
    process_id = int(os.environ.get("PROCESS_ID", "0"))
    num_processes = int(os.environ.get("NUM_PROCESSES", "2"))
    coordinator_address = os.environ.get("COORDINATOR_ADDRESS", "localhost:12345")

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    print(f"Process {process_id} initialization completed, devices: {jax.devices()}")

    mesh = Mesh(jax.devices(), ("tensor",))
    print(
        f"Process {process_id} mesh: {mesh}, mesh shape: {mesh.shape}, mesh devices: {mesh.devices}"
    )

    model_path = os.environ.get("TEST_MODEL_PATH", "./test_models/your_model_dir")
    model_config = ModelConfig(
        model_path=model_path, trust_remote_code=True, dtype="bfloat16"
    )
    load_config = LoadConfig(load_format=LoadFormat.JAX)
    rng = nnx.Rngs(42)

    loader = get_model_loader(load_config, rng, mesh)
    print(f"Process {process_id} starting to load model weights...")
    loader.download_model(model_config)
    model = loader.load_model(model_config=model_config)
    print(f"Process {process_id} weight loading completed!")

    state = nnx.state(model)
    print(f"Process {process_id} weight sharding information:")
    print_sharding(state)


if __name__ == "__main__":
    main()
