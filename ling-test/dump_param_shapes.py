"""Diagnostic: dump Tunix LingModel and AInfer model param paths + shapes.

Runs on the TPU cluster. Uses nnx.eval_shape so no real weights are
materialized. Prints two flat path->shape tables so we can build/verify the
Tunix -> AInfer weight mapping against ground truth instead of guessing.
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx


def dump_state(label, model):
    print(f"\n===== {label} PARAM PATHS =====", flush=True)
    _, state = nnx.split(model)
    flat = state.flat_state()
    rows = []
    for keys, leaf in flat:
        path = ".".join(str(k) for k in keys)
        shape = getattr(getattr(leaf, "value", leaf), "shape", None)
        rows.append((path, shape))
    rows.sort()
    # Collapse layer indices to * for the first 2 layers to keep output short,
    # but print layer 0 and layer 1 fully (dense vs MoE differ).
    for path, shape in rows:
        # only print layers 0,1 + non-layer params to keep concise
        if ".layers." in path:
            seg = path.split(".layers.")[1].split(".")[0]
            if seg not in ("0", "1"):
                continue
        print(f"  {path}\t{shape}", flush=True)
    print(f"  (total {len(rows)} leaves)", flush=True)


def dump_tunix():
    from tunix.models.ling import model as ling_model
    cfg = ling_model.ModelConfig.ling_mini_2p0()
    # shrink layers for speed: eval_shape is cheap but model build loops layers
    mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
    def build():
        rngs = nnx.Rngs(0)
        return ling_model.LingModel(cfg, rngs=rngs)
    with mesh:
        m = nnx.eval_shape(build)
    dump_state("TUNIX LingModel", m)


def dump_ainfer():
    import sys
    sys.path.insert(0, "/app/ainfer")
    from ainfer.models.tpu import TpuModelRegistry
    from ainfer.configs.model_config import ModelConfig as AInferModelConfig
    # Build AInfer model config from HF dir
    hf_dir = "/tmp/models/inclusionAI/Ling-mini-2.0"
    # AInfer worker builds via model_config.hf_config; replicate minimally.
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    arch = hf_cfg.architectures[0]
    model_cls = TpuModelRegistry.resolve(arch)
    mesh = jax.make_mesh((1, len(jax.devices())), ("data", "tp"))

    class _Wrap:
        pass
    # AInfer model_cls signature: model_cls(self.model_config, dtype=, mesh=)
    # self.model_config is AInfer ModelConfig wrapping hf_config.
    # Try to construct via AInfer's ModelConfig.
    mc = AInferModelConfig(model=hf_dir, trust_remote_code=True)
    with mesh:
        m = nnx.eval_shape(lambda: model_cls(mc, dtype=jnp.bfloat16, mesh=mesh))
    dump_state("AINFER model", m)


if __name__ == "__main__":
    print("devices:", len(jax.devices()), flush=True)
    which = os.environ.get("DUMP_WHICH", "both")
    if which in ("tunix", "both"):
        try:
            dump_tunix()
        except Exception as e:
            import traceback
            print("TUNIX DUMP FAILED:", e, flush=True)
            traceback.print_exc()
    if which in ("ainfer", "both"):
        try:
            dump_ainfer()
        except Exception as e:
            import traceback
            print("AINFER DUMP FAILED:", e, flush=True)
            traceback.print_exc()
    print("DUMP_DONE", flush=True)
