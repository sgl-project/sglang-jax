import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from flax import nnx
from safetensors import safe_open
from wanvae import Encoder3d
from weight_mapping import to_mappings

path = "/models/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"


def get_transformer_output(path):
    vae = AutoencoderKLWan.from_pretrained(
        path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    # (1, 2, 3, 4, 16)
    latents_original = (
        torch.tensor(np.arange(1 * 1 * 3 * 4 * 3), dtype=torch.float32)
        .reshape(1, 1, 192, 192, 3)
        .permute((0, 4, 1, 2, 3))
    )
    y = vae.encode(latents_original)
    print(y.latent_dist.shape)
    return y.latent_dist.detach().numpy()


def get_weight_map(path):
    weights_files = glob.glob(os.path.join(path + "/vae", "*.safetensors"))
    if len(weights_files) == 0:
        raise RuntimeError(f"Cannot find any *.safetensors files in {path}")
    weights_files.sort()
    update_map = {}
    for st_file in weights_files:
        with (
            jax.default_device(jax.local_devices(backend="cpu")[0]),
            safe_open(st_file, framework="flax") as f,
        ):
            for name in f.keys():
                weight_tensor = f.get_tensor(name)
                update_map |= {name: weight_tensor}
    return update_map


def get_jax_output(path):
    model = Encoder3d(dim=96, z_dim=16, rngs=nnx.Rngs(0))
    graph_def, state  = nnx.split(model)
    weight_map = get_weight_map(path)
    flat_state = state.flat_state()
    for keys, v in flat_state:
        path = ".".join(str(key) for key in keys)
        mapped = False
        for src, (tgt, transpose_and_sharding) in to_mappings().items():
            regex = "^" + re.escape(tgt).replace("\\.\\*", r"\.(\d+)") + "$"
            matched = re.match(regex, path)
            if matched:
                # Extract wildcards if any
                wildcards = matched.groups()
                src_parts = []
                wc_index = 0
                for part in src.split("."):
                    if part == "*":
                        src_parts.append(wildcards[wc_index])
                        wc_index += 1
                    else:
                        src_parts.append(part)
                actual_src = ".".join(src_parts)
                # Check if this is a scanned parameter (has 'layer' in sharding spec)
                if transpose_and_sharding[0] is not None:
                    v.value = jnp.transpose(
                        jnp.array(weight_map[actual_src], jnp.float32), transpose_and_sharding[0]
                    )
                else:
                    v.value = jnp.array(weight_map[actual_src], jnp.float32)
                if matched:
                    mapped = True
                    break
        if not mapped:
            print(f"{path} not found")
    new_state = state.from_flat_path(flat_state)
    nnx.update(model, new_state)

    x = jnp.array(np.arange(1 * 1 * 192 * 192 * 3), dtype=jnp.float32).reshape(1, 1, 192, 192, 3)

    y, cache_list = model(x, cache_list=tuple([None] * 32), cache_idx=[0])
    print(y.shape)
    return y.transpose((0, 4, 1, 2, 3))


if __name__ == "__main__":
    transformer_y = get_transformer_output(path)
    jax_y = get_jax_output(path)
    print(transformer_y, '---', jax_y)
    print(np.allclose(transformer_y, jax_y, 1e-5, 1e-5))
