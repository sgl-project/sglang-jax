import itertools

import flax.nnx
import jax.tree_util
import numpy as np
from jax import numpy as jnp
from jax._src.tree_util import GetAttrKey
from transformers import PretrainedConfig

from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.grok import Grok1ModelForCausalLM
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

config = PretrainedConfig(
    **{
        "architectures": ["Grok1ForCausalLM"],
        "embedding_multiplier_scale": 90.50966799187809,
        "output_multiplier_scale": 0.5,
        "vocab_size": 131072,
        "hidden_size": 8192,
        "intermediate_size": 1024,
        "moe_intermediate_size": 2048,
        "max_position_embeddings": 131072,
        "num_experts_per_tok": 2,
        "num_local_experts": 8,
        "residual_moe": True,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "head_dim": 128,
        "rms_norm_eps": 1e-05,
        "final_logit_softcapping": 50,
        "attn_logit_softcapping": 30.0,
        "router_logit_softcapping": 30.0,
        "rope_theta": 208533496,
        "attn_temperature_len": 1024,
        "sliding_window_size": -1,
        "global_attn_every_n": 1,
        "model_type": "git",
        "torch_dtype": "bfloat16",
        "rope_type": "original",
        "original_max_position_embeddings": 8192,
        "scaling_factor": 16.0,
        "extrapolation_factor": 1.0,
        "attn_factor": 1.0,
        "beta_fast": 8,
        "beta_slow": 1,
    }
)


def main():
    np.random.seed(0)

    text_ids = [[1, 4, 2, 3]]

    input_ids = jnp.asarray(list(itertools.chain(*text_ids)), dtype=jnp.int32)
    seq_lens = jnp.asarray([len(e) for e in text_ids], dtype=jnp.int32)
    positions = jnp.asarray(
        list(itertools.chain(*[range(len(e)) for e in text_ids])), dtype=jnp.int32
    )

    # exclusive cumsum
    extend_start_loc = jnp.cumsum(seq_lens, axis=0) - seq_lens

    mesh = create_device_mesh(
        ici_parallelism=[-1, 1, 1],
        dcn_parallelism=[1, 1, 1],
        devices=[jax.devices()[0]],
    )

    model = Grok1ModelForCausalLM(
        config=config, rngs=flax.nnx.Rngs(default=0), mesh=mesh
    )
    model = model.model
    param_dict = dict()
    for name, param in jax.tree_util.tree_leaves_with_path(flax.nnx.state(model)):
        if param.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
            name = ".".join(
                [str(e)[1:] if isinstance(e, GetAttrKey) else str(e.key) for e in name]
            )
            param_dict[name] = param
        else:
            raise ValueError("Cannot random initialize non-floating parameters")

    for name, param in sorted(param_dict.items()):
        if "cos_sin_cache" in name:
            continue
        data = np.random.randn(*param.shape)
        param.value = jnp.asarray(data, dtype=param.dtype)
        print(name, data.sum())

    cache_pool = MHATokenToKVPool(
        size=128,
        page_size=1,
        head_num=config.num_key_value_heads,
        head_dim=config.head_dim,
        layer_num=config.num_hidden_layers,
        dtype=jnp.bfloat16,
        mesh=mesh,
    )
    out_cache_loc = jnp.arange(len(seq_lens), dtype=jnp.int32)
    cache_loc = jnp.arange(sum(seq_lens), dtype=jnp.int32)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        out_cache_loc=out_cache_loc,
        extend_start_loc=extend_start_loc,
        token_to_kv_pool=cache_pool,
        attn_backend=NativeAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
        ),
        cache_loc=cache_loc,
        bid=None,
        req_pool_indices=None,
        extend_prefix_lens=jnp.array([0]),
        extend_seq_lens=jnp.array([sum(seq_lens)]),
    )
    with jax.sharding.use_mesh(mesh):
        outputs = model(batch.input_ids, batch.positions, batch)
    print(outputs)


if __name__ == "__main__":
    main()
