# test_vision_attention.py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import KimiK25VisionAttention
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig

def test_attention_layer():
    print(f"JAX devices: {jax.devices()}")
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("tensor",))

    config = KimiK25ModelVitConfig
    config.vt_hidden_size = 1152
    config.vt_num_attention_heads = 16
    
    # Initialize model on mesh
    with jax.set_mesh(mesh):
        attention = KimiK25VisionAttention(config, dtype=jnp.bfloat16, mesh=mesh)

    # Synthetic inputs
    batch_size = 1
    seq_len = 512 # Multiple of 256 to avoid padding logic for simplicity
    hidden_size = config.vt_hidden_size
    
    # hidden_states: [S, D]
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    hidden_states = jax.random.normal(k1, (seq_len, hidden_size), dtype=jnp.bfloat16)
    
    # cu_seqlens for varlen (just single sequence of seq_len)
    cu_seqlens = jnp.array([0, seq_len], dtype=jnp.int32)
    
    head_dim = hidden_size // config.vt_num_attention_heads
    pos_emb_shape = (2, seq_len, head_dim // 2)
    position_embeddings = jax.random.normal(k2, pos_emb_shape, dtype=jnp.bfloat16)

    # Run attention
    print("Running KimiK25VisionAttention...")
    output = attention(hidden_states, cu_seqlens, position_embeddings)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (seq_len, hidden_size), f"Expected {(seq_len, hidden_size)}, got {output.shape}"
    assert not jnp.any(jnp.isnan(output)), "Output contains NaNs"
    assert not jnp.any(jnp.isinf(output)), "Output contains Infs"
    print("SUCCESS: KimiK25VisionAttention test passed!")

if __name__ == "__main__":
    test_attention_layer()
