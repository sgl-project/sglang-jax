import sys
import flax.nnx
import jax
from jax import numpy as jnp
import torch
import numpy as np
from sglang.srt.models.grok import Grok1ForCausalLM as STDModel, Grok1Attention as STDAttention
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sgl_jax.srt.models.grok import Grok1ForCausalLM as SRCModel, Grok1Attention as SRCAttention
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from jax.sharding import Mesh

torch.manual_seed(0)
jax.config.update("jax_default_prng_impl", "unsafe_rbg")

init_distributed_environment(rank=0)
initialize_model_parallel()

config = ModelConfig(model_path="./weights")

# Create models
std_model = STDModel(config=config.hf_config)
src_model = SRCModel(config=config.hf_config, rngs=flax.nnx.Rngs(default=0))

# Extract attention layers
std_attention: STDAttention = std_model.model.layers[0].self_attn.cuda()
src_attention: SRCAttention = src_model.model.layers[0].self_attn

print("Standard attention config:")
print(f"  hidden_size: {std_attention.hidden_size}")
print(f"  num_heads: {std_attention.num_heads}")
print(f"  num_kv_heads: {std_attention.num_kv_heads}")
print(f"  head_dim: {std_attention.head_dim}")
print(f"  q_size: {std_attention.q_size}")
print(f"  kv_size: {std_attention.kv_size}")

print("\nJAX attention config:")
print(f"  hidden_size: {src_attention.hidden_size}")
print(f"  num_heads: {src_attention.num_heads}")
print(f"  num_kv_heads: {src_attention.num_kv_heads}")
print(f"  head_dim: {src_attention.head_dim}")
print(f"  q_size: {src_attention.q_size}")
print(f"  kv_size: {src_attention.kv_size}")

print(f"\nConfig values:")
print(f"  config.num_attention_heads: {config.hf_config.num_attention_heads}")
print(f"  config.num_key_value_heads: {config.hf_config.num_key_value_heads}")
print(f"  config.head_dim: {config.hf_config.head_dim}")

# Initialize weights to match between models
print("\nSynchronizing weights...")

# QKV projection weights
std_attention.qkv_proj.weight[:] = torch.randn_like(std_attention.qkv_proj.weight) * 0.01
# Convert torch weights to bfloat16 to match input dtype
std_attention.qkv_proj.weight.data = std_attention.qkv_proj.weight.data.to(torch.bfloat16)
src_attention.qkv_proj.weight.value = std_attention.qkv_proj.weight.cpu().float().numpy().astype(np.float32)

# Output projection weights  
std_attention.o_proj.weight[:] = torch.randn_like(std_attention.o_proj.weight) * 0.01
# Convert torch weights to bfloat16 to match input dtype
std_attention.o_proj.weight.data = std_attention.o_proj.weight.data.to(torch.bfloat16)
src_attention.o_proj.weight.value = std_attention.o_proj.weight.cpu().float().numpy().astype(np.float32)

print(f"QKV proj weight shape: {std_attention.qkv_proj.weight.shape}")
print(f"O proj weight shape: {std_attention.o_proj.weight.shape}")

# Create test input
seq_len = 4  # Keep it small for testing
batch_size = 1
hidden_size = config.hf_config.hidden_size

std_input = torch.randn([batch_size, seq_len, hidden_size]) * 0.01
std_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

src_input = jnp.asarray(std_input.cpu().numpy().reshape(-1, hidden_size), dtype=jnp.bfloat16)  # [total_tokens, hidden_size]
src_positions = jnp.asarray(std_positions.cpu().numpy().reshape(-1))  # [total_tokens]

print(f"\nInput shapes:")
print(f"  std_input: {std_input.shape}")
print(f"  src_input: {src_input.shape}")
print(f"  std_positions: {std_positions.shape}")
print(f"  src_positions: {src_positions.shape}")

# Create a simple ForwardBatch for JAX version
forward_batch = ForwardBatch(
    bid=0,
    forward_mode=ForwardMode.EXTEND,
    batch_size=batch_size,
    input_ids=jnp.zeros((seq_len,), dtype=jnp.int32),
    req_pool_indices=jnp.zeros((batch_size,), dtype=jnp.int32),
    seq_lens=jnp.array([seq_len]),
    out_cache_loc=jnp.arange(seq_len),
    positions=src_positions,
    extend_start_loc=jnp.array([0]),
    token_to_kv_pool=None,  # Will be set by attention backend
    attn_backend=None,  # Will be set by attention backend
    cache_loc=jnp.arange(seq_len),
    extend_prefix_lens=jnp.array([0]),
    extend_seq_lens=jnp.array([seq_len]),
)

print("\nTesting torch QKV and O projections only...")
# Since torch attention needs a complex forward_batch setup, let's test the components we can
with torch.no_grad():
    # Test QKV projection
    std_input_flat = std_input.cuda().to(torch.bfloat16).reshape(-1, hidden_size)
    std_qkv, _ = std_attention.qkv_proj(std_input_flat)
    print(f"Torch QKV shape: {std_qkv.shape}")
    print(f"Torch QKV sample: {std_qkv.float()[0, :5]}")
    
    # Test output projection with a dummy input
    dummy_attn_output = torch.randn_like(std_qkv[:, :std_attention.q_size]).to(torch.bfloat16)
    std_o_output, _ = std_attention.o_proj(dummy_attn_output)
    print(f"Torch O proj output shape: {std_o_output.shape}")
    print(f"Torch O proj sample: {std_o_output.float()[:3]}")

print("\nSkipping full torch attention due to forward_batch complexity...")

print("\nRunning JAX attention...")
# Create a mesh that avoids shard_map manual axis issues
devices = jax.devices()
print(f"Available devices: {devices}")

# For testing, use a simple approach that works with the cache system
if len(devices) >= 2:
    # Multi-device setup
    mesh = Mesh(devices[:2], ('data', 'tensor'))
else:
    # Single device - use None partitioning to avoid manual axis issues
    mesh = Mesh(devices, ('data',))
    print("Using single device mesh without tensor axis to avoid sharding issues...")

# Create minimal cache structure with proper mesh
req_to_token_pool = ReqToTokenPool(
    size=100,
    max_context_len=1024,
)

# Create a custom pool class for single device that doesn't use tensor axis
if len(devices) == 1:
    # Create a patched version of MHATokenToKVPool for single device
    class SingleDeviceMHATokenToKVPool(MHATokenToKVPool):
        def __init__(self, *args, **kwargs):
            # Temporarily override the partition axis before initialization
            super().__init__(*args, **kwargs)
            
        def _create_buffers(self):
            """Override to use None partitioning for single device"""
            from jax.sharding import NamedSharding, PartitionSpec as P
            self.kv_partition_axis = None
            self.kv_sharding = NamedSharding(self.mesh, P(None, None))
            
            import time
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Creating fused KV buffers for {self.layer_num} layers")
            start_time = time.time()

            # Use the same shape calculation as original but without sharding
            fused_buffer_shape = (
                self.size + self.page_size,
                self.head_num * 2,  # [K0,V0,K1,V1,...]  
                self.head_dim,
            )
            
            # For single device, tensor_size = 1 (no sharding)
            # Now self.head_num is correctly num_kv_heads, so the calculation should work
            tensor_size = 1
            local_shape = (
                fused_buffer_shape[0],
                fused_buffer_shape[1] // tensor_size,  # num_kv_heads * 2 sharded (no-op for single device)
                fused_buffer_shape[2],  # head_dim not sharded
            )

            logger.info(
                f"Total fused KV cache memory per layer: {local_shape[0] * local_shape[1] * local_shape[2] / 1024**3:.2f} GB, dtype: {self.dtype}"
            )

            # Create buffers with correct local shape
            self.kv_buffer = []
            for _ in range(self.layer_num):
                kv_buf = jnp.zeros(local_shape, dtype=self.dtype)
                self.kv_buffer.append(kv_buf)

            end_time = time.time()
            logger.info(f"Created fused KV buffers in {end_time - start_time:.2f}s")
    
    token_to_kv_pool = SingleDeviceMHATokenToKVPool(
        size=1000,
        page_size=1,
        dtype=jnp.bfloat16,
        head_num=config.hf_config.num_key_value_heads,  # Use num_kv_heads, not num_attention_heads
        head_dim=config.hf_config.head_dim,
        layer_num=config.hf_config.num_hidden_layers,
        mesh=mesh,
    )
else:
    token_to_kv_pool = MHATokenToKVPool(
        size=1000,
        page_size=1,
        dtype=jnp.bfloat16,
        head_num=config.hf_config.num_key_value_heads,  # Use num_kv_heads, not num_attention_heads
        head_dim=config.hf_config.head_dim,
        layer_num=config.hf_config.num_hidden_layers,
        mesh=mesh,
    )

attn_backend = NativeAttention(
    num_attn_heads=config.hf_config.num_attention_heads,
    num_kv_heads=config.hf_config.num_key_value_heads,
)

# Update forward_batch with cache info
forward_batch.token_to_kv_pool = token_to_kv_pool
forward_batch.attn_backend = attn_backend

src_output = src_attention(
    positions=src_positions,
    hidden_states=src_input,
    forward_batch=forward_batch,
)
print(f"JAX output shape: {src_output.shape}")
print(f"JAX output sample: {src_output[:3]}")

# Since we can't easily run full torch attention, we'll focus on JAX validation
print(f"\nJAX attention completed successfully!")
print(f"  Output shape: {src_output.shape}")
print(f"  Output dtype: {src_output.dtype}")
print(f"  Sample values: {src_output[:3]}")

print("\nTesting individual components...")

# Test QKV projection comparison
print("\nComparing QKV projections...")
qkv_jax, _ = src_attention.qkv_proj(src_input)
print(f"JAX QKV shape: {qkv_jax.shape}")
print(f"JAX QKV sample: {qkv_jax[0, :5]}")

# Compare with the torch QKV we computed earlier
print(f"Torch QKV shape: {std_qkv.shape}")
print(f"Torch QKV sample: {std_qkv.float()[0, :5]}")

diff = np.abs(std_qkv.float().cpu().numpy() - np.array(qkv_jax.astype(jnp.float32)))
print(f"QKV difference - max: {np.max(diff):.6f}, mean: {np.mean(diff):.6f}")
print(f"QKV outputs close? {np.allclose(std_qkv.float().cpu().numpy(), qkv_jax.astype(jnp.float32), atol=1e-2)}")

# Test rotary embedding
print("\nTesting Rotary Embedding...")
q, k, v = qkv_jax.split([src_attention.q_size, src_attention.kv_size, src_attention.kv_size], axis=-1)
print(f"Split shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

q_rot, k_rot = src_attention.rotary_emb(src_positions, q, k)
print(f"After RoPE - Q: {q_rot.shape}, K: {k_rot.shape}")
print(f"Q sample after RoPE: {q_rot[0, :5]}")

print("\nTest completed!") 