## Goals
SGLang JAX supports multiple attention backends. Each of them has different pros and cons. You can test them according to your needs.

## Design

SGLang JAX adopts a plugin-based design pattern for Attention Backends, enabling developers to flexibly add and choose different attention mechanism implementations. The core design principles include:

### Abstract Base Class Design
All attention backends inherit from the `AttentionBackend` base class (`python/sgl_jax/srt/layers/attention/base_attn_backend.py`), which defines two core abstract methods:

- `init_forward_metadata(forward_batch)`: Initialize metadata required for forward pass
- `__call__(q, k, v, layer, forward_batch, **kwargs)`: Execute the actual attention computation

### Unified Interface Design
All backends follow the same input/output interface:
- **Input**: Query, Key, Value, RadixAttention layer object, ForwardBatch batch information
- **Output**: Attention output array and updated KV cache

### Metadata Management
Each backend can define its own specific metadata structure to optimize computation, for example:
- FlashAttention uses `FlashAttentionMetadata` to manage paged indices and sequence length information
- Native backend requires no additional metadata

## Implementation

To implement a new attention backend, follow these steps:

### 1. Inherit from AttentionBackend Base Class

```python
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend

class YourCustomAttention(AttentionBackend):
    def __init__(self, num_attn_heads, num_kv_heads, **kwargs):
        # Initialize your backend-specific parameters
        pass
```

### 2. Implement Required Methods

#### ```init_forward_metadata``` Method
```python
def init_forward_metadata(self, forward_batch: ForwardBatch):
    """Initialize forward pass metadata"""
    # Create your required metadata based on forward_batch
    # e.g., compute sequence lengths, create indices, etc.
    pass
```

#### ```__call__``` Method
```python
def __call__(self, q, k, v, layer, forward_batch, **kwargs):
    """Execute attention computation"""
    # 1. Get and update KV cache
    k_buffer, v_buffer = self._get_and_set_kv_cache(k, v, forward_batch, layer.layer_id)

    # 2. Execute your attention computation logic
    attn_output = your_attention_implementation(q, k_buffer, v_buffer, ...)

    # 3. Return results
    return attn_output, k_buffer, v_buffer
```

### 3. PyTree Support

To support JAX JIT compilation, implement PyTree support:

```python
def tree_flatten(self):
    children = (self.forward_metadata,)  # Mutable state
    aux_data = {
        "num_heads": self.num_heads,
        "param1": self.param1,
        # Other immutable parameters
    }
    return (children, aux_data)

@classmethod
def tree_unflatten(cls, aux_data, children):
    obj = cls(**aux_data)
    obj.forward_metadata = children[0]
    return obj
```

### 4. Import Custom Backend
Please import you backend in ```_get_attention_backend``` func, see in ```python/sgl_jax/srt/model_executor/model_runner.py```

### 5. Pallas Kernel
Please put you pallas kernel file in ```python/sgl_jax/srt/layers/attention``` folder (see flash attention implement). The implementation of the Pallas kernel should be designed with a single TPU in mind, so it is not advisable to incorporate distributed logic into the kernel. When using you pallas kernel in distributed scenario, please use ```shard_map``` to wrap you kernel call in you custom attention backend implement.

### 6. Test and Benchmark
Test and benchmark is important and required. Please add benchmark in ```benchmark/kernels``` folder and test in ```python/sgl_jax/test``` folder.

## Usage Guide
### Supporting matrix for different attention backends

| Backend | Paged Attention | Spec Decoding | MLA | Sliding Window |
|---------|----------------|---------------|-----|----------------|
| FlashAttention | ✅ | ❌ | ❌ | ✅ |
| NativeAttention | ❌ | ❌ | ❌ | ❌ |

### Launch command for different attention backends

#### FlashAttention
Recommended for production environments, memory efficient, supports long sequences
```
 python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code --device=tpu --attention-backend=fa
```

#### NativeAttention
Recommended for debugging and development, simple and straightforward logic
```
 python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code --device=tpu --attention-backend=native
```
