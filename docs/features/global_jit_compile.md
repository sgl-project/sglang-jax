# Global JIT Compile

## Goals

1. To improve the performance, wrap forward, logits and sample with JIT and padding separately.
2. To avoid cache miss, we support precompile and padding.

## Design

### JIT Forward

The wrapped forward function is `jitted_run_model` and `jitted_sampler`, which are used by prefill and decode. The input parameter `forward_batch`, `logits_metadata` and `sampling_metadata` have to be registered as PyTrees. At the same time, the subclasses in it are required to register too. Besides, we use `nnx.split` and `nnx.merge` on model to keep satisfy the `jit` requirements.

Note: `return_logprob` is not supported in `jitted_sampler`, this feature with jit may be supported in the future.

### Precompile and padding

Cache miss is unacceptable because it results a few seconds to a several tens of seconds compile time. In order to improve the performance, precompile and padding are necessary. Precompile is executed before Scheduler's loop. The precompile includes prefill and decode phase. The former pads the input parameters with token_paddings and the latter pads them with bs_paddings. Both phases need to pad tokens(like input_ids, positions, etc.) and batch_size.

#### token padding

`--precompile-token-paddings` is used to configure the token padding list. Token padding uses a fixed batch_size to make a tradeoff between performance and precompile time. So the padding pair will be {bs = fixed_bs, num_tokens = token1}, {bs = fixed_bs, num_tokens = token2} and so on. The fixed batch_size is calculated through `get_max_padded_size()` which takes the `max_prefill_tokens`, `chunked_prefill_size` and `max_running_requests` into the consideration.

#### batch size padding

`--precompile-bs-paddings` is used to configure the batch size padding list. Decode padding pair likes {bs = bs1, num_tokens = bs1}, {bs = bs2, num_tokens = bs2} and so on.




## Implementation

### JIT Forward

#### How to register the custom class as a PyTree?

```python
# Register the ForwardBatch as PyTree. Here omit the registration of the subclasses in ForwardBatch.
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids [total_tokens]
    input_ids: jax.Array
    # The indices of requests in the req_to_token_pool
    req_pool_indices: jax.Array
    # The sequence length for each request [batch_size]
    seq_lens: jax.Array
    # Token position in kv cache
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None

    # KV cache
    token_to_kv_pool: KVCache = None
    attn_backend: AttentionBackend = None

    cache_loc: jax.Array = None

    # For extend
    extend_prefix_lens: Optional[jax.Array] = None
    extend_seq_lens: Optional[jax.Array] = None

    def tree_flatten(self):
        children = (
            self.input_ids,
            self.req_pool_indices,
            self.seq_lens,
            self.out_cache_loc,
            self.positions,
            self.extend_start_loc,
            self.token_to_kv_pool,
            self.attn_backend,
            self.cache_loc,
            self.extend_prefix_lens,
            self.extend_seq_lens,
        )

        aux_data = {
            "forward_mode": self.forward_mode,
            "batch_size": self.batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.forward_mode = aux_data["forward_mode"]
        obj.batch_size = aux_data["batch_size"]

        obj.input_ids = children[0]
        obj.req_pool_indices = children[1]
        obj.seq_lens = children[2]
        obj.out_cache_loc = children[3]
        obj.positions = children[4]
        obj.extend_start_loc = children[5]
        obj.token_to_kv_pool = children[6]
        obj.attn_backend = children[7]
        obj.cache_loc = children[8]
        obj.extend_prefix_lens = children[9]
        obj.extend_seq_lens = children[10]

        return obj
```

#### How to use `nnx.split` and `nnx.merge`?

```python
def initialize_jit(self):
    self.graphdef, self.state = nnx.split(self.model)

    @jax.jit
    def jitted_run_model(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model(*args)

    self.jitted_run_model = partial(jitted_run_model, self.graphdef, self.state)

# forward
result, layers_k, layers_v = self.jitted_run_model(
    forward_batch, logits_metadata
)
```

### Precompile and Padding

Note: Padding strategy used in precompile and real forward is the same. The former padding is implemented in `generate_model_worker_batch` and the latter is implemented in `get_model_worker_batch`.

#### Precompile

Run precompile before scheduler's loop. Generate the `ModelWorkerBatch` and call `forward_batch_generation` to reuse the current codes as much as possible.

#### Padding

As mentioned above, padding use {bs = *bs*, num_tokens = *token*}:
- extend: *bs* = fixed_bs, *token* is the first item in the token paddings list which is not less than len(fields), like len(input_ids)
- decode: *bs* (= *token*) is the the first item in the batch size paddings list which is not less than len(fields), like len(seq_lens)

The following fields are required to padding:
- input_ids/out_cache_loc/positions: pad its length to num_tokens
- cache_loc: pad its length to the product of batch_size * max_req_len
- req_pool_indices/seq_lens/req_pool_indices/extend_start_loc: pad its length to batch size
- extend_prefix_lens/extend_seq_lens: pad its length to batch size only for prefill


## Usage

### JIT Forward

JIT Forward is default to use.

### Precompile and padding

- `--precompile-token-paddings`: set like 8192 16384
- `--precompile-bs-paddings`: set like 1 10 32
- `--disable-jax-precompile`: default to False
- `--max-running-requests`
- `--chunked-prefill-size`
