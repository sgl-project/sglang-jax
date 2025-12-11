# Global JIT Compile

## Goals

1. To improve the performance, wrap forward, logits and sample with JIT.
2. To avoid cache miss, we support precompile and padding.

## Design

### JIT Forward

The wrapped forward function is `jitted_run_model` and `jitted_sampler`, which are used in prefill and decode. The input parameter `forward_batch`, `logits_metadata`, `logits_output` and `sampling_metadata` have to be registered as PyTrees. At the same time, the subclasses in it are required to register too. Besides, we use `nnx.split` and `nnx.merge` on model to keep satisfy the `jit` requirements.

Note: `return_logprob` is supported in `jitted_run_model` and `jitted_sampler` with performance degradation, It will be optimized later. However, if you only specify `return_logprob` and only need the log probabilities of the output tokens, it is optimized.

### Precompile and padding

Cache miss is unacceptable because it results in a few seconds to a several tens of seconds compile time. In order to improve the performance, precompile and padding are necessary. Precompile is executed before Scheduler's loop. The precompile includes prefill and decode phase. Both phases need to pad tokens(like input_ids, positions, etc.) and batch_size.

#### token padding

`--precompile-token-paddings` is used to configure the token padding list. Token padding uses a fixed batch_size to make a tradeoff between performance and precompile time. So the padding pair will be {bs = fixed_bs, num_tokens = token1}, {bs = fixed_bs, num_tokens = token2} and so on. The fixed batch_size is calculated through `get_max_padded_size()` which takes the `max_prefill_tokens`, `chunked_prefill_size` and `max_running_requests` into the consideration.

#### batch size padding

`--precompile-bs-paddings` is used to configure the batch size padding list. Decode padding pair likes {bs = bs1, num_tokens = bs1}, {bs = bs2, num_tokens = bs2} and so on.




## Implementation

Note: Padding strategy used in precompile and real forward is the same. The former padding is implemented in `generate_model_worker_batch` and the latter is implemented in `get_model_worker_batch`.

### Precompile

Run precompile before scheduler's loop. Generate the `ModelWorkerBatch` and call `forward_batch_generation` to reuse the current codes as much as possible.

### Padding

As mentioned above, padding use {bs = *bs*, num_tokens = *token*}:
- extend: *bs* = fixed_bs, *token* is the first item in the token paddings list which is not less than len(field), like len(input_ids)
- decode: *bs* (= *token*) is the the first item in the batch size paddings list which is not less than len(field), like len(seq_lens)

The following fields are required to padding:
- input_ids/out_cache_loc/positions: pad its length to num_tokens
- cache_loc: pad its length to the product of batch_size * max_req_len
- req_pool_indices/seq_lens/req_pool_indices/extend_start_loc: pad its length to batch size
- extend_prefix_lens/extend_seq_lens: pad its length to batch size only for prefill


## Usage

### JIT Forward

JIT Forward is default to use.

### Precompile and padding

- `--precompile-token-paddings`: default values is recommended, but you still can set like 8192 16384
- `--precompile-bs-paddings`: default values is recommended, but you still can set like 1 10 32
- `--disable-precompile`: default to False
- `--max-running-requests`
- `--chunked-prefill-size`
