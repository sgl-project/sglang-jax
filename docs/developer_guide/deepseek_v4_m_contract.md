# DeepSeek V4 M model interfaces

This model follows `epic/dsv4`'s Flash decoder and owns embeddings, attention projections, compressor and indexer parameters, mHC parameters, layer assembly, the LM head, and global checkpoint inventory. The shared config and layer classification live in `configs/deepseek_v4.py`.

## Forward boundaries

| Boundary | M sends | M receives |
| --- | --- | --- |
| B attention backend | Projected Q `[T, heads, head_dim]`, one shared KV `[T, head_dim]`, layer and batch metadata, compressor/indexer inputs where applicable | Per-head attention result `[T, heads, head_dim]` and functional pool updates |
| E MoE | Collapsed BF16 activation `[T, hidden]`, flattened token IDs `[T]`, valid-token mask `[T]`, and routing/output sharding | Routed-plus-shared output `[T, hidden]` and logical top-k expert IDs |
| H mHC | Residual streams `[T, hc_mult, hidden]` and FP32 gates/mixing parameters | Collapsed activation, post residual streams, and final head collapse |
| R model runner | Logits-processor output | M returns `(output, updates, True, ids)`; R commits the B-produced updates |

M calls `batch.attn_backend` once per layer and packs its updates. It never allocates or writes KV/compressor-state pages. The mHC sequence is attention `pre`, attention, `post`, activation-dtype cast, FFN `pre`, MoE, FFN `post`, activation-dtype cast; final head collapse precedes the final norm and LM head. The activation cast after each `pre` is before its norm. Sequence parallelism may shard token rows, while mHC parameters remain replicated.

## Checkpoint boundary

M obtains one complete `LocalSource.metadata` inventory, rejects duplicate/unknown/missing backbone keys, and excludes the documented MTP tail. It loads only M-owned tensors, including attention/indexer FP8 block scales and FP32 mHC parameters. Each E layer receives only its own MoE entries and reports its consumed keys. M requires the M and E reports to be disjoint and cover every required backbone key.

M borrows the framework's `_weight_source` when supplied and leaves its lifetime to the caller. Standalone loading creates a warmup-enabled `LocalSource` and closes it on success or failure. Inventory validation precedes prefetch and payload reads. M reads ordinary tensors through the source's tensor API and decodes E8M0 scales from owned byte ranges, rejecting reserved code 255. The E call retains `load_owned_weights(assigned, expert_format=...)`; sharing the source inside E requires its separate integration change.

An absent `sglang_jax_expert_format` marker selects E's MXFP4-to-FP8 path. The existing `sglang-jax-deepseek-v4-expert-fp8-per-channel-v1` marker selects E's static expert loader after M validates `static-fp8-complete.json`, config/index hashes, and shard coverage. `quantization_config.is_static_checkpoint` independently selects the resident FP8 linear representation. M does not export a checkpoint or convert routed experts.

The M model requires the E, H, and B implementations at runtime. The model import and registration are available independently; full forward and checkpoint loading require those dependency branches to be integrated.

## Post-load preparation and verification

Concrete loading materializes RoPE tables, grouped/fused output weights, and HCA fused projections under the model's concrete mesh, even when the caller has no ambient mesh. Dummy AOT loading uses abstract mesh contexts throughout initialization and preparation so `nnx.eval_shape` does not allocate device arrays. Dummy loading does not import E's checkpoint loader.

The CPU suite includes the config, inventory, and `test_deepseek_v4_m_loading.py` tests. Loading tests use explicit E/B boundary substitutes so this branch can verify source lifetime, ownership reports, BF16/FP8/E8M0 decoding, and concrete/abstract preparation before dependency branches merge. They do not establish full-model forward correctness or TPU kernel compatibility.

Source reuse and preparation behavior are adapted from PR #1789, including commit `9f99352e4a94179672c7c5328dad9f76cdf4f4a1`, within M's ownership boundary.
