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

An absent `sglang_jax_expert_format` marker selects E's MXFP4-to-FP8 path. The existing `sglang-jax-deepseek-v4-expert-fp8-per-channel-v1` marker selects E's static expert loader after M validates `static-fp8-complete.json`, config/index hashes, and shard coverage. `quantization_config.is_static_checkpoint` independently selects the resident FP8 linear representation. M does not export a checkpoint or convert routed experts.

The M model requires the E, H, and B implementations at runtime. The model import and registration are available independently; full forward and checkpoint loading require those dependency branches to be integrated.
