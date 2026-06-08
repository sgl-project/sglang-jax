# Final report

## Summary

Simulated adding `python/sgl_jax/srt/layers/attention/block_sparse_attention_backend.py` and adding a selector entry so `attention_backend="block_sparse"` can choose it.

## Documentation impact

Updated `wiki/docs/projects/sglang-jax/06-layers-and-attention.md` in the Attention Backend section. The new text describes block sparse attention as a faster backend for long context workloads because it skips zero or unused blocks, reducing attention cost compared with dense attention.

I also updated the backend list in `wiki/docs/projects/sglang-jax/04-model-executor.md` because the model runner creates the attention backend during initialization.

## Changed files

- `sglang-jax/python/sgl_jax/srt/layers/attention/block_sparse_attention_backend.py` - new backend implementation.
- `sglang-jax/python/sgl_jax/srt/layers/attention/utils.py` - selector entry for block sparse attention.
- `wiki/docs/projects/sglang-jax/06-layers-and-attention.md` - added the new backend to the list.
- `wiki/docs/projects/sglang-jax/04-model-executor.md` - added the backend to the initialization table.

## Validation

I did not run the full test suite. I reviewed the selector and docs manually and confirmed the new backend name is consistent across the implementation and docs.
