# DeepSeek V4 M/E boundary

This is the model-facing contract for the E-owned DeepSeek V4 Flash MoE layer
in `sgl_jax.srt.layers.deepseek_v4_moe`. It describes the current standalone E
implementation; complete model integration is M's work in [#1718](https://github.com/sgl-project/sglang-jax/issues/1718).
The E requirements are in [#1719](https://github.com/sgl-project/sglang-jax/issues/1719).

## Construction and forward

M constructs `DeepseekV4MoE(config, mesh, layer_id, dtype=jnp.bfloat16)` for
each backbone layer. The shared V4 configuration determines hash versus learned
routing, expert counts, top-k, the shared expert, clamped SwiGLU, and the
`quantization_config.is_static_checkpoint` shared-expert representation.
`layer_id` is not passed again during forward.

V4 enables the epic SparseCore permute path by default, with
`SGL_JAX_MOE_SC_PERMUTE=false` as an override. It also opts into the epic
sort-free permutation for short routing vectors. The generic EPMoE defaults
remain unchanged. CPU tests compare these permutations with stable sorting;
the v7x performance effect still requires TPU validation.

M passes the activation **after FFN normalization**, with shape
`[T, hidden_size]` and the selected model activation dtype. It passes flattened
integer token IDs and a boolean valid-token mask, both `[T]`, along with
`dispatch_info` and the agreed token-only `out_sharding` and
`output_sharding`. The hash layers require checkpoint-loaded `gate.tid2eid`;
learned-routing layers do not use token IDs for their gate choice. Masked rows,
and hash-routed rows with out-of-range token IDs, produce zero MoE output,
including when their input contains non-finite values.

Calling `layer(..., return_expert_ids=False)` returns routed-plus-shared output
of shape `[T, hidden_size]` in the input activation dtype. With
`return_expert_ids=True`, it returns `(output, logical_topk_ids)`, where IDs
have shape `[T, num_experts_per_tok]` and are the same routing choice used for
that output. `dispatch_info` may map those logical IDs to physical expert
slots inside E; M receives logical IDs and collects them across layers for
the model-to-runner `(output, updates, True, ids)` return tuple. E never takes
the four mHC residual streams and does not perform H's FFN-side `pre` or
`post` operations.

## Checkpoint handoff

M scans the safetensors inventory once and passes each E layer only its
assigned `dict[str, list[dict]]` to `layer.load_owned_weights(...)`. Each key
has exactly one entry with `file`, `dtype`, `shape`, `byte_offset`, and
`byte_size`. Offsets are absolute file offsets. E rejects missing, duplicate,
wrong-layer, and incompatible entries; it reads assigned payloads without
rescanning the safetensors headers. M owns the complete checkpoint partition,
unexpected-key and MTP accounting, format selection, and static-export
publication validation.

The current `WeightLoader.source.metadata` provides this entry schema. M can
partition that inventory and pass the E-owned subset directly, keeping one
checkpoint scan after #1716. E's current payload reader requires the `file`
entries to be locally readable paths.

The E-owned key families for layer `i` are:

| Source keys under `layers.i.ffn.` | Selection |
| --- | --- |
| `gate.weight` | Every layer |
| `gate.tid2eid` | Hash layers only |
| `gate.bias` | Learned-routing layers only |
| `shared_experts.w{1,2,3}.{weight,scale}` | Every layer |
| `experts.{expert_id}.w{1,2,3}.{weight,scale}` | Every logical routed expert |

M passes `expert_format=None` for original MXFP4 routed experts, or
`STATIC_EXPERT_FORMAT` for the existing published static expert-FP8 export.
This marker selects **routed-expert loading**. Separately,
`config.quantization_config.is_static_checkpoint` selects the shared-expert
FP8 linear representation before construction. The static expert format
requires that shared-expert representation; M must preserve and pass both
configuration fields instead of inferring one from the other. E loads routed
MXFP4 through strict bounded conversion or static per-channel FP8 directly;
both populate the same resident expert layout.

`load_owned_weights` returns `MoELoadReport`. Its `consumed_keys` is the exact
E-owned source-key partition that M must reconcile with its own consumed keys.
`local_payload_keys` identifies bytes read on this process; in EP, a process
may not read experts placed on other hosts. M checks that its and E's consumed
sets do not overlap and cover all required backbone tensors. The report also
records converted pair count, conversion error, and calculated peak converter
host allocation. It is not a measured process RSS or device-memory peak.

## Conversion memory evidence

Run the profile in a fresh process so the operating system's RSS high-water
mark starts before conversion. The script reads `resource.getrusage(...).ru_maxrss`
before and after one strict pair conversion and reports the difference. It also
prints E's calculated overlapping-array allocation account; neither number is
the peak of complete model loading or TPU HBM use.

```bash
PYTHONPATH=python python test/srt/quantization/profile_mxfp4_fp8_memory.py
```

On a local macOS CPU run with synthetic constant MXFP4 rows shaped like one
Flash `w1` expert (`[2048, 4096]`, 128 rows per chunk), conversion was exact
with maximum absolute reconstruction error 0. The process RSS high-water
increment was 14,024,704 bytes; the calculated converter allocation estimate was
25,976,832 bytes. These are fixture results, not measurements of a released
checkpoint. To profile an assigned real pair, pass `--weight-file`,
`--weight-name`, and, if necessary, `--scale-file` to the same script. Record
the checkpoint revision and hardware alongside those results.

The module-level CPU tests use synthetic checkpoints and two-device placement.
As a cross-implementation check, the pinned `epic/dsv4` exporter at
`ce1ebb637` converted a synthetic one-layer, two-expert original MXFP4
checkpoint into its published static format. E loaded all six routed pairs
directly from that export, reported all 20 assigned source keys as consumed,
and reproduced the known resident expert weights and scales on CPU. This
checks the original exporter-to-E handoff, not a released model artifact.
The `reference` backend also dequantizes the resident static shared-expert FP8
weights and block scales for an unfused numerical check; the TPU test exercises
the actual quantized shared projection and expert kernel. The actual M call,
real published checkpoint, TPU expert kernel, multi-host placement, and R's
v7x serving gate remain integration checks.
