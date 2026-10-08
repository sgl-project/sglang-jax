# Weight loading

Models declare checkpoint names and final parameter layouts with `WeightSpec`.
`WeightLoader` validates the declarations, reads addressable shards, runs grouped
conversions, and assigns the results. Local safetensors and RunAI use this same
execution path.

```python
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec

WeightLoader(self, model_config, self.mesh).load({
    "model.layers.*.mlp.gate_proj.weight": WeightSpec(
        "model.layers.*.mlp.gate_proj.weight",
        sharding=(None, "tensor"),
        transpose=True,
    ),
})
```

The mapping key is the checkpoint name. Wildcards expand against checkpoint
metadata; the corresponding captures fill target paths. Sharding describes the
output axes. If omitted, it comes from the prepared parameter. `transpose_axes`,
`reshape`, `repeat`, `pad_width`, and `concat_axis` cover ordinary layout changes.
For fused visual QKV, declare `split_sizes` explicitly so vision and text towers
can use different head configurations.

## Interfaces and implementations

`WeightSource` is the storage interface. `LocalSource` handles safetensors files
and `RunaiWeightSource` handles streamed byte ranges. Both expose metadata,
checkpoint identity, tensor slices, bulk ranges, and session lifetime. Consumers
do not use file handles or SDK objects. `prefetch` and `release` have default
no-op implementations; `retains_views` tells the loader to wait for outstanding
transfers before releasing completed file mappings. RunAI owns its returned
buffers and does not add those file-release waits.

`WeightReader` is the device materialization interface. `JaxShardReader` implements
`read` for ordinary, split and stacked expert tensors, and `read_host_group` for
prefused experts. It reads addressable slices, retains host buffers until local
H2D completes, coordinates read errors, and assembles JAX arrays. It does not
hold a model or model configuration, bind NNX parameters, or assign them.

`TensorLayout` takes target shapes/dtypes/shardings and returns transformed
arrays. The same conversion functions run during schema tracing and execution.
`WeightLoader` owns preparation, planning, source lifetime, cache reuse, and the
single validated parameter-assignment path. It remains a concrete orchestrator;
model-specific prepare hooks and recipes do not require a loader subclass.

The model entry above supplies default implementations. Tools and integrations
can instead pass `source=` and `reader=` to `WeightLoader`. An explicitly supplied
source must already be open; its caller owns closing it. The outer model-loading
session shares one source across nested model loads through the model config.

## Grouped inputs

Use `sources` when an output depends on multiple checkpoint tensors. A device
recipe receives those host arrays in declaration order and returns one JAX array
per target. It must not perform file I/O or replace model modules.

```python
WeightSpec(
    ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"],
    sources=("attn.qkv.weight", "attn.qkv.weight_scale_inv"),
    recipe=partial(load_fused_qkv, ...),
)
```

A spec with `sources` and no recipe stacks separate expert tensors. It supports
explicit physical-to-logical expert placement, TP slices, split files, and bulk
reads. `host_recipe` handles prefused expert tensors: the reader reads local
expert intervals, applies the NumPy conversion, and uploads each output's local
shards. This avoids materializing all global experts on each process.

MiMo checkpoint interpretation lives in `models/mimo_weight_loading.py`.
Absorbed MLA preparation lives in `layers/weight_loading.py`. Shared numerical
layout conversions live in `model_loader/weights/recipes.py`.

## Preparation and ownership

The outer loader constructs the model with `nnx.eval_shape` and prepares static
quantization. Before binding parameters, `WeightLoader` invokes optional hooks:

- Model: `prepare_weight_loading(loader, mappings)`.
- Nested layer: `prepare_weight_loading(loader, mappings, prefix)`.

A hook returns the resulting mapping and may change abstract module structure,
parameter shapes, or aliases. It must not read weight data, allocate full weight
arrays, transfer data, or issue collectives. For example, MiMo replaces an
abstract quantized projection with its final BF16 projection here; its recipe
later performs the original FP32 dequantization and BF16 rounding.

After preparation, target identity is fixed. A mapping with no checkpoint inputs,
or with targets absent from the current model, is skipped and reported with a
warning. This lets a shared mapping cover checkpoint aliases and model variants.
Missing inputs on optional entries or excluded layers do not produce warnings.
Skipped entries appear in the load report; their declared sources do not count as
unexpected checkpoint keys. Unmapped checkpoint keys remain in `unexpected` and
only fail when `validate_checkpoint_coverage=True`.

A partially present input group still fails unless it is explicitly optional:
loading only some inputs would change a fused conversion or expert stack.
Duplicate writers (including two paths to one shared parameter) and ordinary
layout shape mismatches also fail during planning. Group outputs are checked
against the prepared schema. The outer model loader rejects remaining abstract
parameters, so skipping a mapping cannot make an incomplete model ready to serve.

A model-loading session owns one source. Local mmap handles are released after
the last group using a file completes. Groups have deterministic file ordering
so late conversions do not retain the whole checkpoint's mappings. RunAI copies
borrowed SDK buffers before advancing its chunk iterator. Host owners remain
alive until their device transfers complete.

## Multiple processes and resource limits

Metadata is scanned by the coordinator, then broadcast using basenames. Each
process binds its own checkpoint root. Slice assignment uses the target
sharding's addressable devices, including devices managed remotely by a single
controller. No physical-host filter is applied.

Processes compare expanded plan digests before reading weights. Initialization,
planning, and read failures are coordinated at fixed boundaries outside worker
threads and callbacks. Global layout operations follow the same group order.
P/D cache identity includes checkpoint identity, mapping, dtype and mesh shape;
a hit can bypass reads only when all processes agree.

`SGLANG_WEIGHT_LOAD_MAX_INFLIGHT_BYTES` defaults to 4 GiB and is a concurrency
target for estimated owned working sets and pending groups. It must be positive.
An indivisible group larger than the target is allowed with a warning. Device
recipes estimate their working set as eight times the total checkpoint input
bytes; oversized recipes drain pending work before reading and wait for their
own outputs before continuing. This reserves space for inputs, conversion copies
and outputs, but is a conservative estimate rather than a measured peak.

Host recipes process local expert intervals one at a time. Expert readers reduce
worker counts or split device batches to fit the target where possible, falling
back to one worker or one device batch for an oversized shard. Host owners are
released after their uploads finish, before reading the next group. Recipe input
owners are also drained before handing off to another reader group.
`SGLANG_MOE_LOAD_WORKERS` defaults to 16 and remains an upper bound on expert read
concurrency. A larger byte target can allow more overlap when memory permits.

This target does not cap individual tensor size or process RSS: an oversized
group still needs enough actual memory, and mmap pages, allocator retention, SDK
staging, and device buffers are separate. GCSFuse prefetch keeps the existing
policy and runs once per source session after planning.

Dummy loading uses the final abstract schema without opening a checkpoint.
Local and RunAI selection remains controlled by the existing load-format CLI.

## Validation

`test_weight_loading_distributed.py` starts two real JAX CPU controllers and
checks rank-local roots, inconsistent plans, read failures, and asymmetric
cache state. SDK fakes in `test_runai_loader.py` specifically exercise borrowed
buffer reuse; they do not establish native SDK or GCS performance.
