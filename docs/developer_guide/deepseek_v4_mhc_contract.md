# DeepSeek V4 mHC Boundary

H exposes stream expansion, sequential `pre`/`post`, and final head collapse.
M owns parameters, decoder placement, mesh selection, and checkpoint loading.
The TPU7x schedule and model-side sharding pattern follow PR #1789 at
`72fed1403c19d95fb5120f717ead4b11d6aae9c9`.

## Dtypes and Residual Flow

Residual streams have shape `[..., hc_mult, hidden_size]`. Each attention and
FFN sublayer uses its own FP32 `fn`, `base`, and `scale` parameter set, with
shapes from `mhc_param_shapes(config)`. Head collapse uses a separate model-level
set. All post gates and mixing matrices stay FP32.

`hc_sinkhorn_iters` must be a built-in Python `int` greater than or equal to one.
Booleans, floats, strings, and NumPy integers are rejected without coercion.

For each sublayer, retain the original streams, run `pre`, cast the collapsed
activation to the model activation dtype before normalization, and pass the
sublayer output and original streams to `post`. Cast the updated streams to the
model activation dtype before the next operation. The reference path can retain
FP32 activation intermediates; Pallas returns activations in the input dtype.
Head collapse includes its specified BF16 normalization boundary and returns the
streams' dtype. It runs before the final model normalization and LM head.

## Backend Selection

`backend="auto"` uses the explicit AOT compilation target when present and the
serving runtime otherwise. TPU selects Pallas; CPU selects the JAX reference.
The registered Pallas schedules support TPU v6e and TPU7x, including the
`TPU v7x` and `v7x` aliases. Other TPU generations require an explicit schedule;
they must not inherit an unvalidated device's VMEM budget.

TPU7x uses 32 MiB scoped VMEM and a 32 MiB XLA live-set budget with a 64 KiB
reserve. Tile selection and the XLA/Pallas post boundary use these budgets.

## Sharding and Padding

Token rows may be sharded; residual-stream and hidden axes stay local, and gate
parameters are replicated. Operations are token-local, so padded rows cannot
affect valid-token results. H does not mask the final model outputs.

The Pallas entry points operate inside a manual mesh scope. For an Explicit
mesh, M wraps them with `jax.shard_map`; directly passing a globally sharded
array to Pallas is not the interface. For a row axis selected by M, use:

| Operation | Input specs | Output specs |
| --- | --- | --- |
| `pre(streams, fn, base, scale)` | `P(row, None, None), P(), P(), P()` | `P(row, None), P(row, None), P(row, None, None)` |
| `post(y, residual, post_gate, comb)` | `P(row, None), P(row, None, None), P(row, None), P(row, None, None)` | `P(row, None, None)` |
| `collapse_head(streams, fn, base, scale)` | `P(row, None, None), P(), P(), P()` | `P(row, None)` |

The model in PR #1789 combines this manual boundary with
`jax.sharding.auto_axes` and explicit output shardings. M chooses the row axis
and any resharding needed by attention or MoE; H does not add collectives or
choose a model mesh.

## Validation

The CPU suite configures two CPU devices and checks reference row sharding,
padding, and the model's dtype conversion boundaries. A Pallas interpreter test
uses the model-side Explicit-mesh wrapper for all three operations, with one or
four local rows and a NaN padded row. These local counts exercise kernel padding.
The interpreter verifies arithmetic and the manual calling boundary, not Mosaic
compilation or real TPU numerics.

The focused TPU test compares Pallas with the reference on real hardware when
available. Full-model results recorded in PR #1789 do not replace independent
H-layer numerical checks or establish exact logits equivalence.
