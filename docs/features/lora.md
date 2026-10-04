# LoRA

sglang-jax supports serving LoRA (Low-Rank Adaptation) adapters on top of a base model. LoRA adds
adaptation capability without modifying the original weights: for the original weight `W`, the LoRA
output is `W·x + (B·A)·x * scaling`, where `A` is the down-projection (`hidden → rank`), `B` is the
up-projection (`rank → hidden`), and `scaling = alpha / rank`.

LoRA adapters are loaded from standard HuggingFace LoRA adapter directories containing
`adapter_config.json` and the adapter `safetensors` weights. Deep implementation details live in
[Architecture: LoRA Dynamic Adapters](../architecture/10-lora.md).

## Two Modes

sglang-jax provides two LoRA modes:

| Mode | Flag | Description |
|------|------|-------------|
| Dynamic LoRA | `--enable-lora` | Multiple adapters, per-request adapter switching with per-request KV-cache isolation. |
| Static LoRA | `--enable-static-lora` | Single adapter for RL scenarios; weights are merged into the base model at load time. Mutually exclusive with `--enable-lora`. |

`--enable-lora` is auto-enabled whenever `--lora-paths` is provided.

## Dynamic LoRA

### Launching the server

Pass the adapter paths to `--lora-paths`. Adapters can be given as:

- `name=path` — explicit adapter name (used in requests)
- `path` — name is derived from the path basename
- HuggingFace repo IDs (both forms above apply)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
uv run python -u -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --trust-remote-code \
  --dist-init-addr=0.0.0.0:10011 \
  --nnodes=1 \
  --tp-size=1 \
  --device=tpu \
  --random-seed=3 \
  --node-rank=0 \
  --mem-fraction-static=0.8 \
  --max-prefill-tokens=8192 \
  --download-dir=/tmp \
  --dtype=bfloat16 \
  --skip-server-warmup \
  --enable-lora \
  --lora-paths adapter1=/path/to/adapter1 adapter2=/path/to/adapter2
```

When constructing `ServerArgs` programmatically, adapters can also be specified as a dict `{"adapter1": "/path/to/adapter1", ...}` (the CLI `--lora-paths` only accepts `name=path` / `path` forms).

### Sending requests with a specific adapter

Requests select an adapter by the adapter **name** (the key/name registered via `--lora-paths`)
through the `lora_path` request field. Requests that omit `lora_path` run on the base model.

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "lora_path": "adapter1"
  }'
```

The same field is used in the Python engine API
(`Engine.generate(..., lora_path="adapter1")`). Chat completions does not yet
forward `lora_path` to the generation request, so the adapter would be ignored
there.

### Server arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-lora` | `None` | Enable dynamic LoRA (auto-enabled when `--lora-paths` is set). |
| `--lora-paths` | `None` | Adapters to preload; `name=path` / `path` / dict format. |
| `--max-loras-per-batch` | `8` | Maximum number of different adapters per batch. Preloaded adapters are pinned, so at most `max_loras_per_batch - 1` (7 with the default) can be preloaded. |
| `--max-lora-rank` | `None` | Maximum LoRA rank; auto-inferred from the loaded adapters when unset. |
| `--lora-target-modules` | `None` | Modules to apply LoRA to (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, or `all`); auto-inferred from the adapters when unset. |


`qkv` and `gate_up` projections are automatically handled by merging the corresponding per-module
LoRA weights, and adapters with different ranks are padded to `max_lora_rank`.

CLI string preloads are pinned. To allow device-slot eviction, configure unpinned startup
adapters programmatically through `Engine`/`ServerArgs`, for example:

```python
lora_paths = [{"name": "adapter1", "path": "/path/to/adapter1", "pinned": False}]
```

This enables device-slot reuse across batches; it does not enable runtime registration or
host-memory eviction.

With overlap scheduling, the bootstrap handler signals sampling readiness before waiting
for the first batch's `launch_done` event. This prevents the next preparation from
replacing shared weight references before the worker captures them. The wait also applies
after idle and to non-LoRA bootstrap batches. It does not add a device-wide synchronization;
it can delay the next preparation until the first forward call returns.

### Testing device-slot eviction

Follow the [contribution guide](../developer_guide/contribution_guide.md) to install the
CPU development dependencies. From the repository root, run the focused pool and manager tests:

```bash
python -m unittest discover -s test/srt/lora -p 'test_lora_m*.py' -v
```

The pool tests use small JAX arrays without downloading a model. They cover LRU order,
pinned and batch-required adapters, reserved slot zero, capacity errors, and numerical
equivalence between reused and fresh slots. Manager tests check pin propagation and
the conditional refresh of model-layer buffer references.

The pool tests also check that captured JAX arrays retain their numerical output after
slot replacement. For scheduler bootstrap ordering, run:

```bash
python -m unittest discover -s test/srt/lora -p 'test_lora_overlap_bootstrap.py' -v
```

These scheduler tests require the serving runtime dependencies on a supported platform.
They use controlled thread events to test reference capture before the next preparation,
repeated bootstrap calls, and sampling-readiness-before-wait ordering. They do not launch
a model or validate TPU execution, JIT/AOT serving integration, or latency.

For serving validation on a TPU environment, also run `python test/srt/lora/test_dynamic_lora.py`.
That existing test downloads model/adapter checkpoints and checks mixed-adapter serving;
it does not replace the pool tests that deliberately force eviction.

## Static LoRA

Static LoRA is intended for RL scenarios where only a single adapter is used and requests never
switch adapters at runtime. It is mutually exclusive with `--enable-lora`:

- Supports exactly one adapter via fixed `A`/`B` buffers updated externally between RL steps (weights are not merged into the base model; BGMV is not used)
- Requires explicit `--lora-scaling` (`alpha / rank`)
- Does not accept `--lora-paths` and requires `--max-loras-per-batch 1`

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
uv run python -u -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --trust-remote-code \
  --dist-init-addr=0.0.0.0:10011 \
  --nnodes=1 \
  --tp-size=1 \
  --device=tpu \
  --node-rank=0 \
  --mem-fraction-static=0.8 \
  --download-dir=/tmp \
  --dtype=bfloat16 \
  --skip-server-warmup \
  --enable-static-lora \
  --lora-scaling 1.0 \
  --max-lora-rank 8 \
  --lora-target-modules q_proj \
  --max-loras-per-batch 1
```

## Limitations

- All configured adapters are loaded into host memory at startup; dynamic adapter (un)loading
  and host-memory eviction are not yet supported. Device slots use LRU eviction for unpinned
  adapters not required by the current batch. Preloaded pinned adapters are never evicted.
  A batch that cannot fit alongside resident pinned adapters raises a capacity error.
- KV caches are fully isolated per adapter, so different adapters do not share prefix caches even
  with identical prompts.
- DFLASH (speculative decoding) does not support LoRA.
