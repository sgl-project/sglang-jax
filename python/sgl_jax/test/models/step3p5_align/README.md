# Step 3.5 Flash — real-weight HF ⇆ sglang-jax alignment (manual tooling)

NOT part of CI. Manual real-weight correctness verification against the official
HuggingFace reference (`modeling_step3p5.py`). Complements the microscale
`naive==HF` alignment (function-level) with a **real-weight** check.

## gsm8k vs upstream sglang (apples-to-apples, run this first)

`run_sglang_gsm8k_against_jax.py` runs **upstream sglang's OWN gsm8k eval** (no
port, no divergence) with the EXACT setup sglang uses for Step-3.5-Flash
(`GSM8KMixin` in `test_step3p5_flash_chain_mtp.py`: api=completion, num_shots=5,
max_tokens=512, num_examples=200, threshold 0.83), pointed at the sglang-jax
server's OpenAI `/v1/completions`:

```bash
pip install sglang          # upstream sglang, for its eval code
# NOTE: --base-url is the server ROOT, NO /v1 (sglang's run_eval appends /v1;
# passing /v1 yields /v1/v1/completions -> 404).
python run_sglang_gsm8k_against_jax.py --base-url http://<node0>:30000 --model <served-model>
```
score ≥ 0.83 (identical harness) ⇒ functionally equivalent to sglang for gsm8k.
(This replaces the old self-referenced 0-shot gsm8k/mmlu in test_step3p5_models.py,
which had no external reference and could not prove correctness.)

## HF reference comparison

Two comparisons, both off the same fixed `inputs.json` (raw token ids — no
tokenizer skew):

- **A. end-to-end next-token (decision-level, real prompts).** HF full model
  (bf16, 8×H100) vs my running server, per-prompt next-token top-k. argmax
  agreement + top-k overlap across many prompts = real-weight correctness signal.
  **Fully runnable now.**
- **B. per-layer growth curve (localizer).** Intended to dump per-layer hidden
  states. NOTE: the real checkpoint's `modeling_step3p5.py` does NOT populate
  `hidden_states` even with `output_hidden_states=True`, so B is currently NOT
  producible from the HF side (see gotchas below). Rely on A; if A diverges, B
  needs a custom forward that captures hidden states on both sides.

Note on precision: full-model fp32 (784GB) does NOT fit on 640GB, so the
full-model checks are **bf16** (decision-level + bf16 band, like flash==naive).
The tight fp32 (~1e-5) per-element check is the isolated single-layer variant
(separate, fits on 1 GPU/CPU) — add if a layer looks suspicious in B.

## Environment & gotchas (verified end-to-end on 16×A100-40GB, real 196B)

`dump_hf.py` loads the real ~398GB bf16 checkpoint. Getting it to run required the
following — all already handled inside `dump_hf.py`, listed here so the setup is
reproducible:

- **`transformers==4.57.1` is required** (pinned in `python/pyproject.toml`).
  transformers 5.x turns the config into a strict dataclass and rejects this
  checkpoint: `layer_types` has 48 entries (45 decoder + 3 MTP) while
  `num_hidden_layers=45`.
- **Do NOT rely on `device_map="auto"`.** It routes through `get_balanced_memory`,
  which — confused by the uneven layer sizes (dense ~0.43GB vs MoE ~9GB) — injects
  a CPU budget and offloads MoE layers to CPU even though the ~394GB fits on the
  GPUs (forward then crawls / hits meta-device errors). `dump_hf.py` instead
  precomputes an explicit GPU-only `device_map` on a meta skeleton via
  `infer_auto_device_map` (no `cpu` entry, `no_split_module_classes=
  ["Step3p5DecoderLayer"]`) → all weights on GPU (~394GB across 11×A100-40GB).
- **Use `dtype=` not `torch_dtype=`** (deprecated in 4.57, silently ignored →
  fp32 → 2× memory → offload).
- **Checkpoint `modeling_step3p5.py` quirks**: `get_input_embeddings()` is
  non-standard (takes `input_ids` and returns embeddings, not the module), so the
  embedding device is read from `model.model.embed_tokens` directly; and it does
  not populate `hidden_states`, so B is skipped (A is the working criterion).

## Run order

```bash
# 0. (once) generate shared inputs
python step3p5_align_inputs.py --vocab <VOCAB> --num-prompts 64 --len 24

# 1. HF reference on the GPU box (smoke with --num-prompts 1 first!)
pip install torch "transformers==4.57.1" accelerate safetensors
python dump_hf.py --ckpt <CHECKPOINT_PATH> --out hf_dump --topk 20 --num-prompts 1   # smoke
python dump_hf.py --ckpt <CHECKPOINT_PATH> --out hf_dump --topk 20                    # full

# 2. my side via the running server (same inputs)
STEP35_BASE_URL=http://<node0>:30000 python dump_server.py --out jax_dump --topk 20

# 3. compare (offline, no accelerator)
python compare.py --hf hf_dump --jax jax_dump
```

Copy `inputs.json` to whichever machine each step runs on (or share the dir) so
both sides use identical inputs.

## Reading the result (A)

- **argmax agreement ≈ 1.0** (a few borderline flips OK = bf16 reduction noise):
  the implementation matches HF on real weights end-to-end. ✅
- **low agreement**: a real implementation divergence → run B to localize the
  layer, then the isolated fp32 per-layer check to pin it.
