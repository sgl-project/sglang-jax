# Step 3.5 Flash — real-weight HF ⇆ sglang-jax alignment (manual tooling)

NOT part of CI. Manual real-weight correctness verification against the official
HuggingFace reference (`modeling_step3p5.py`). Complements the microscale
`naive==HF` alignment (function-level) with a **real-weight** check.

Two comparisons, both off the same fixed `inputs.json` (raw token ids — no
tokenizer skew):

- **A. end-to-end next-token (decision-level, real prompts).** HF full model
  (bf16, 8×H100) vs my running server, per-prompt next-token top-k. argmax
  agreement + top-k overlap across many prompts = real-weight correctness signal.
  **Fully runnable now.**
- **B. per-layer growth curve (localizer).** HF dumps per-layer hidden states
  (`dump_hf.py` already does this). The jax side needs a heavier multi-host
  hidden-state dump (load the 196B outside the server with per-layer capture) —
  written separately / on demand. Run B only if A shows divergence, to localize.

Note on precision: full-model fp32 (784GB) does NOT fit on 640GB, so the
full-model checks are **bf16** (decision-level + bf16 band, like flash==naive).
The tight fp32 (~1e-5) per-element check is the isolated single-layer variant
(separate, fits on 1 GPU/CPU) — add if a layer looks suspicious in B.

## Run order

```bash
# 0. (once) generate shared inputs
python step3p5_align_inputs.py --vocab <VOCAB> --num-prompts 64 --len 24

# 1. HF reference on the GPU box (smoke with --num-prompts 1 first!)
pip install torch transformers accelerate safetensors
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
