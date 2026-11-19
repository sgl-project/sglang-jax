"""Write the tiny reference model as a checkpoint in the real format (fp8+32x32 e8m0 dense, packed fp4 experts) + the matching
dequantized reference weights/outputs. Usage: python make_synth_ckpt.py <ref_cpu_dir> <out_dir>"""

import json
import os
import sys

R, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import model as M
import numpy as np
import torch
from safetensors.torch import save_file

os.makedirs(OUT, exist_ok=True)
sd = {k: v.float() for k, v in torch.load(f"{R}/ref_weights.pt").items()}
FP8_NAMES = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "indexer.wq_b.weight",
    "shared_experts.w1.weight",
    "shared_experts.w2.weight",
    "shared_experts.w3.weight",
)
GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
tensors, deq = {}, {}
for k, v in sd.items():
    if ".ffn.experts." in k and k.endswith(".weight"):
        n, kk = v.shape
        vb = v.reshape(n, kk // 32, 32)
        amax = vb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        e = torch.ceil(torch.log2(amax / 6.0)).clamp(-100, 100)
        sc = torch.exp2(e)
        t = vb / sc
        idx = (t.abs().unsqueeze(-1) - GRID).abs().argmin(-1)
        q = torch.sign(t) * GRID[idx]  # [n,k/32,32] fp4 values
        code = torch.where(q < 0, idx + 8, idx).to(torch.uint8).reshape(n, kk)  # sign bit 3
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
        tensors[k] = packed
        tensors[k.replace(".weight", ".scale")] = (
            (e[..., 0] + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        )
        deq[k] = (q * sc).reshape(n, kk)
    elif any(k.endswith(s) for s in FP8_NAMES):
        o, i = v.shape
        vb = v.reshape(o // 32, 32, i // 32, 32)
        amax = vb.abs().amax(dim=(1, 3)).clamp_min(1e-8)
        e = torch.ceil(torch.log2(amax / 448.0))
        sc = torch.exp2(e)
        q = (vb / sc[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        tensors[k] = q.reshape(o, i)
        tensors[k.replace(".weight", ".scale")] = (
            (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        )
        deq[k] = (q.float() * sc[:, None, :, None]).reshape(o, i)
    else:
        tensors[k] = v.contiguous()
        deq[k] = v
names = list(tensors)
half = len(names) // 2
for i, part in enumerate((names[:half], names[half:])):
    save_file(
        {n: tensors[n].contiguous() for n in part}, f"{OUT}/model-0000{i+1}-of-00002.safetensors"
    )
json.dump(
    {
        "weight_map": {
            n: f"model-0000{1 if j < half else 2}-of-00002.safetensors" for j, n in enumerate(names)
        }
    },
    open(f"{OUT}/model.safetensors.index.json", "w"),
)
torch.save(deq, f"{OUT}/ref_weights_deq.pt")
# reference outputs with the dequantized weights (no-quant mode)
torch.set_default_dtype(torch.float32)
args = M.ModelArgs(
    dtype="bf16",
    expert_dtype=None,
    n_mtp_layers=0,
    compress_ratios=(0, 2, 2, 1, 1, 1),
    kv_source_layers=(1, 3),
    index_source_layers=(1, 3, 5),
    n_layers=6,
    max_seq_len=256,
    max_batch_size=2,
    window_size=32,
    candidate_source_layer=3,
    candidate_topk_blocks=4,
    candidate_block_size=4,
    index_topk=8,
)
torch.manual_seed(0)
model = M.Transformer(args)
model.float()
model.load_state_dict(deq)
model.eval()
ids = torch.from_numpy(np.load(f"{R}/ref_outputs.npz")["input_ids"])
out = {}
with torch.no_grad():
    _, lg, _ = model(ids[:, :64])
    out["prefill_logits"] = lg.numpy()
    for i in range(64, 68):
        _, lg, _ = model(ids[:, i : i + 1], i)
        out[f"decode_logits_{i}"] = lg.numpy()
np.savez(f"{OUT}/ref_outputs_deq_noquant.npz", input_ids=ids.numpy(), **out)
print("synthetic checkpoint written:", len(names), "tensors")
