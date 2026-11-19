import json
import os
import sys
import tempfile

sys.path.insert(0, ".")
import loader as L
import numpy as np
import torch
from safetensors.torch import save_file

rng = np.random.default_rng(0)
# ---- FP4 experts: mine vs the DeepSeek convert.py convention (FP4_TABLE, low nibble first, scale per 32 along K)
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
N, K = 96, 256
packed = torch.from_numpy(rng.integers(-128, 128, (N, K // 2), dtype=np.int8))
scale_e = torch.from_numpy(rng.integers(118, 134, (N, K // 32), dtype=np.uint8))
scale_t = scale_e.view(torch.float8_e8m0fnu)
x = packed.view(torch.uint8)
lo, hi = x & 0x0F, (x >> 4) & 0x0F
ref = torch.stack([FP4_TABLE[lo.long()], FP4_TABLE[hi.long()]], dim=-1).flatten(
    1
)  # [N, K] e2m1 values
ref = (ref.unflatten(-1, (-1, 32)) * scale_t.float().unsqueeze(-1)).flatten(1)
mine = L.dequant_expert(L.unpack_fp4_x2(packed.numpy()), scale_e.numpy())
print(
    "fp4 expert dequant == reference convention:",
    np.array_equal(ref.numpy(), mine),
    "| max|d|",
    np.abs(ref.numpy() - mine).max(),
)

# ---- dense FP8 with 32x32 e8m0 block scales
O, I = 64, 128
w8 = torch.randn(O, I).clamp(-4, 4).to(torch.float8_e4m3fn)
s8 = torch.from_numpy(rng.integers(120, 130, (O // 32, I // 32), dtype=np.uint8)).view(
    torch.float8_e8m0fnu
)
ref = (w8.float().view(O // 32, 32, I // 32, 32) * s8.float()[:, None, :, None]).reshape(O, I)
print(
    "fp8 block dequant == reference:",
    np.array_equal(
        ref.numpy(),
        L.dequant_fp8_block(
            w8.view(torch.uint8).numpy().view(L._DT["F8_E4M3"]), s8.view(torch.uint8).numpy()
        ),
    ),
)

# ---- safetensors round trip with the checkpoint's dtypes through the minimal reader + index.json
tmp = tempfile.mkdtemp()
tensors = {
    "a.weight": w8,
    "a.scale": s8,
    "e.weight": packed,
    "e.scale": scale_t,
    "n.weight": torch.randn(7, dtype=torch.bfloat16),
    "f": torch.randn(3),
}
save_file(tensors, f"{tmp}/model-00001.safetensors")
json.dump(
    {"weight_map": {k: "model-00001.safetensors" for k in tensors}},
    open(f"{tmp}/model.safetensors.index.json", "w"),
)
ck = L.SafetensorsDir(tmp)
ok = True
for k, v in tensors.items():
    got = ck.get(k)
    ok &= (
        got.shape == tuple(v.shape)
        and np.array_equal(got.view(np.uint8).reshape(-1), v.view(torch.uint8).numpy().reshape(-1))
        if v.dtype != torch.float32
        else np.array_equal(got, v.numpy())
    )
print(
    "safetensors reader (fp8 / e8m0 / int8 / bf16 / f32) round trip exact:",
    bool(ok),
    "| shapes:",
    ck.shape("e.weight"),
    ck.get("a.scale").dtype,
)
