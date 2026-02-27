import sys
from safetensors import safe_open

if len(sys.argv) < 2:
    print("Usage: python check_shape_ltx2.py <path_to_safetensors>")
    sys.exit(1)

file_path = sys.argv[1]

with safe_open(file_path, framework="pt", device="cpu") as f:
    print(f.get_tensor("model.diffusion_model.transformer_blocks.0.scale_shift_table").shape)
