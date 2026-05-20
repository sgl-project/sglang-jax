# test_vit_tower_jax.py  
"""  
JAX/TPU test for the Kimi-K2.5 Vision Tower.  
  
Assumptions about your JAX port:  
  - Your file is importable as `kimi_k25` (or adjust the import below)  
  - It exposes a class `Kimi_K25_VisionModel` with:  
      - `load_weights(weights_path_or_dict)` method  
      - `self.vision_tower` attribute  
  - `vision_tower.__call__(pixel_values, grid_thws)` returns either:  
      - A list of jnp arrays, one per image, each shape (H'*W', 4, hidden_size)  
      - OR a single concatenated jnp array of shape (total_vis_tokens, 4, hidden_size)  
  - pixel_values dtype: jnp.bfloat16 (TPU native)  
  - grid_thws dtype: jnp.int32  
  
Usage:  
  # Shape/logic test only (random weights, fast):  
  python test_vit_tower_jax.py  
  
  # With real weights from a checkpoint directory:  
  python test_vit_tower_jax.py --weights /path/to/weights --atol 0.02  
"""  
  
import argparse  
import sys  
from dataclasses import dataclass  
from typing import List, Tuple, Union  
  
import jax  
import jax.numpy as jnp  
import numpy as np  
from flax import nnx
  
from kimi_k25_vit import Kimi_K25_VisionModel  
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (
    KimiK25ModelVitConfig,
)
# ───────────────────────────────────────────────────────────────────────────  
  
# ---------------------------------------------------------------------------  
# Config (mirrors python/sglang/srt/configs/kimi_k25.py defaults)  
# ---------------------------------------------------------------------------  
HIDDEN_SIZE   = 1152  
NUM_HEADS     = 16  
NUM_LAYERS    = 27  
PATCH_SIZE    = 14  
MERGE_KH      = 2  
MERGE_KW      = 2  
INIT_H        = 64  
INIT_W        = 64  
INIT_T        = 4  
  
# ---------------------------------------------------------------------------  
# Test case definition  
# ---------------------------------------------------------------------------  
@dataclass  
class TestCase:  
    name: str  
    # list of (t, h, w) in PATCH units — NOT pixel units  
    # e.g. (1, 4, 6) means 1 frame, 4 patch-rows, 6 patch-cols  
    #      → image was 56×84 pixels (4*14 × 6*14)  
    grid_thws: List[Tuple[int, int, int]]  
  
    def total_patches(self) -> int:  
        return sum(t * h * w for t, h, w in self.grid_thws)  
  
    def expected_output_shapes(self) -> List[Tuple[int, int, int]]:  
        """  
        After tpool_patch_merger with merge_kernel=(2,2):  
          temporal: t frames → 1 (mean)  
          spatial:  (h, w) → (h//2, w//2) blocks, each block kept as 4 tokens  
        Output per image: (h//2 * w//2, 4, HIDDEN_SIZE)  
        """  
        shapes = []  
        for t, h, w in self.grid_thws:  
            nh = h // MERGE_KH  
            nw = w // MERGE_KW  
            shapes.append((nh * nw, MERGE_KH * MERGE_KW, HIDDEN_SIZE))  
        return shapes  
  
  
TEST_CASES = [  
    TestCase(  
        name="single_image_small",  
        grid_thws=[(1, 4, 6)],          # 56×84 px → 12 output tokens  
    ),  
    TestCase(  
        name="single_image_medium",  
        grid_thws=[(1, 8, 12)],         # 112×168 px → 48 output tokens  
    ),  
    TestCase(  
        name="batch_two_images_same_size",  
        grid_thws=[(1, 4, 4), (1, 4, 4)],  
    ),  
    TestCase(  
        name="batch_two_images_different_size",  
        grid_thws=[(1, 4, 6), (1, 8, 8)],   # varlen batch  
    ),  
    TestCase(  
        name="video_2frames",  
        grid_thws=[(2, 4, 6)],          # 2 frames → temporal mean → same output shape as 1 frame  
    ),  
    TestCase(  
        name="video_4frames",  
        grid_thws=[(4, 4, 6)],          # 4 frames → temporal mean  
    ),  
    TestCase(  
        name="batch_image_and_video",  
        grid_thws=[(1, 4, 6), (4, 4, 4)],  
    ),  
]  
  
# ---------------------------------------------------------------------------  
# Helpers  
# ---------------------------------------------------------------------------  
  
def make_pixel_values(grid_thws: List[Tuple[int, int, int]],  
                      rng: jax.Array,  
                      dtype=jnp.bfloat16) -> jnp.ndarray:  
    """  
    Create synthetic pixel_values of shape (L, 3, PATCH_SIZE, PATCH_SIZE).  
    L = sum(t * h * w) across all images.  
    Values are in [0, 1] (normalised pixel range).  
    """  
    total = sum(t * h * w for t, h, w in grid_thws)  
    return jax.random.uniform(  
        rng,  
        shape=(total, 3, PATCH_SIZE, PATCH_SIZE),  
        dtype=dtype,  
        minval=0.0,  
        maxval=1.0,  
    )  
  
  
def make_grid_thws(grid_thws: List[Tuple[int, int, int]]) -> jnp.ndarray:  
    """Create grid_thws tensor of shape (N, 3) with dtype int32."""  
    return jnp.array(grid_thws, dtype=jnp.int32)  
  
  
def normalise_output(  
    output: Union[List[jnp.ndarray], jnp.ndarray]  
) -> List[np.ndarray]:  
    """  
    Accept either a list of jnp arrays or a single concatenated array.  
    Always returns a list of numpy arrays for easy indexing.  
    """  
    if isinstance(output, (list, tuple)):  
        return [np.array(x) for x in output]  
    # single concatenated tensor — caller must split manually if needed  
    return [np.array(output)]  
  
  
def check_no_nan_inf(arr: np.ndarray, label: str) -> bool:  
    has_nan = np.any(np.isnan(arr.astype(np.float32)))  
    has_inf = np.any(np.isinf(arr.astype(np.float32)))  
    if has_nan:  
        print(f"  FAIL [{label}]: output contains NaN")  
        return False  
    if has_inf:  
        print(f"  FAIL [{label}]: output contains Inf")  
        return False  
    return True  
  
  
def check_shape(arr: np.ndarray, expected: Tuple, label: str) -> bool:  
    if arr.shape != expected:  
        print(f"  FAIL [{label}]: expected shape {expected}, got {arr.shape}")  
        return False  
    return True  
  
  
def check_dtype(arr: np.ndarray, expected_dtype, label: str) -> bool:  
    # bfloat16 is represented as jnp.bfloat16; numpy uses ml_dtypes.bfloat16  
    # We just check it is a float type and not float64 (which would indicate  
    # an accidental upcast on TPU).  
    if arr.dtype == np.float64:  
        print(f"  WARN [{label}]: dtype is float64 — expected bfloat16 on TPU")  
    return True  # non-fatal  
  
  
def check_value_range(arr: np.ndarray, label: str,  
                      lo: float = -1e4, hi: float = 1e4) -> bool:  
    f32 = arr.astype(np.float32)  
    if f32.min() < lo or f32.max() > hi:  
        print(f"  WARN [{label}]: values outside [{lo}, {hi}]: "  
              f"min={f32.min():.4f} max={f32.max():.4f}")  
    return True  # non-fatal; just a sanity warning  
  
  
# ---------------------------------------------------------------------------  
# Core test runner  
# ---------------------------------------------------------------------------  
  
def run_test_case(model: Kimi_K25_VisionModel,  
                  tc: TestCase,  
                  rng: jax.Array,  
                  verbose: bool = True) -> bool:  
    """Run one test case. Returns True if all assertions pass."""  
    if verbose:  
        print(f"\n{'='*60}")  
        print(f"Test: {tc.name}")  
        print(f"  grid_thws : {tc.grid_thws}")  
        print(f"  total patches: {tc.total_patches()}")  
        print(f"  expected output shapes: {tc.expected_output_shapes()}")  
  
    pixel_values = make_pixel_values(tc.grid_thws, rng)  
    grid_thws    = make_grid_thws(tc.grid_thws)  
  
    if verbose:  
        print(f"  pixel_values : shape={pixel_values.shape}, dtype={pixel_values.dtype}")  
        print(f"  grid_thws    : shape={grid_thws.shape}, dtype={grid_thws.dtype}")  
  
    # ── call your JAX vision tower ──────────────────────────────────────────  
    try:  
        output = model.vision_tower(pixel_values, grid_thws)  
    except Exception as e:  
        print(f"  FAIL [{tc.name}]: vision_tower raised an exception: {e}")  
        return False  
    # ────────────────────────────────────────────────────────────────────────  
  
    outputs = normalise_output(output)  
    expected_shapes = tc.expected_output_shapes()  
  
    # If the model returns a single concatenated tensor, we can only check  
    # the total token count and the last two dims.  
    if len(outputs) == 1 and len(tc.grid_thws) > 1:  
        arr = outputs[0]  
        total_expected_tokens = sum(s[0] for s in expected_shapes)  
        # shape should be (total_tokens, 4, HIDDEN_SIZE)  
        expected_concat = (total_expected_tokens, MERGE_KH * MERGE_KW, HIDDEN_SIZE)  
        ok = True  
        ok &= check_shape(arr, expected_concat, tc.name + "/concat")  
        ok &= check_no_nan_inf(arr, tc.name)  
        ok &= check_dtype(arr, jnp.bfloat16, tc.name)  
        ok &= check_value_range(arr, tc.name)  
        if verbose and ok:  
            print(f"  PASS (concatenated output) shape={arr.shape} dtype={arr.dtype}")  
        return ok  
  
    # Per-image list output  
    if len(outputs) != len(tc.grid_thws):  
        print(f"  FAIL [{tc.name}]: expected {len(tc.grid_thws)} output tensors, "  
              f"got {len(outputs)}")  
        return False  
  
    all_ok = True  
    for i, (arr, exp_shape) in enumerate(zip(outputs, expected_shapes)):  
        label = f"{tc.name}/img{i}"  
        ok = True  
        ok &= check_shape(arr, exp_shape, label)  
        ok &= check_no_nan_inf(arr, label)  
        ok &= check_dtype(arr, jnp.bfloat16, label)  
        ok &= check_value_range(arr, label)  
        if verbose:  
            status = "PASS" if ok else "FAIL"  
            print(f"  {status} img[{i}]: shape={arr.shape} dtype={arr.dtype} "  
                  f"min={arr.astype(np.float32).min():.4f} "  
                  f"max={arr.astype(np.float32).max():.4f}")  
        all_ok &= ok  
  
    return all_ok  
  
  
# ---------------------------------------------------------------------------  
# Determinism test: same input → same output  
# ---------------------------------------------------------------------------  
  
def run_determinism_test(model: Kimi_K25_VisionModel,  
                         rng: jax.Array) -> bool:  
    print(f"\n{'='*60}")  
    print("Test: determinism (same input → same output)")  
    tc = TestCase("determinism", [(1, 4, 6)])  
    pixel_values = make_pixel_values(tc.grid_thws, rng)  
    grid_thws    = make_grid_thws(tc.grid_thws)  
  
    out1 = normalise_output(model.vision_tower(pixel_values, grid_thws))  
    out2 = normalise_output(model.vision_tower(pixel_values, grid_thws))  
  
    for i, (a, b) in enumerate(zip(out1, out2)):  
        if not np.array_equal(a, b):  
            print(f"  FAIL: run 1 and run 2 differ for output[{i}]")  
            return False  
    print("  PASS: two runs produce identical outputs")  
    return True  
  
  
# ---------------------------------------------------------------------------  
# Optional: numerical comparison against a saved numpy reference  
# ---------------------------------------------------------------------------  
  
def run_reference_comparison(model: Kimi_K25_VisionModel,  
                              ref_npz_path: str,  
                              atol: float = 0.02,  
                              rtol: float = 0.01) -> bool:  
    """  
    Load a pre-saved reference output (numpy .npz) and compare.  
  
    The .npz should have been saved with:  
        np.savez("reference.npz",  
                 pixel_values=...,   # (L, 3, 14, 14) float32  
                 grid_thws=...,      # (N, 3) int32  
                 output_0=...,       # (H'*W', 4, 1152) float32  ← image 0  
                 output_1=...,       # optional image 1  
                 ...)  
    """  
    print(f"\n{'='*60}")  
    print(f"Test: reference comparison from {ref_npz_path}")  
  
    data = np.load(ref_npz_path)  
    pixel_values = jnp.array(data["pixel_values"], dtype=jnp.bfloat16)  
    grid_thws    = jnp.array(data["grid_thws"],    dtype=jnp.int32)  
  
    output = normalise_output(model.vision_tower(pixel_values, grid_thws))  
  
    all_ok = True  
    i = 0  
    while f"output_{i}" in data:  
        ref = data[f"output_{i}"].astype(np.float32)  
        got = output[i].astype(np.float32)  
        label = f"output_{i}"  
  
        if ref.shape != got.shape:  
            print(f"  FAIL [{label}]: shape mismatch ref={ref.shape} got={got.shape}")  
            all_ok = False  
            i += 1  
            continue  
  
        max_abs_err = np.abs(ref - got).max()  
        max_rel_err = (np.abs(ref - got) / (np.abs(ref) + 1e-6)).max()  
        ok = (max_abs_err <= atol) and (max_rel_err <= rtol)  
        status = "PASS" if ok else "FAIL"  
        print(f"  {status} [{label}]: max_abs_err={max_abs_err:.6f} (atol={atol})  "  
              f"max_rel_err={max_rel_err:.6f} (rtol={rtol})")  
        all_ok &= ok  
        i += 1  
  
    if i == 0:  
        print("  WARN: no output_N keys found in .npz — nothing compared")  
    return all_ok  
  
  
# ---------------------------------------------------------------------------  
# Main  
# ---------------------------------------------------------------------------  
  
def parse_args():  
    p = argparse.ArgumentParser(description="JAX Vision Tower test for Kimi-K2.5")  
    p.add_argument("--weights", default=None,  
                   help="Path to weights directory or file to pass to load_weights(). "  
                        "If omitted, the model is tested with whatever weights it "  
                        "initialises with (shape/logic test only).")  
    p.add_argument("--reference", default=None,  
                   help="Path to a .npz file with reference outputs for numerical "  
                        "comparison (optional).")  
    p.add_argument("--atol", type=float, default=0.02,  
                   help="Absolute tolerance for reference comparison (default 0.02 "  
                        "for bfloat16).")  
    p.add_argument("--rtol", type=float, default=0.01,  
                   help="Relative tolerance for reference comparison.")  
    p.add_argument("--seed", type=int, default=42,  
                   help="JAX PRNG seed for synthetic inputs.")  
    p.add_argument("--verbose", action="store_true", default=True)  
    return p.parse_args()  
  
  
def main():  
    args = parse_args()  
  
    print(f"JAX devices: {jax.devices()}")  
    print(f"Default backend: {jax.default_backend()}")  
  
    # ── instantiate your JAX model ──────────────────────────────────────────  
    # Adjust constructor arguments to match your Flax class signature.  

    config = KimiK25ModelVitConfig
    config.model_path = '/local/kimi'
    config.model_class = Kimi_K25_VisionModel

    model = Kimi_K25_VisionModel(  
        config,
    )  

    devices = jax.devices()  
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))  
  
    with jax.set_mesh(mesh):  
        model = nnx.eval_shape(  
            lambda: Kimi_K25_VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)  
        )  
      
    model.load_weights(config)  
  
    rng = jax.random.PRNGKey(args.seed)  
    results = {}  
  
    # ── run all test cases ──────────────────────────────────────────────────  
    for tc in TEST_CASES:  
        rng, sub = jax.random.split(rng)  
        results[tc.name] = run_test_case(model, tc, sub, verbose=args.verbose)  
  
    # ── determinism ─────────────────────────────────────────────────────────  
    rng, sub = jax.random.split(rng)  
    results["determinism"] = run_determinism_test(model, sub)  
  
    # ── optional reference comparison ────────────────────────────────────────  
    if args.reference is not None:  
        results["reference_comparison"] = run_reference_comparison(  
            model, args.reference, atol=args.atol, rtol=args.rtol  
        )  
  
    # ── summary ─────────────────────────────────────────────────────────────  
    print(f"\n{'='*60}")  
    print("SUMMARY")  
    print(f"{'='*60}")  
    all_passed = True  
    for name, passed in results.items():  
        status = "PASS" if passed else "FAIL"  
        print(f"  [{status}] {name}")  
        all_passed &= passed  
  
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")  
    sys.exit(0 if all_passed else 1)  
  
  
if __name__ == "__main__":  
    main()
