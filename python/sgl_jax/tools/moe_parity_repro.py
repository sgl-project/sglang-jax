import argparse
import contextlib
import difflib
import glob
import json
import os
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.quantization_config import QuantizationConfig, _str_to_dtype
from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe
from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.layers.moe import EPMoE, TopK
from sgl_jax.srt.utils.jax_utils import get_device_name, is_tpu_runtime
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor


@dataclass(frozen=True)
class HFMoEKeys:
    prefix: str
    moe_path: str
    source_expert_pattern: str
    gate_name: str
    up_name: str
    down_name: str

    def key(self, expert_idx: int, name: str) -> str:
        return (
            f"{self.prefix}.{self.moe_path}."
            f"{self.source_expert_pattern.format(i=expert_idx)}.{name}.weight"
        )

    def scale_key(self, expert_idx: int, name: str) -> str:
        return (
            f"{self.prefix}.{self.moe_path}."
            f"{self.source_expert_pattern.format(i=expert_idx)}.{name}.weight_scale"
        )


def _find_index_json(model_dir: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(model_dir, "*.safetensors.index.json")))
    return candidates[0] if candidates else None


def _build_weight_map(model_dir: str) -> dict[str, str] | None:
    index_json = _find_index_json(model_dir)
    if not index_json:
        return None
    with open(index_json, encoding="utf-8") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        return None
    return {k: os.path.join(model_dir, v) for k, v in weight_map.items()}


def _iter_weight_map_keys(model_dir: str) -> tuple[list[str], str | None]:
    weight_map = _build_weight_map(model_dir)
    if weight_map is None:
        return [], None
    index_json = _find_index_json(model_dir)
    return list(weight_map.keys()), index_json


def _format_key_suggestions(
    *,
    missing_key: str,
    keys: list[str],
    substrings: list[str],
    limit: int = 30,
) -> str:
    if not keys:
        return ""

    suggestions: list[str] = []
    substrings = [s for s in substrings if s]
    if substrings:
        for k in keys:
            if any(sub in k for sub in substrings):
                suggestions.append(k)
                if len(suggestions) >= limit:
                    break

    # If substring search didn't find much, fall back to fuzzy matches.
    if len(suggestions) < min(5, limit):
        # Fuzzy matching can be expensive for very large keyspaces.
        pool = keys if len(keys) <= 50_000 else keys[:50_000]
        suggestions.extend(difflib.get_close_matches(missing_key, pool, n=limit, cutoff=0.6))

    # De-dup while preserving order.
    seen: set[str] = set()
    uniq = []
    for s in suggestions:
        if s not in seen:
            uniq.append(s)
            seen.add(s)

    if not uniq:
        return ""
    out = ["", "Did you mean one of these keys?"]
    out.extend(f"  - {k}" for k in uniq[:limit])
    return "\n".join(out)


def _infer_source_expert_pattern(
    *,
    keys: list[str],
    prefix: str,
    moe_path: str,
    weight_name: str,
) -> str | None:
    pat = re.compile(
        rf"^{re.escape(prefix)}\.{re.escape(moe_path)}\.(?P<expert>.+?)\.{re.escape(weight_name)}\.weight$"
    )
    candidates: list[tuple[int, str, int]] = []
    for k in keys:
        m = pat.match(k)
        if not m:
            continue
        expert_str = m.group("expert")
        m2 = re.search(r"(\d+)$", expert_str)
        if not m2:
            continue
        idx = int(m2.group(1))
        pre = expert_str[: m2.start(1)]
        width = len(m2.group(1))
        candidates.append((idx, pre, width))
        if len(candidates) >= 1000:
            break

    if not candidates:
        return None

    # Prefer the pattern that includes expert 0 (common in HF checkpoints).
    candidates.sort(key=lambda t: (t[0] != 0, t[0], len(t[1]), t[2]))
    _, pre, width = candidates[0]
    if width > 1:
        return f"{pre}{{i:0{width}d}}"
    return f"{pre}{{i}}"


def _try_load_stacked_experts(
    model_dir: str,
    hf_keys: HFMoEKeys,
    num_experts: int,
    weight_map: dict[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    gate_key = f"{hf_keys.prefix}.{hf_keys.moe_path}.{hf_keys.gate_name}.weight"
    up_key = f"{hf_keys.prefix}.{hf_keys.moe_path}.{hf_keys.up_name}.weight"
    down_key = f"{hf_keys.prefix}.{hf_keys.moe_path}.{hf_keys.down_name}.weight"

    if weight_map is None:
        return None
    if gate_key not in weight_map or up_key not in weight_map or down_key not in weight_map:
        return None

    wi_0 = _load_safetensor(model_dir, gate_key, weight_map=weight_map)
    wi_1 = _load_safetensor(model_dir, up_key, weight_map=weight_map)
    wo = _load_safetensor(model_dir, down_key, weight_map=weight_map)

    if wi_0.ndim < 2 or wi_1.ndim < 2 or wo.ndim < 2:
        raise ValueError(
            "Stacked expert weights are expected to have an expert dimension (ndim>=2)."
        )
    if wi_0.shape[0] < num_experts or wi_1.shape[0] < num_experts or wo.shape[0] < num_experts:
        raise ValueError(
            f"Stacked expert weights have fewer experts than requested: "
            f"gate={wi_0.shape[0]} up={wi_1.shape[0]} down={wo.shape[0]} requested={num_experts}"
        )

    return wi_0[:num_experts], wi_1[:num_experts], wo[:num_experts]


def _list_keys_by_scanning_safetensors(
    model_dir: str,
    *,
    match: re.Pattern[str] | None,
    limit: int,
) -> tuple[int, int]:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'safetensors'. Install it in your environment to scan keys."
        ) from e

    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        raise RuntimeError(f"No *.safetensors files found under {model_dir}")

    printed = 0
    total_seen = 0
    for st_path in st_files:
        with safe_open(st_path, framework="np", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                total_seen += 1
                if match is not None and not match.search(k):
                    continue
                print(k)
                printed += 1
                if printed >= limit:
                    return printed, total_seen
    return printed, total_seen


def _load_safetensor(model_dir: str, key: str, *, weight_map: dict[str, str] | None) -> np.ndarray:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'safetensors'. Install it in your environment to load HF weights."
        ) from e

    if weight_map is not None and key in weight_map:
        st_path = weight_map[key]
        with safe_open(st_path, framework="np", device="cpu") as f:
            return f.get_tensor(key)

    if weight_map is not None:
        keys = list(weight_map.keys())
        suffix = ".".join(key.split(".")[-2:])  # e.g. "w1.weight"
        expert_hint = ".".join(key.split(".")[-4:])  # e.g. "experts.0.w1.weight"
        suggestion_text = _format_key_suggestions(
            missing_key=key,
            keys=keys,
            substrings=[expert_hint, suffix, "experts.", ".mlp.", ".block_sparse_moe."],
        )
        raise KeyError(
            f"Key not found in safetensors index for {model_dir}: {key}"
            f"\nTip: run with --list-keys --match '<regex>' to find the correct key pattern."
            f"{suggestion_text}"
        )

    # Fallback: scan all safetensors files (slow for large checkpoints).
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        raise RuntimeError(f"No *.safetensors files found under {model_dir}")
    for st_path in st_files:
        with safe_open(st_path, framework="np", device="cpu") as f:
            if key in f.keys():  # noqa: SIM118
                return f.get_tensor(key)
    raise KeyError(f"Key not found in {model_dir}: {key}")


def _try_load_safetensor_key(
    model_dir: str,
    key: str,
    *,
    weight_map: dict[str, str] | None,
) -> np.ndarray | None:
    if weight_map is None:
        return None
    if key not in weight_map:
        return None
    return _load_safetensor(model_dir, key, weight_map=weight_map)


def _maybe_load_router_params(
    model_dir: str,
    *,
    prefix: str,
    weight_map: dict[str, str] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gate_w_key = f"{prefix}.mlp.gate.weight"
    expert_bias_key = f"{prefix}.mlp.gate.expert_bias"
    gate_w = _try_load_safetensor_key(model_dir, gate_w_key, weight_map=weight_map)
    expert_bias = _try_load_safetensor_key(model_dir, expert_bias_key, weight_map=weight_map)
    return gate_w, expert_bias


def _maybe_load_shared_experts(
    model_dir: str,
    *,
    prefix: str,
    weight_map: dict[str, str] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    w1_key = f"{prefix}.mlp.shared_experts.gate_proj.weight"
    w3_key = f"{prefix}.mlp.shared_experts.up_proj.weight"
    w2_key = f"{prefix}.mlp.shared_experts.down_proj.weight"
    w1 = _try_load_safetensor_key(model_dir, w1_key, weight_map=weight_map)
    w2 = _try_load_safetensor_key(model_dir, w2_key, weight_map=weight_map)
    w3 = _try_load_safetensor_key(model_dir, w3_key, weight_map=weight_map)
    if w1 is None or w2 is None or w3 is None:
        return None, None, None
    return w1, w2, w3


def _apply_score_function(x: jax.Array, score_func: str | None) -> jax.Array:
    if not score_func:
        return x
    if score_func == "softmax":
        return jax.nn.softmax(x, axis=-1)
    if score_func == "sigmoid":
        return jax.nn.sigmoid(x)
    if score_func == "tanh":
        return jax.nn.tanh(x)
    raise ValueError(f"Unsupported score_function: {score_func}")


def _rms_norm(x: jax.Array, *, weight: jax.Array | None, eps: float) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    y = x_f32 / rms
    if weight is not None:
        y = y * weight.astype(jnp.float32)
    return y.astype(x.dtype)


def _make_deterministic_router_logits(
    *,
    key: jax.Array,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    dtype: jnp.dtype,
) -> jax.Array:
    router_logits = jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)
    token_keys = jax.random.split(key, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    return (router_logits + one_hot).astype(dtype)


def _quantize_fused_weights(
    w_dtype: jnp.dtype,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_size: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    w1_q, w1_scale_3d = quantize_tensor(w_dtype, w1, axis=1, block_size=block_size)
    w3_q, w3_scale_3d = quantize_tensor(w_dtype, w3, axis=1, block_size=block_size)
    w2_q, w2_scale_3d = quantize_tensor(w_dtype, w2, axis=1, block_size=block_size)

    w1_scale = w1_scale_3d.reshape(
        w1_scale_3d.shape[0], w1_scale_3d.shape[1], 1, w1_scale_3d.shape[2]
    )
    w3_scale = w3_scale_3d.reshape(
        w3_scale_3d.shape[0], w3_scale_3d.shape[1], 1, w3_scale_3d.shape[2]
    )
    w2_scale = w2_scale_3d.reshape(
        w2_scale_3d.shape[0], w2_scale_3d.shape[1], 1, w2_scale_3d.shape[2]
    )
    return w1_q, w2_q, w3_q, w1_scale, w2_scale, w3_scale


def _shared_mlp_fp8_static(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    w1_q: jax.Array,  # (se_inter, hidden)
    w2_q: jax.Array,  # (hidden, se_inter)
    w3_q: jax.Array,  # (se_inter, hidden)
    w1_scale: jax.Array,  # (se_inter,)
    w2_scale: jax.Array,  # (hidden,)
    w3_scale: jax.Array,  # (se_inter,)
    act_fn: str,
    compute_dtype: jnp.dtype | None = None,
) -> jax.Array:
    gate = QuantizedLinear(
        weight_q=w1_q,
        weight_scale=w1_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=(None, "tensor"),
        params_dtype=jnp.bfloat16,
        scope_name="shared_gate_proj",
    )
    up = QuantizedLinear(
        weight_q=w3_q,
        weight_scale=w3_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=(None, "tensor"),
        params_dtype=jnp.bfloat16,
        scope_name="shared_up_proj",
    )
    down = QuantizedLinear(
        weight_q=w2_q,
        weight_scale=w2_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        scope_name="shared_down_proj",
    )

    a1, _ = gate(x)
    a2, _ = up(x)
    if act_fn == "silu":
        inter = a2 * jax.nn.silu(a1)
    elif act_fn == "gelu":
        inter = a2 * jax.nn.gelu(a1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")
    out, _ = down(inter)
    return out


def _shared_mlp_fp8_static_intermediates(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    w1_q: jax.Array,  # (se_inter, hidden)
    w2_q: jax.Array,  # (hidden, se_inter)
    w3_q: jax.Array,  # (se_inter, hidden)
    w1_scale: jax.Array,  # (se_inter,)
    w2_scale: jax.Array,  # (hidden,)
    w3_scale: jax.Array,  # (se_inter,)
    act_fn: str,
    compute_dtype: jnp.dtype | None = None,
) -> dict[str, jax.Array]:
    gate = QuantizedLinear(
        weight_q=w1_q,
        weight_scale=w1_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=(None, "tensor"),
        params_dtype=jnp.bfloat16,
        scope_name="shared_gate_proj",
    )
    up = QuantizedLinear(
        weight_q=w3_q,
        weight_scale=w3_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=(None, "tensor"),
        params_dtype=jnp.bfloat16,
        scope_name="shared_up_proj",
    )
    down = QuantizedLinear(
        weight_q=w2_q,
        weight_scale=w2_scale,
        bias=None,
        activation_dtype=None,
        compute_dtype=compute_dtype,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        scope_name="shared_down_proj",
    )

    a1, _ = gate(x)
    a2, _ = up(x)
    if act_fn == "silu":
        inter = a2 * jax.nn.silu(a1)
    elif act_fn == "gelu":
        inter = a2 * jax.nn.gelu(a1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")
    out, _ = down(inter)
    return {"a1": a1, "a2": a2, "inter": inter, "out": out}


def _shared_mlp_bf16(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    w1: jax.Array,  # (hidden, se_inter)
    w2: jax.Array,  # (se_inter, hidden)
    w3: jax.Array,  # (hidden, se_inter)
    act_fn: str,
) -> jax.Array:
    def _linear(lhs: jax.Array, rhs: jax.Array, kernel_axes: tuple[str | None, str | None]):
        output_pspec = P(*([None] * (lhs.ndim - 1)), kernel_axes[-1])
        output_sharding = NamedSharding(mesh, output_pspec)
        return lax.dot_general(
            lhs,
            rhs,
            (((lhs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=lhs.dtype,
            out_sharding=output_sharding,
        )

    a1 = _linear(x, w1, (None, "tensor"))
    a2 = _linear(x, w3, (None, "tensor"))
    if act_fn == "silu":
        inter = a2 * jax.nn.silu(a1)
    elif act_fn == "gelu":
        inter = a2 * jax.nn.gelu(a1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")
    return _linear(inter, w2, ("tensor", None))


def _shared_mlp_bf16_intermediates(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    w1: jax.Array,  # (hidden, se_inter)
    w2: jax.Array,  # (se_inter, hidden)
    w3: jax.Array,  # (hidden, se_inter)
    act_fn: str,
) -> dict[str, jax.Array]:
    def _linear(lhs: jax.Array, rhs: jax.Array, kernel_axes: tuple[str | None, str | None]):
        output_pspec = P(*([None] * (lhs.ndim - 1)), kernel_axes[-1])
        output_sharding = NamedSharding(mesh, output_pspec)
        return lax.dot_general(
            lhs,
            rhs,
            (((lhs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=lhs.dtype,
            out_sharding=output_sharding,
        )

    a1 = _linear(x, w1, (None, "tensor"))
    a2 = _linear(x, w3, (None, "tensor"))
    if act_fn == "silu":
        inter = a2 * jax.nn.silu(a1)
    elif act_fn == "gelu":
        inter = a2 * jax.nn.gelu(a1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")
    out = _linear(inter, w2, ("tensor", None))
    return {"a1": a1, "a2": a2, "inter": inter, "out": out}


def _shared_mlp_ref_fp32_intermediates(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    w1: jax.Array,  # (hidden, se_inter) possibly fp8
    w2: jax.Array,  # (se_inter, hidden) possibly fp8
    w3: jax.Array,  # (hidden, se_inter) possibly fp8
    w1_scale: jax.Array | None,  # (1, 1, se_inter)
    w2_scale: jax.Array | None,  # (1, 1, hidden)
    w3_scale: jax.Array | None,  # (1, 1, se_inter)
    act_fn: str,
) -> dict[str, jax.Array]:
    """Shared MLP "reference" intermediates using fused-style semantics.

    This is intentionally *not* the end-to-end QuantizedLinear path. Instead it
    models the fused kernel's shared branch more closely:
    - Dot inputs use the original dtypes (typically bf16 activations and fp8 weights),
      accumulating in fp32 (`preferred_element_type=float32`).
    - Per-channel (1D) static scales are applied *after* the dot, matching the
      kernel's `acc *= scale_slice` pattern (important when fp8 dot math is used).
    """

    def _linear(lhs: jax.Array, rhs: jax.Array, kernel_axes: tuple[str | None, str | None]):
        output_pspec = P(*([None] * (lhs.ndim - 1)), kernel_axes[-1])
        output_sharding = NamedSharding(mesh, output_pspec)
        return lax.dot_general(
            lhs,
            rhs,
            (((lhs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            out_sharding=output_sharding,
        )

    a1 = _linear(x, w1, (None, "tensor")).astype(jnp.float32)
    a2 = _linear(x, w3, (None, "tensor")).astype(jnp.float32)
    if w1_scale is not None:
        a1 *= w1_scale.astype(jnp.float32)[0, 0, :][None, :]
    if w3_scale is not None:
        a2 *= w3_scale.astype(jnp.float32)[0, 0, :][None, :]
    if act_fn == "silu":
        inter = a2 * jax.nn.silu(a1)
    elif act_fn == "gelu":
        inter = a2 * jax.nn.gelu(a1)
    else:
        raise ValueError(f"Unsupported act_fn: {act_fn}")
    out = _linear(inter, w2, ("tensor", None)).astype(jnp.float32)
    if w2_scale is not None:
        out *= w2_scale.astype(jnp.float32)[0, 0, :][None, :]
    return {"a1": a1, "a2": a2, "inter": inter, "out": out}


def _describe_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(b.astype(np.float32)), 1e-6)
    rel = abs_diff / denom
    return {
        "max_abs": float(abs_diff.max(initial=0.0)),
        "p99_abs": float(np.quantile(abs_diff, 0.99)),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel.max(initial=0.0)),
        "p99_rel": float(np.quantile(rel, 0.99)),
        "mean_rel": float(rel.mean()),
    }


def _load_or_extract_weights(
    *,
    model_dir: str,
    hf_keys: HFMoEKeys,
    num_experts: int,
    cache_npz: str | None,
    _allow_infer: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_npz and os.path.exists(cache_npz):
        data = np.load(cache_npz, allow_pickle=False)
        meta = None
        if "meta" in data:
            try:
                meta = data["meta"].item()
            except Exception:
                meta = None
        if isinstance(meta, dict):
            cached_num_experts = meta.get("num_experts")
            if cached_num_experts is not None and int(cached_num_experts) != int(num_experts):
                raise ValueError(
                    f"--cache-npz mismatch: cache has num_experts={cached_num_experts}, "
                    f"requested num_experts={num_experts}. Delete the cache or use a new path."
                )
            cached_model_dir = meta.get("model_dir")
            if cached_model_dir is not None and str(cached_model_dir) != str(model_dir):
                raise ValueError(
                    f"--cache-npz mismatch: cache was created for model_dir={cached_model_dir}, "
                    f"requested model_dir={model_dir}. Delete the cache or use a new path."
                )
        wi_0 = data["wi_0"]
        wi_1 = data["wi_1"]
        wo = data["wo"]
        return wi_0, wi_1, wo

    weight_map = _build_weight_map(model_dir)

    stacked = _try_load_stacked_experts(model_dir, hf_keys, num_experts, weight_map)
    if stacked is not None:
        wi_0, wi_1, wo = stacked
        if cache_npz:
            os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
            np.savez(cache_npz, wi_0=wi_0, wi_1=wi_1, wo=wo)
        return wi_0, wi_1, wo

    wi_0_list, wi_1_list, wo_list = [], [], []
    try:
        for e in range(num_experts):
            gate_key = hf_keys.key(e, hf_keys.gate_name)
            up_key = hf_keys.key(e, hf_keys.up_name)
            down_key = hf_keys.key(e, hf_keys.down_name)

            wi_0_list.append(_load_safetensor(model_dir, gate_key, weight_map=weight_map))
            wi_1_list.append(_load_safetensor(model_dir, up_key, weight_map=weight_map))
            wo_list.append(_load_safetensor(model_dir, down_key, weight_map=weight_map))
    except KeyError:
        if weight_map is None or not _allow_infer:
            raise

        keys = list(weight_map.keys())
        inferred = _infer_source_expert_pattern(
            keys=keys,
            prefix=hf_keys.prefix,
            moe_path=hf_keys.moe_path,
            weight_name=hf_keys.gate_name,
        )
        if inferred and inferred != hf_keys.source_expert_pattern:
            print(
                "[keys] could not find per-expert keys with "
                f'--source-expert-pattern "{hf_keys.source_expert_pattern}". '
                f'Trying inferred pattern: "{inferred}"'
            )
            hf_keys2 = HFMoEKeys(
                prefix=hf_keys.prefix,
                moe_path=hf_keys.moe_path,
                source_expert_pattern=inferred,
                gate_name=hf_keys.gate_name,
                up_name=hf_keys.up_name,
                down_name=hf_keys.down_name,
            )
            return _load_or_extract_weights(
                model_dir=model_dir,
                hf_keys=hf_keys2,
                num_experts=num_experts,
                cache_npz=cache_npz,
                _allow_infer=False,
            )

        raise

    wi_0 = np.stack(wi_0_list, axis=0)
    wi_1 = np.stack(wi_1_list, axis=0)
    wo = np.stack(wo_list, axis=0)

    if cache_npz:
        os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
        np.savez(
            cache_npz,
            wi_0=wi_0,
            wi_1=wi_1,
            wo=wo,
            meta=np.asarray(
                {"model_dir": model_dir, "num_experts": int(num_experts)}, dtype=object
            ),
        )

    return wi_0, wi_1, wo


def _load_static_scales(
    model_dir: str,
    *,
    hf_keys: HFMoEKeys,
    num_experts: int,
    weight_map: dict[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    wi_0_s, wi_1_s, wo_s = [], [], []
    for e in range(num_experts):
        gate_s_key = hf_keys.scale_key(e, hf_keys.gate_name)
        up_s_key = hf_keys.scale_key(e, hf_keys.up_name)
        down_s_key = hf_keys.scale_key(e, hf_keys.down_name)
        s0 = _try_load_safetensor_key(model_dir, gate_s_key, weight_map=weight_map)
        s1 = _try_load_safetensor_key(model_dir, up_s_key, weight_map=weight_map)
        s2 = _try_load_safetensor_key(model_dir, down_s_key, weight_map=weight_map)
        if s0 is None or s1 is None or s2 is None:
            return None
        wi_0_s.append(np.asarray(s0).reshape(-1))
        wi_1_s.append(np.asarray(s1).reshape(-1))
        wo_s.append(np.asarray(s2).reshape(-1))
    return np.stack(wi_0_s, axis=0), np.stack(wi_1_s, axis=0), np.stack(wo_s, axis=0)


def _repeat_scales_for_fused(
    scales: np.ndarray,
    *,
    num_blocks: int,
) -> np.ndarray:
    # Convert (E, out_dim) -> (E, num_blocks, 1, out_dim) where the scale is shared across blocks.
    if scales.ndim != 2:
        raise ValueError(f"Expected scales to have shape (E, out_dim), got {scales.shape}")
    s = scales.astype(np.float32)
    s = s.reshape(s.shape[0], 1, 1, s.shape[1])
    if num_blocks <= 0:
        raise ValueError(f"Expected num_blocks > 0, got {num_blocks}")
    return np.repeat(s, repeats=num_blocks, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quick parity repro between fused_ep_moe (fused) and EPMoE (epmoe) "
            "on a single MoE layer's weights."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Local HF-style safetensors directory")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--act-fn", type=str, default="silu", choices=("silu", "gelu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokens-apply-post-attn-rmsnorm",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Apply a post-attention RMSNorm to synthetic tokens to better match real MoE inputs. "
            "When --from-config and model_type=bailing_moe, defaults to True."
        ),
    )
    parser.add_argument(
        "--tokens-rmsnorm-from-ckpt",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "When applying RMSNorm, multiply by the checkpoint scale vector if available. "
            "When --from-config and model_type=bailing_moe, defaults to True."
        ),
    )
    parser.add_argument(
        "--tokens-rmsnorm-eps",
        type=float,
        default=1e-6,
        help="Epsilon used for RMSNorm when --tokens-apply-post-attn-rmsnorm is set.",
    )

    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--moe-path", type=str, default="mlp")
    parser.add_argument("--source-expert-pattern", type=str, default="experts.{i}")
    parser.add_argument("--gate-name", type=str, default="gate_proj")
    parser.add_argument("--up-name", type=str, default="up_proj")
    parser.add_argument("--down-name", type=str, default="down_proj")

    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Read MoE settings from --model-dir/config.json and override related flags.",
    )

    parser.add_argument("--use-grouped-topk", action="store_true")
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--top-k-groups", type=int, default=1)

    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List safetensors keys using the *.safetensors.index.json (fast).",
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Regex filter used with --list-keys (e.g. 'layers\\.0\\..*w1\\.weight$').",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=200,
        help="Max keys to print for --list-keys.",
    )

    parser.add_argument("--renormalize-topk-logits", action="store_true")
    parser.add_argument("--routed-scaling-factor", type=float, default=None)
    parser.add_argument(
        "--score-function",
        type=str,
        default=None,
        choices=("softmax", "sigmoid", "tanh"),
        help="Optional router score function applied before top-k (matches GateLogit score_func).",
    )
    parser.add_argument(
        "--router-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="Dtype for router logits passed to the MoE kernels.",
    )
    parser.add_argument(
        "--subc-quant-wsz",
        type=int,
        default=256,
        help="Sub-channel quantization block size used when --moe-weight-dtype is set.",
    )

    parser.add_argument(
        "--moe-weight-dtype",
        type=str,
        default=None,
        help="Enable MoE weight quantization (e.g. float8_e4m3fn, float8_e5m2, int8).",
    )
    parser.add_argument(
        "--static-checkpoint",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Prefer using checkpoint-provided `*.weight_scale` tensors (static checkpoint) "
            "instead of online quantization. Default is auto (tries static first)."
        ),
    )
    parser.add_argument(
        "--cache-npz",
        type=str,
        default=None,
        help="Optional cache for extracted MoE weights (npz). Speeds up repeated runs.",
    )
    parser.add_argument(
        "--dump-npz",
        type=str,
        default=None,
        help="Optional dump of inputs/outputs for sharing (npz).",
    )
    parser.add_argument(
        "--debug-topn",
        type=int,
        default=0,
        help="Print the top-N largest per-element output diffs (0 disables).",
    )
    parser.add_argument(
        "--debug-metric",
        type=str,
        default="abs",
        choices=("abs", "rel"),
        help="Metric used for --debug-topn ranking.",
    )
    parser.add_argument(
        "--debug-rel-eps",
        type=float,
        default=1e-6,
        help="Epsilon for relative diff denominator: |a-b|/max(|b|, eps).",
    )
    parser.add_argument(
        "--dump-shared-intermediates",
        type=str,
        default=None,
        help=(
            "Dump shared_experts intermediates (a1/a2/inter/out) for ep/ref, plus fused shared output "
            "(fused_total - fused_no_shared). Writes an npz."
        ),
    )
    parser.add_argument(
        "--shared-qmm-fp32-accum",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "When shared_experts uses the static FP8 QuantizedLinear path, use float32 accumulation "
            "inside the quantized matmul (preferred_element_type=float32) to better align with fused "
            "shared math; defaults to False to match end-to-end QuantizedLinear behavior."
        ),
    )
    parser.add_argument(
        "--debug-use-ref-shared",
        action="store_true",
        help="[Debug] Disable internal shared experts in fused kernel, and manually add the Ref-MLP output instead.",
    )
    parser.add_argument(
        "--load-input", type=str, default=None, help="Path to input.npy to replace random tokens"
    )
    args = parser.parse_args()
    user_set_tokens_apply_rmsnorm = args.tokens_apply_post_attn_rmsnorm is not None
    user_set_tokens_rmsnorm_from_ckpt = args.tokens_rmsnorm_from_ckpt is not None

    prefix = f"model.layers.{args.layer_idx}" if args.prefix is None else args.prefix

    if args.list_keys:
        pat = re.compile(args.match) if args.match else None
        keys, index_json = _iter_weight_map_keys(args.model_dir)
        if keys:
            printed = 0
            for k in keys:
                if pat is not None and not pat.search(k):
                    continue
                print(k)
                printed += 1
                if printed >= args.list_limit:
                    break
            src = index_json if index_json else "<unknown index>"
            print(f"[list-keys] printed={printed} total_keys={len(keys)} index={src}")
        else:
            printed, total_seen = _list_keys_by_scanning_safetensors(
                args.model_dir, match=pat, limit=args.list_limit
            )
            print(
                f"[list-keys] printed={printed} scanned_keys={total_seen} "
                "(no *.safetensors.index.json found)"
            )
        return 0

    if args.from_config:
        config_path = os.path.join(args.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"--from-config is set but config.json not found: {config_path}"
            )
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        def _first_int(keys: list[str]) -> int | None:
            for k in keys:
                v = cfg.get(k)
                if v is None:
                    continue
                try:
                    return int(v)
                except Exception:
                    continue
            return None

        def _first_float(keys: list[str]) -> float | None:
            for k in keys:
                v = cfg.get(k)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
            return None

        def _first_bool(keys: list[str]) -> bool | None:
            for k in keys:
                if k in cfg:
                    return bool(cfg.get(k))
            return None

        cfg_num_experts = _first_int(
            [
                "num_local_experts",
                "num_experts",
                "n_experts",
                "moe_num_experts",
            ]
        )
        cfg_topk = _first_int(
            [
                "num_experts_per_tok",
                "num_experts_per_token",
                "moe_topk",
                "top_k",
                "topk",
            ]
        )
        cfg_num_groups = _first_int(
            [
                "num_expert_group",
                "num_expert_groups",
                "num_groups",
                "moe_num_groups",
                "n_group",
            ]
        )
        cfg_topk_groups = _first_int(
            [
                "topk_group",
                "top_k_groups",
                "topk_groups",
            ]
        )
        cfg_use_grouped = _first_bool(["use_grouped_topk"])

        cfg_renorm = _first_bool(
            [
                "norm_topk_prob",
                "renormalize_topk_logits",
                "renormalize",
                "moe_renormalize",
            ]
        )
        cfg_routed_scale = _first_float(["routed_scaling_factor", "moe_routed_scaling_factor"])
        cfg_score_func = cfg.get("score_function")
        cfg_router_dtype = cfg.get("router_dtype")
        cfg_first_k_dense_replace = _first_int(["first_k_dense_replace"])
        cfg_rms_norm_eps = _first_float(["rms_norm_eps"])

        if cfg_num_experts is not None:
            args.num_experts = cfg_num_experts
        if cfg_topk is not None:
            args.top_k = cfg_topk
        if cfg_num_groups is not None:
            args.num_groups = cfg_num_groups
        if cfg_topk_groups is not None:
            args.top_k_groups = cfg_topk_groups

        if cfg_use_grouped is not None:
            args.use_grouped_topk = cfg_use_grouped
        else:
            args.use_grouped_topk = args.num_groups > 1 or args.top_k_groups > 1

        if cfg_renorm is not None:
            args.renormalize_topk_logits = cfg_renorm
        if cfg_routed_scale is not None:
            args.routed_scaling_factor = cfg_routed_scale
        if isinstance(cfg_score_func, str) and cfg_score_func in ("softmax", "sigmoid", "tanh"):
            args.score_function = cfg_score_func
        if cfg_router_dtype == "fp32":
            args.router_dtype = "float32"
        if cfg_rms_norm_eps is not None:
            args.tokens_rmsnorm_eps = cfg_rms_norm_eps

        def _is_bailing_cfg(cfg: dict) -> bool:
            model_type = cfg.get("model_type")
            arch = cfg.get("architectures")
            auto_map = cfg.get("auto_map")
            return (
                (isinstance(model_type, str) and "bailing" in model_type.lower())
                or (
                    isinstance(arch, list)
                    and any(isinstance(a, str) and "bailing" in a.lower() for a in arch)
                )
                or (
                    isinstance(auto_map, dict)
                    and any(
                        isinstance(v, str) and "bailing" in v.lower() for v in auto_map.values()
                    )
                )
            )

        # BailingMoE-style checkpoints: experts are stored under `mlp.experts.{i}.{gate,up,down}_proj`.
        # If the user passed a different layout, prefer the config-driven layout.
        if _is_bailing_cfg(cfg):
            args.moe_path = "mlp"
            args.source_expert_pattern = "experts.{i}"
            args.gate_name = "gate_proj"
            args.up_name = "up_proj"
            args.down_name = "down_proj"

            # Prefer first_k_dense_replace when present; otherwise infer the first MoE layer by scanning keys.
            if args.prefix is None:
                inferred_moe_layer = None
                weight_map = _build_weight_map(args.model_dir)
                if weight_map is not None:
                    rx = re.compile(
                        r"^model\.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight(_scale)?$"
                    )
                    idxs = []
                    for k in weight_map:
                        m = rx.match(k)
                        if m:
                            with contextlib.suppress(Exception):
                                idxs.append(int(m.group(1)))
                    inferred_moe_layer = min(idxs) if idxs else None

                target_layer = None
                if cfg_first_k_dense_replace is not None:
                    target_layer = cfg_first_k_dense_replace
                elif inferred_moe_layer is not None:
                    target_layer = inferred_moe_layer

                if target_layer is not None and args.layer_idx != target_layer:
                    print(
                        f"[config] using layer_idx={target_layer} for MoE (requested layer_idx={args.layer_idx})."
                    )
                    args.layer_idx = target_layer
                    prefix = f"model.layers.{args.layer_idx}"

            # Default synthetic tokens to match BailingMoE's MoE input more closely: post-attn RMSNorm
            # (and its checkpoint weight) unless the user explicitly turned it on/off.
            if not user_set_tokens_apply_rmsnorm:
                args.tokens_apply_post_attn_rmsnorm = True
            if not user_set_tokens_rmsnorm_from_ckpt:
                args.tokens_rmsnorm_from_ckpt = True

        print(
            "[config] "
            f"layer_idx={args.layer_idx} "
            f"num_experts={args.num_experts} top_k={args.top_k} "
            f"use_grouped_topk={args.use_grouped_topk} num_groups={args.num_groups} "
            f"top_k_groups={args.top_k_groups} renorm={args.renormalize_topk_logits} "
            f"routed_scaling_factor={args.routed_scaling_factor} score_function={args.score_function} "
            f"router_dtype={args.router_dtype}"
        )

    # Finalize defaults for boolean-optional flags when user didn't set them.
    if args.tokens_apply_post_attn_rmsnorm is None:
        args.tokens_apply_post_attn_rmsnorm = False
    if args.tokens_rmsnorm_from_ckpt is None:
        args.tokens_rmsnorm_from_ckpt = False

    hf_keys = HFMoEKeys(
        prefix=prefix,
        moe_path=args.moe_path,
        source_expert_pattern=args.source_expert_pattern,
        gate_name=args.gate_name,
        up_name=args.up_name,
        down_name=args.down_name,
    )

    try:
        device_name = get_device_name()
    except Exception:
        device_name = jax.devices()[0].device_kind if jax.devices() else "unknown"
    print(
        f"[env] backend={jax.default_backend()} device={device_name} num_devices={len(jax.devices())}"
    )

    if not is_tpu_runtime():
        raise RuntimeError("This repro expects TPU for fused_ep_moe; current runtime is not TPU.")

    mesh = create_device_mesh(ici_parallelism=[-1, 1], dcn_parallelism=[1, 1])
    ep_size = int(mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1))
    if args.num_tokens % ep_size != 0:
        raise ValueError(
            f"--num-tokens ({args.num_tokens}) must be divisible by ep_size ({ep_size})."
        )

    weight_map = _build_weight_map(args.model_dir)
    static_requested = True if args.static_checkpoint is None else bool(args.static_checkpoint)
    static_scales = _load_static_scales(
        args.model_dir,
        hf_keys=hf_keys,
        num_experts=args.num_experts,
        weight_map=weight_map,
    )
    use_static = static_requested and static_scales is not None
    cache_npz = None if use_static else args.cache_npz
    if use_static and args.cache_npz:
        print("[static] ignoring --cache-npz (avoid mixing cached BF16 weights with static scales)")

    wi_0_np, wi_1_np, wo_np = _load_or_extract_weights(
        model_dir=args.model_dir,
        hf_keys=hf_keys,
        num_experts=args.num_experts,
        cache_npz=cache_npz,
    )
    if use_static:
        w_dtype = np.asarray(wi_0_np).dtype
        is_fp8 = w_dtype in (
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.float8_e5m2,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        )
        if not is_fp8:
            print(
                f"[static] checkpoint=scales_found but weights dtype is {w_dtype}; disabling static path"
            )
            use_static = False
            static_scales = None

    # EPMoE expects checkpoint layout (no transpose):
    # wi_0/wi_1: (E, inter, hidden), wo: (E, hidden, inter)
    hidden_size = int(wi_0_np.shape[-1])
    intermediate_size = int(wi_0_np.shape[-2])

    key = jax.random.key(args.seed)
    if args.load_input:
        print(f"[input] Loading custom tokens from {args.load_input}...")
        loaded_data = jnp.load(args.load_input)

        if isinstance(loaded_data, np.lib.npyio.NpzFile):
            keys = loaded_data.files
            target_key = next((k for k in ["input", "tokens", "data"] if k in keys), keys[0])
            print(f"[input] Detected .npz file, using key: '{target_key}'")
            loaded_data = loaded_data[target_key]

        if loaded_data.ndim > 2:
            loaded_data = loaded_data.reshape(-1, loaded_data.shape[-1])

        if loaded_data.shape[-1] != hidden_size:
            raise ValueError(
                f"Input hidden_size mismatch! File: {loaded_data.shape[-1]}, "
                f"Model: {hidden_size}. (Check if you dumped the correct layer output)"
            )

        input_np = np.array(loaded_data[: args.num_tokens])

        if input_np.shape[0] < args.num_tokens:
            print(
                f"[warning] Input file only has {input_np.shape[0]} tokens, but --num-tokens={args.num_tokens}."
            )
            print(f"[warning] Using actual available tokens: {input_np.shape[0]}")
            args.num_tokens = input_np.shape[0]

        replicated_sharding = NamedSharding(mesh, P())
        tokens_f32 = jax.device_put(jnp.asarray(input_np, dtype=jnp.float32), replicated_sharding)
    else:
        tokens_f32 = jax.random.normal(key, (args.num_tokens, hidden_size), dtype=jnp.float32)
    if args.tokens_apply_post_attn_rmsnorm:
        norm_weight = None
        if args.tokens_rmsnorm_from_ckpt and weight_map is not None:
            norm_key = f"{prefix}.post_attention_layernorm.weight"
            norm_weight = _try_load_safetensor_key(args.model_dir, norm_key, weight_map=weight_map)
        print(
            "[tokens] post_attn_rmsnorm=yes "
            f"weight={'from_ckpt' if norm_weight is not None else 'none'} "
            f"eps={args.tokens_rmsnorm_eps}"
        )
        tokens = _rms_norm(
            tokens_f32.astype(jnp.bfloat16),
            weight=(None if norm_weight is None else jnp.asarray(norm_weight, dtype=jnp.bfloat16)),
            eps=float(args.tokens_rmsnorm_eps),
        )
    else:
        tokens = tokens_f32.astype(jnp.bfloat16)
    gate_w_np, router_bias_np = _maybe_load_router_params(
        args.model_dir,
        prefix=prefix,
        weight_map=weight_map,
    )
    router_dtype = jnp.float32 if args.router_dtype == "float32" else jnp.bfloat16
    if gate_w_np is not None:
        gate_w = jnp.asarray(gate_w_np, dtype=jnp.float32)
        if gate_w.shape == (args.num_experts, hidden_size):
            gate_w = gate_w.T
        if gate_w.shape != (hidden_size, args.num_experts):
            raise ValueError(
                f"Unexpected gate weight shape: {gate_w.shape}, expected {(hidden_size, args.num_experts)}"
            )
        router_logits = _apply_score_function(
            (tokens.astype(jnp.float32) @ gate_w).astype(jnp.float32),
            args.score_function,
        ).astype(router_dtype)
        if router_bias_np is not None:
            router_bias = jax.device_put(
                jnp.asarray(router_bias_np, dtype=jnp.bfloat16), NamedSharding(mesh, P())
            )
        else:
            router_bias = None
        print(
            f"[router] from_checkpoint gate_weight=yes expert_bias={'yes' if router_bias_np is not None else 'no'}"
        )
    else:
        router_logits = _apply_score_function(
            _make_deterministic_router_logits(
                key=jax.random.key(args.seed + 1),
                num_tokens=args.num_tokens,
                num_experts=args.num_experts,
                top_k=args.top_k,
                dtype=router_dtype,
            ),
            args.score_function,
        )
        router_bias = None
        print("[router] from_checkpoint gate_weight=no (using synthetic router logits)")

    w1_shared_np, w2_shared_np, w3_shared_np = _maybe_load_shared_experts(
        args.model_dir,
        prefix=prefix,
        weight_map=weight_map,
    )
    w1_shared_scale_np = w2_shared_scale_np = w3_shared_scale_np = None
    if weight_map is not None:
        w1_shared_scale_np = _try_load_safetensor_key(
            args.model_dir,
            f"{prefix}.mlp.shared_experts.gate_proj.weight_scale",
            weight_map=weight_map,
        )
        w3_shared_scale_np = _try_load_safetensor_key(
            args.model_dir,
            f"{prefix}.mlp.shared_experts.up_proj.weight_scale",
            weight_map=weight_map,
        )
        w2_shared_scale_np = _try_load_safetensor_key(
            args.model_dir,
            f"{prefix}.mlp.shared_experts.down_proj.weight_scale",
            weight_map=weight_map,
        )
    has_shared = w1_shared_np is not None
    has_shared_scales = (
        w1_shared_scale_np is not None
        and w2_shared_scale_np is not None
        and w3_shared_scale_np is not None
    )
    print(
        f"[shared] from_checkpoint shared_experts={'yes' if has_shared else 'no'} "
        f"shared_scales={'yes' if has_shared_scales else 'no'}"
    )

    # Match end-to-end epmoe selection logic (BailingMoE uses TopK module).
    print("compute topk weights/ids")
    topk = TopK(
        topk=args.top_k,
        renormalize=args.renormalize_topk_logits,
        num_expert_group=(args.num_groups if args.use_grouped_topk else 0),
        topk_group=(args.top_k_groups if args.use_grouped_topk else 0),
        routed_scaling_factor=args.routed_scaling_factor,
    )
    with jax.set_mesh(mesh):
        topk_weights, topk_ids = topk(router_logits, router_bias)

    print("generate moe weights")
    moe_weight_dtype = _str_to_dtype(args.moe_weight_dtype)
    if use_static:
        w_dtype = np.asarray(wi_0_np).dtype
        if w_dtype == ml_dtypes.float8_e4m3fn:
            moe_weight_dtype = jnp.float8_e4m3fn
        elif w_dtype == ml_dtypes.float8_e5m2:
            moe_weight_dtype = jnp.float8_e5m2
        else:
            raise ValueError(f"Unsupported static checkpoint weight dtype: {w_dtype}")
        subc_quant_wsz = 256
    else:
        subc_quant_wsz = int(args.subc_quant_wsz) if moe_weight_dtype is not None else None
    quant_config = QuantizationConfig(
        linear_rules=[],
        moe_weight_dtype=moe_weight_dtype,
        moe_activation_dtype=None,
        is_static_checkpoint=use_static,
    )

    # Construct the module structure without materializing giant random params.
    # This matches the loader pattern and avoids eager param initialization.
    with jax.set_mesh(mesh):
        epmoe = nnx.eval_shape(
            lambda: EPMoE(
                hidden_size=hidden_size,
                num_experts=args.num_experts,
                num_experts_per_tok=args.top_k,
                ep_size=ep_size,
                mesh=mesh,
                intermediate_dim=intermediate_size,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                activation=args.act_fn,
                quantization_config=quant_config,
            )
        )

    print("load epmoe weights")
    # Put MoE weights on device with the intended sharding to avoid a large replicated
    # array on a single device (important for FP8 quantization to not OOM).
    expert_w_np = np.asarray(wi_0_np).dtype if use_static else ml_dtypes.bfloat16
    wi_0 = jax.device_put(
        np.asarray(wi_0_np, dtype=expert_w_np),
        NamedSharding(epmoe.moe_mesh, P("expert", "tensor", None)),
    )
    wi_1 = jax.device_put(
        np.asarray(wi_1_np, dtype=expert_w_np),
        NamedSharding(epmoe.moe_mesh, P("expert", "tensor", None)),
    )
    wo = jax.device_put(
        np.asarray(wo_np, dtype=expert_w_np),
        NamedSharding(epmoe.moe_mesh, P("expert", None, "tensor")),
    )

    # Fused kernel expects transposed layout.
    fused_w_sharding = NamedSharding(mesh, P(("data", "tensor"), None, None))
    w1 = jax.device_put(
        np.asarray(wi_0_np, dtype=expert_w_np).transpose(0, 2, 1),
        fused_w_sharding,
    )
    w3 = jax.device_put(
        np.asarray(wi_1_np, dtype=expert_w_np).transpose(0, 2, 1),
        fused_w_sharding,
    )
    w2 = jax.device_put(
        np.asarray(wo_np, dtype=expert_w_np).transpose(0, 2, 1),
        fused_w_sharding,
    )

    print("load fused weights")
    # Optional shared experts for BailingMoE: HF stores (out, in), transpose to match JAX shapes.
    # - Fused kernel expects these replicated (P()).
    # - EPMoE end-to-end computes shared_experts via MLP (BF16 or static FP8 QuantizedLinear).
    w1_shared_fused = w2_shared_fused = w3_shared_fused = None
    w1_shared_scale = w2_shared_scale = w3_shared_scale = None
    w1_shared_ep = w2_shared_ep = w3_shared_ep = None
    w1_shared_q = w2_shared_q = w3_shared_q = None
    w1_shared_scale_vec = w2_shared_scale_vec = w3_shared_scale_vec = None
    shared_out = None
    if has_shared:
        shared_w_np = np.asarray(w1_shared_np).dtype
        # Some safetensors readers represent FP8 tensors as uint8; interpret them using the
        # same FP8 dtype as the MoE expert weights when we are in static-checkpoint mode.
        if (
            use_static
            and shared_w_np == np.uint8
            and moe_weight_dtype
            in (
                jnp.float8_e4m3fn,
                jnp.float8_e5m2,
            )
        ):
            shared_w_np = (
                ml_dtypes.float8_e4m3fn
                if moe_weight_dtype == jnp.float8_e4m3fn
                else ml_dtypes.float8_e5m2
            )
        if shared_w_np not in (
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.float8_e5m2,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ):
            shared_w_np = ml_dtypes.bfloat16

        w1_shared_fused = jax.device_put(
            np.asarray(w1_shared_np, dtype=shared_w_np).T,
            NamedSharding(mesh, P()),
        )  # (hidden, se_inter)
        w3_shared_fused = jax.device_put(
            np.asarray(w3_shared_np, dtype=shared_w_np).T,
            NamedSharding(mesh, P()),
        )  # (hidden, se_inter)
        w2_shared_fused = jax.device_put(
            np.asarray(w2_shared_np, dtype=shared_w_np).T,
            NamedSharding(mesh, P()),
        )  # (se_inter, hidden)

        w1_shared_ep = jax.device_put(
            np.asarray(w1_shared_np, dtype=ml_dtypes.bfloat16).T,
            NamedSharding(mesh, P(None, "tensor")),
        )
        w3_shared_ep = jax.device_put(
            np.asarray(w3_shared_np, dtype=ml_dtypes.bfloat16).T,
            NamedSharding(mesh, P(None, "tensor")),
        )
        w2_shared_ep = jax.device_put(
            np.asarray(w2_shared_np, dtype=ml_dtypes.bfloat16).T,
            NamedSharding(mesh, P("tensor", None)),
        )

        if (
            w1_shared_scale_np is not None
            and w2_shared_scale_np is not None
            and w3_shared_scale_np is not None
        ):
            # Fused kernel expects scales reshaped to (1, 1, out_dim).
            w1_shared_scale = jax.device_put(
                np.asarray(w1_shared_scale_np, dtype=np.float32).reshape(1, 1, -1),
                NamedSharding(mesh, P()),
            )
            w3_shared_scale = jax.device_put(
                np.asarray(w3_shared_scale_np, dtype=np.float32).reshape(1, 1, -1),
                NamedSharding(mesh, P()),
            )
            w2_shared_scale = jax.device_put(
                np.asarray(w2_shared_scale_np, dtype=np.float32).reshape(1, 1, -1),
                NamedSharding(mesh, P()),
            )

            # End-to-end static quantization for shared_experts uses QuantizedLinear with 1D scales.
            # Only enable this path when weights are FP8 and we have corresponding per-output scales.
            if use_static and shared_w_np in (
                ml_dtypes.float8_e4m3fn,
                ml_dtypes.float8_e5m2,
                jnp.float8_e4m3fn,
                jnp.float8_e5m2,
            ):
                w1_shared_q = jax.device_put(
                    np.asarray(w1_shared_np, dtype=shared_w_np),
                    NamedSharding(mesh, P("tensor", None)),
                )  # (se_inter, hidden)
                w3_shared_q = jax.device_put(
                    np.asarray(w3_shared_np, dtype=shared_w_np),
                    NamedSharding(mesh, P("tensor", None)),
                )  # (se_inter, hidden)
                w2_shared_q = jax.device_put(
                    np.asarray(w2_shared_np, dtype=shared_w_np),
                    NamedSharding(mesh, P(None, "tensor")),
                )  # (hidden, se_inter)
                w1_shared_scale_vec = jax.device_put(
                    np.asarray(w1_shared_scale_np, dtype=np.float32).reshape(-1),
                    NamedSharding(mesh, P("tensor")),
                )
                w3_shared_scale_vec = jax.device_put(
                    np.asarray(w3_shared_scale_np, dtype=np.float32).reshape(-1),
                    NamedSharding(mesh, P("tensor")),
                )
                w2_shared_scale_vec = jax.device_put(
                    np.asarray(w2_shared_scale_np, dtype=np.float32).reshape(-1),
                    NamedSharding(mesh, P(None)),
                )
        elif np.asarray(w1_shared_np).dtype in (
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.float8_e5m2,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ):
            print(
                "[shared] warning: shared_experts weights appear to be FP8 but `*.weight_scale` "
                "keys were not found; fused shared path may be incorrect."
            )

    print("load epmoe params")
    # Replace randomly initialized params with checkpoint slice.
    with jax.sharding.use_abstract_mesh(epmoe.updated_mesh):
        epmoe.wi_0 = nnx.Param(wi_0, out_sharding=P("expert", "tensor", None))
        epmoe.wi_1 = nnx.Param(wi_1, out_sharding=P("expert", "tensor", None))
        epmoe.wo = nnx.Param(wo, out_sharding=P("expert", None, "tensor"))

    print("load epmoe scales")
    if use_static:
        wi_0_scale_np, wi_1_scale_np, wo_scale_np = static_scales
        subc_quant_wsz = 256

        # Prepare epmoe scale params (E, 1, 1, out_dim). Use `quantize_weights(is_static=True)`
        # to create nnx.Params with the correct pytree status, then overwrite their values.
        wi_0_scale = jax.device_put(
            wi_0_scale_np.astype(np.float32).reshape(args.num_experts, 1, 1, -1),
            NamedSharding(epmoe.moe_mesh, P("expert", None, None, "tensor")),
        )
        wi_1_scale = jax.device_put(
            wi_1_scale_np.astype(np.float32).reshape(args.num_experts, 1, 1, -1),
            NamedSharding(epmoe.moe_mesh, P("expert", None, None, "tensor")),
        )
        wo_scale = jax.device_put(
            wo_scale_np.astype(np.float32).reshape(args.num_experts, 1, 1, -1),
            NamedSharding(epmoe.moe_mesh, P("expert", None, None, None)),
        )
        with jax.set_mesh(epmoe.moe_mesh):
            epmoe.quantize_weights(is_static=True)
        epmoe.wi_0_scale.value = wi_0_scale
        epmoe.wi_1_scale.value = wi_1_scale
        epmoe.wo_scale.value = wo_scale

        # Prepare fused scales (E, hidden//256, 1, inter) etc.
        fused_scale_sharding = NamedSharding(mesh, P(("data", "tensor"), None, None, None))
        w1_scale_np = _repeat_scales_for_fused(
            wi_0_scale_np, num_blocks=hidden_size // subc_quant_wsz
        )
        w3_scale_np = _repeat_scales_for_fused(
            wi_1_scale_np, num_blocks=hidden_size // subc_quant_wsz
        )
        w2_scale_np = _repeat_scales_for_fused(
            wo_scale_np, num_blocks=intermediate_size // subc_quant_wsz
        )
        w1_scale = jax.device_put(w1_scale_np, fused_scale_sharding)
        w3_scale = jax.device_put(w3_scale_np, fused_scale_sharding)
        w2_scale = jax.device_put(w2_scale_np, fused_scale_sharding)

        print(
            f"[static] checkpoint=yes weight_dtype={np.asarray(wi_0_np).dtype} subc_quant_wsz={subc_quant_wsz}"
        )
    else:
        if static_requested:
            print("[static] checkpoint=no (missing `*.weight_scale`); falling back to online mode")

        if moe_weight_dtype is not None:
            with jax.set_mesh(mesh):
                epmoe.quantize_weights(is_static=False)
            w1, w2, w3, w1_scale, w2_scale, w3_scale = _quantize_fused_weights(
                moe_weight_dtype,
                w1,
                w2,
                w3,
                block_size=subc_quant_wsz,
            )
        else:
            w1_scale = w2_scale = w3_scale = None

    call_w1_shared = w1_shared_fused
    call_w2_shared = w2_shared_fused
    call_w3_shared = w3_shared_fused
    call_w1_shared_scale = w1_shared_scale
    call_w2_shared_scale = w2_shared_scale
    call_w3_shared_scale = w3_shared_scale

    if args.debug_use_ref_shared:
        print("[debug] 🛡️ Shielding Fused Shared Experts (passing None to kernel)...")
        call_w1_shared = None
        call_w2_shared = None
        call_w3_shared = None
        call_w1_shared_scale = None
        call_w2_shared_scale = None
        call_w3_shared_scale = None

    print("run fused_ep_moe")
    fused_out = fused_ep_moe(
        mesh=mesh,
        tokens=tokens,
        w1=w1,
        w2=w2,
        w3=w3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=args.top_k,
        act_fn=args.act_fn,
        subc_quant_wsz=subc_quant_wsz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w3_scale=w3_scale,
        w1_shared=call_w1_shared,
        w2_shared=call_w2_shared,
        w3_shared=call_w3_shared,
        w1_shared_scale=call_w1_shared_scale,
        w2_shared_scale=call_w2_shared_scale,
        w3_shared_scale=call_w3_shared_scale,
        tp_axis_name="tensor",
    )
    fused_out_no_shared = None
    if has_shared:
        # Keep the same call signature (shared args are not None) to avoid extra compilation;
        # pass zeros to isolate the shared-expert contribution inside fused_ep_moe.
        z_w1_shared = jnp.zeros_like(w1_shared_fused)
        z_w2_shared = jnp.zeros_like(w2_shared_fused)
        z_w3_shared = jnp.zeros_like(w3_shared_fused)
        fused_out_no_shared = fused_ep_moe(
            mesh=mesh,
            tokens=tokens,
            w1=w1,
            w2=w2,
            w3=w3,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=args.top_k,
            act_fn=args.act_fn,
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=z_w1_shared,
            w2_shared=z_w2_shared,
            w3_shared=z_w3_shared,
            w1_shared_scale=None,
            w2_shared_scale=None,
            w3_shared_scale=None,
            tp_axis_name="tensor",
        )
    # EPMoE uses `jax.sharding.reshard` under an ("expert","tensor") abstract mesh and
    # needs a concrete mesh context for device assignment.
    print("run epmoe")
    with jax.set_mesh(epmoe.moe_mesh):
        ep_out_expert = epmoe(tokens, topk_weights, topk_ids)
        ep_out = ep_out_expert

    if has_shared:
        with jax.set_mesh(mesh):
            if (
                use_static
                and w1_shared_q is not None
                and w2_shared_q is not None
                and w3_shared_q is not None
                and w1_shared_scale_vec is not None
                and w2_shared_scale_vec is not None
                and w3_shared_scale_vec is not None
            ):
                shared_out = _shared_mlp_fp8_static(
                    tokens,
                    mesh=mesh,
                    w1_q=w1_shared_q,
                    w2_q=w2_shared_q,
                    w3_q=w3_shared_q,
                    w1_scale=w1_shared_scale_vec,
                    w2_scale=w2_shared_scale_vec,
                    w3_scale=w3_shared_scale_vec,
                    act_fn=args.act_fn,
                    compute_dtype=jnp.float32 if args.shared_qmm_fp32_accum else None,
                )
            else:
                shared_out = _shared_mlp_bf16(
                    tokens,
                    mesh=mesh,
                    w1=w1_shared_ep,
                    w2=w2_shared_ep,
                    w3=w3_shared_ep,
                    act_fn=args.act_fn,
                )
        ep_out = ep_out + shared_out

    if args.debug_use_ref_shared and shared_out is not None:
        print("[debug] ➕ Adding Ref-MLP Shared Output to Fused-MoE output manually.")
        fused_out = (fused_out.astype(jnp.float32) + shared_out.astype(jnp.float32)).astype(
            jnp.bfloat16
        )
    # fused_out = jax.sharding.reshard(fused_out, NamedSharding(mesh, P(None)))
    # if fused_out_no_shared is not None:
    #     fused_out_no_shared = jax.sharding.reshard(
    #         fused_out_no_shared, NamedSharding(mesh, P(None))
    #     )
    print("reshape fused out")
    fused_np = np.asarray(jax.device_get(fused_out))
    fused_no_shared_np = (
        None if fused_out_no_shared is None else np.asarray(jax.device_get(fused_out_no_shared))
    )
    # ep_out = jax.sharding.reshard(ep_out, NamedSharding(mesh, P(None)))
    ep_np = np.asarray(jax.device_get(ep_out))
    ep_expert_np = np.asarray(jax.device_get(ep_out_expert))

    stats = _describe_diff(fused_np, ep_np)
    print(
        "[diff] " + " ".join(f"{k}={v:.6g}" for k, v in stats.items()) + f" shape={fused_np.shape}"
    )
    if has_shared and fused_no_shared_np is not None and shared_out is not None:
        expert_stats = _describe_diff(fused_no_shared_np, ep_expert_np)
        fused_shared_np = fused_np.astype(np.float32) - fused_no_shared_np.astype(np.float32)
        # shared_out = jax.sharding.reshard(shared_out, NamedSharding(mesh, P(None)))
        shared_np = np.asarray(jax.device_get(shared_out)).astype(np.float32)
        shared_stats = _describe_diff(fused_shared_np, shared_np)
        print(
            "[diff.expert] "
            + " ".join(f"{k}={v:.6g}" for k, v in expert_stats.items())
            + f" shape={fused_no_shared_np.shape}"
        )
        print(
            "[diff.shared] "
            + " ".join(f"{k}={v:.6g}" for k, v in shared_stats.items())
            + f" shape={shared_np.shape}"
        )

    if args.dump_shared_intermediates:
        if not has_shared or fused_no_shared_np is None or shared_out is None:
            print("[dump.shared] skipped (shared_experts not enabled in this run)")
        else:
            with jax.set_mesh(mesh):
                if (
                    use_static
                    and w1_shared_q is not None
                    and w2_shared_q is not None
                    and w3_shared_q is not None
                    and w1_shared_scale_vec is not None
                    and w2_shared_scale_vec is not None
                    and w3_shared_scale_vec is not None
                ):
                    ep_shared_int = _shared_mlp_fp8_static_intermediates(
                        tokens,
                        mesh=mesh,
                        w1_q=w1_shared_q,
                        w2_q=w2_shared_q,
                        w3_q=w3_shared_q,
                        w1_scale=w1_shared_scale_vec,
                        w2_scale=w2_shared_scale_vec,
                        w3_scale=w3_shared_scale_vec,
                        act_fn=args.act_fn,
                        compute_dtype=jnp.float32 if args.shared_qmm_fp32_accum else None,
                    )
                else:
                    ep_shared_int = _shared_mlp_bf16_intermediates(
                        tokens,
                        mesh=mesh,
                        w1=w1_shared_ep,
                        w2=w2_shared_ep,
                        w3=w3_shared_ep,
                        act_fn=args.act_fn,
                    )

                ref_shared_int = _shared_mlp_ref_fp32_intermediates(
                    tokens,
                    mesh=mesh,
                    w1=w1_shared_fused,
                    w2=w2_shared_fused,
                    w3=w3_shared_fused,
                    w1_scale=w1_shared_scale,
                    w2_scale=w2_shared_scale,
                    w3_scale=w3_shared_scale,
                    act_fn=args.act_fn,
                )

            fused_shared_out = fused_np.astype(np.float32) - fused_no_shared_np.astype(np.float32)
            ep_a1 = np.asarray(jax.device_get(ep_shared_int["a1"])).astype(np.float32)
            ep_a2 = np.asarray(jax.device_get(ep_shared_int["a2"])).astype(np.float32)
            ep_inter = np.asarray(jax.device_get(ep_shared_int["inter"])).astype(np.float32)
            ep_out_shared = np.asarray(jax.device_get(ep_shared_int["out"])).astype(np.float32)

            ref_a1 = np.asarray(jax.device_get(ref_shared_int["a1"])).astype(np.float32)
            ref_a2 = np.asarray(jax.device_get(ref_shared_int["a2"])).astype(np.float32)
            ref_inter = np.asarray(jax.device_get(ref_shared_int["inter"])).astype(np.float32)
            ref_out = np.asarray(jax.device_get(ref_shared_int["out"])).astype(np.float32)

            # NOTE: `ref_shared_*` are fp32 by construction, but both ep_shared and fused outputs
            # are ultimately bf16. Casting the ref to bf16 helps attribute residual diffs to bf16
            # rounding vs true semantic/kernel mismatches.
            ref_a1_bf16 = ref_a1.astype(ml_dtypes.bfloat16).astype(np.float32)
            ref_a2_bf16 = ref_a2.astype(ml_dtypes.bfloat16).astype(np.float32)
            ref_inter_bf16 = ref_inter.astype(ml_dtypes.bfloat16).astype(np.float32)
            ref_out_bf16 = ref_out.astype(ml_dtypes.bfloat16).astype(np.float32)

            print(
                "[shared.stage] ref_vs_ep a1 "
                + " ".join(f"{k}={v:.6g}" for k, v in _describe_diff(ref_a1, ep_a1).items())
            )
            print(
                "[shared.stage] ref_vs_ep a2 "
                + " ".join(f"{k}={v:.6g}" for k, v in _describe_diff(ref_a2, ep_a2).items())
            )
            print(
                "[shared.stage] ref_vs_ep inter "
                + " ".join(f"{k}={v:.6g}" for k, v in _describe_diff(ref_inter, ep_inter).items())
            )
            print(
                "[shared.stage] ref_vs_ep out "
                + " ".join(
                    f"{k}={v:.6g}" for k, v in _describe_diff(ref_out, ep_out_shared).items()
                )
            )
            print(
                "[shared.stage] fused_vs_ref out "
                + " ".join(
                    f"{k}={v:.6g}" for k, v in _describe_diff(fused_shared_out, ref_out).items()
                )
            )

            print(
                "[shared.stage] refbf16_vs_ep a1 "
                + " ".join(f"{k}={v:.6g}" for k, v in _describe_diff(ref_a1_bf16, ep_a1).items())
            )
            print(
                "[shared.stage] refbf16_vs_ep a2 "
                + " ".join(f"{k}={v:.6g}" for k, v in _describe_diff(ref_a2_bf16, ep_a2).items())
            )
            print(
                "[shared.stage] refbf16_vs_ep inter "
                + " ".join(
                    f"{k}={v:.6g}" for k, v in _describe_diff(ref_inter_bf16, ep_inter).items()
                )
            )
            print(
                "[shared.stage] refbf16_vs_ep out "
                + " ".join(
                    f"{k}={v:.6g}" for k, v in _describe_diff(ref_out_bf16, ep_out_shared).items()
                )
            )
            print(
                "[shared.stage] fused_vs_refbf16 out "
                + " ".join(
                    f"{k}={v:.6g}"
                    for k, v in _describe_diff(fused_shared_out, ref_out_bf16).items()
                )
            )

            os.makedirs(os.path.dirname(args.dump_shared_intermediates) or ".", exist_ok=True)
            np.savez(
                args.dump_shared_intermediates,
                fused_shared_out=fused_shared_out,
                ep_shared_a1=ep_a1,
                ep_shared_a2=ep_a2,
                ep_shared_inter=ep_inter,
                ep_shared_out=ep_out_shared,
                ref_shared_a1=ref_a1,
                ref_shared_a2=ref_a2,
                ref_shared_inter=ref_inter,
                ref_shared_out=ref_out,
                ref_shared_out_bf16=ref_out_bf16,
                meta=np.asarray(
                    {
                        "model_dir": args.model_dir,
                        "prefix": prefix,
                        "layer_idx": args.layer_idx,
                        "act_fn": args.act_fn,
                        "use_static": bool(use_static),
                    },
                    dtype=object,
                ),
            )
            print(f"[dump.shared] wrote {args.dump_shared_intermediates}")

    if args.debug_topn and args.debug_topn > 0:
        abs_diff = np.abs(fused_np.astype(np.float32) - ep_np.astype(np.float32))
        denom = np.maximum(np.abs(ep_np.astype(np.float32)), float(args.debug_rel_eps))
        rel_diff = abs_diff / denom
        metric = abs_diff if args.debug_metric == "abs" else rel_diff

        flat = metric.reshape(-1)
        n = int(min(args.debug_topn, flat.size))
        top = np.argpartition(flat, -n)[-n:]
        top = top[np.argsort(flat[top])[::-1]]

        topk_ids_np = np.asarray(jax.device_get(topk_ids))
        topk_weights_np = np.asarray(jax.device_get(topk_weights))
        fused_no_shared_f32 = (
            None
            if fused_no_shared_np is None
            else fused_no_shared_np.astype(np.float32, copy=False)
        )

        print(f"[debug] topn={n} metric={args.debug_metric}")
        for rank, flat_idx in enumerate(top, start=1):
            t, h = np.unravel_index(int(flat_idx), metric.shape)
            fused_v = float(fused_np[t, h].astype(np.float32))
            ep_v = float(ep_np[t, h].astype(np.float32))
            abs_v = float(abs_diff[t, h])
            rel_v = float(rel_diff[t, h])

            msg = (
                f"[debug.{rank}] idx=({t},{h}) "
                f"fused={fused_v:.6g} ep={ep_v:.6g} abs={abs_v:.6g} rel={rel_v:.6g}"
            )
            if fused_no_shared_f32 is not None and has_shared and shared_out is not None:
                fused_expert_v = float(fused_no_shared_f32[t, h])
                ep_expert_v = float(ep_expert_np[t, h].astype(np.float32))
                fused_shared_v = float(fused_v - fused_expert_v)
                ep_shared_v = float(shared_np[t, h])
                msg += (
                    f" expert(abs={abs(fused_expert_v-ep_expert_v):.6g})"
                    f" shared(abs={abs(fused_shared_v-ep_shared_v):.6g})"
                )
            print(msg)
            print(f"  topk_ids={topk_ids_np[t].tolist()} topk_w={topk_weights_np[t].tolist()}")

    if args.dump_npz:
        os.makedirs(os.path.dirname(args.dump_npz) or ".", exist_ok=True)
        np.savez(
            args.dump_npz,
            tokens=np.asarray(jax.device_get(tokens)),
            router_logits=np.asarray(jax.device_get(router_logits)),
            topk_weights=np.asarray(jax.device_get(topk_weights)),
            topk_ids=np.asarray(jax.device_get(topk_ids)),
            fused_out=fused_np,
            ep_out=ep_np,
            wi_0=np.asarray(wi_0_np),
            wi_1=np.asarray(wi_1_np),
            wo=np.asarray(wo_np),
            shared_out=(None if shared_out is None else np.asarray(jax.device_get(shared_out))),
            meta=np.asarray(
                {
                    "prefix": hf_keys.prefix,
                    "moe_path": hf_keys.moe_path,
                    "source_expert_pattern": hf_keys.source_expert_pattern,
                    "gate_name": hf_keys.gate_name,
                    "up_name": hf_keys.up_name,
                    "down_name": hf_keys.down_name,
                    "num_experts": args.num_experts,
                    "top_k": args.top_k,
                    "num_tokens": args.num_tokens,
                    "act_fn": args.act_fn,
                    "moe_weight_dtype": args.moe_weight_dtype,
                    "renormalize_topk_logits": args.renormalize_topk_logits,
                    "routed_scaling_factor": args.routed_scaling_factor,
                    "has_shared_experts": has_shared,
                    "tokens_apply_post_attn_rmsnorm": args.tokens_apply_post_attn_rmsnorm,
                    "tokens_rmsnorm_from_ckpt": args.tokens_rmsnorm_from_ckpt,
                    "tokens_rmsnorm_eps": args.tokens_rmsnorm_eps,
                },
                dtype=object,
            ),
        )
        print(f"[dump] wrote {args.dump_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
