"""Real DeepSeek-V4.1-Flash configuration -> model_ref.Args, and the production parameter table (shapes/dtypes/shardings)."""

import json

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from model_ref import Args
from sharding import param_spec


def args_from_hf_config(path, max_seq_len=32768) -> Args:
    c = json.load(open(path))
    t = c.get("text_config", c)
    n = t["num_hidden_layers"]
    return Args(
        dim=t["hidden_size"],
        n_layers=n,
        n_heads=t["num_attention_heads"],
        head_dim=t["head_dim"],
        rope_head_dim=t["qk_rope_head_dim"],
        q_lora_rank=t["q_lora_rank"],
        o_groups=t["o_groups"],
        o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"],
        compress_ratios=tuple(t["compress_ratios"][:n]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]),
        index_source_layers=tuple(t["index_source_layer_ids"]),
        index_n_heads=t["index_n_heads"],
        index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"],
        candidate_source_layer=t["candidate_source_layer_id"],
        candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"],
        n_routed_experts=t["n_routed_experts"],
        n_activated_experts=t["num_experts_per_tok"],
        moe_inter_dim=t["moe_intermediate_size"],
        swiglu_limit=t["swiglu_limit"],
        score_func=t["scoring_func"],
        gate_temp=1.0,
        norm_topk_prob=t["norm_topk_prob"],
        route_scale=t["routed_scaling_factor"],
        norm_eps=t["rms_norm_eps"],
        hc_mult=t["hc_mult"],
        hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"],
        rope_theta=t["rope_theta"],
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=t["rope_scaling"]["original_max_position_embeddings"],
        rope_factor=t["rope_scaling"]["factor"],
        beta_fast=t["rope_scaling"]["beta_fast"],
        beta_slow=t["rope_scaling"]["beta_slow"],
        max_seq_len=max_seq_len,
        vocab_size=t["vocab_size"],
        engram_layer_ids=tuple(t["engram_layer_ids"]),
        engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"],
    )


def param_table(a: Args, act=jnp.bfloat16):
    """name -> (shape, dtype). Production layout: dense bf16 (dequantized at load), experts int8 + uint8 scales, f32 for mHC/gates."""
    D, H, hd, rd = a.dim, a.n_heads, a.head_dim, a.rope_head_dim
    T = {
        "embed.weight": ((a.vocab_size, D), act),
        "head.weight": ((a.vocab_size, D), act),
        "norm.weight": ((D,), jnp.float32),
    }
    mix = (2 + a.hc_mult) * a.hc_mult
    for l in range(a.n_layers):
        p = f"layers.{l}."
        r = a.compress_ratios[l]
        for n in ("attn_norm", "ffn_norm"):
            T[p + n + ".weight"] = ((D,), jnp.float32)
        for k in ("attn", "ffn"):
            T[p + f"hc_{k}_fn"] = ((mix, a.hc_mult * D), jnp.float32)
            T[p + f"hc_{k}_base"] = ((mix,), jnp.float32)
            T[p + f"hc_{k}_scale"] = ((3,), jnp.float32)
        q = p + "attn."
        T[q + "wq_a.weight"] = ((a.q_lora_rank, D), act)
        T[q + "q_norm.weight"] = ((a.q_lora_rank,), jnp.float32)
        T[q + "wq_b.weight"] = ((H * hd, a.q_lora_rank), act)
        T[q + "wkv.weight"] = ((hd, D), act)
        T[q + "kv_norm.weight"] = ((hd,), jnp.float32)
        T[q + "attn_sink"] = ((H,), jnp.float32)
        T[q + "wo_a.weight"] = ((a.o_groups * a.o_lora_rank, H * hd // a.o_groups), act)
        T[q + "wo_b.weight"] = ((D, a.o_groups * a.o_lora_rank), act)
        if r and l in a.kv_source_layers:
            T[q + "compressor.wkv.weight"] = ((hd, D), jnp.float32 if r > 1 else act)
            if r > 1:
                T[q + "compressor.wgate.weight"] = ((hd, D), jnp.float32)
            T[q + "compressor.norm.weight"] = ((hd,), jnp.float32)
        if r and l in a.index_source_layers:
            T[q + "indexer.wq_b.weight"] = (
                (a.index_n_heads * a.index_head_dim, a.q_lora_rank),
                act,
            )
            T[q + "indexer.weights_proj.weight"] = ((a.index_n_heads, D), act)
            if l in a.kv_source_layers:
                T[q + "indexer.wk.weight"] = ((a.index_head_dim, hd), act)
                T[q + "indexer.k_norm.weight"] = ((a.index_head_dim,), jnp.float32)
        f, I, E = p + "ffn.", a.moe_inter_dim, a.n_routed_experts
        T[f + "gate.weight"] = ((E, D), jnp.float32)
        T[f + "gate.bias"] = ((E,), jnp.float32)
        T[f + "shared_experts.w1.weight"] = ((I, D), act)
        T[f + "shared_experts.w3.weight"] = ((I, D), act)
        T[f + "shared_experts.w2.weight"] = ((D, I), act)
        f8 = jnp.float8_e4m3fn
        T[f + "experts.w1"] = ((E, I, D), f8)
        T[f + "experts.w1_rowscale"] = ((E, I), jnp.float32)
        T[f + "experts.w3"] = ((E, I, D), f8)
        T[f + "experts.w3_rowscale"] = ((E, I), jnp.float32)
        T[f + "experts.w2"] = ((E, D, I), f8)
        T[f + "experts.w2_rowscale"] = ((E, D), jnp.float32)
        if l in a.engram_layer_ids:
            ncols = (a.engram_max_ngram_size - 1) * a.engram_n_heads
            T[p + "engram.wkv.weight"] = ((D * (a.hc_mult + 1), ncols * a.engram_head_dim), act)
            T[p + "engram.q_weight"] = ((a.hc_mult, D), jnp.float32)
            T[p + "engram.k_weight"] = ((a.hc_mult, D), jnp.float32)
    return T


def table_bytes(T, per_device_fn=None):
    return sum(int(np.prod(s)) * np.dtype(d).itemsize for s, d in T.values())
