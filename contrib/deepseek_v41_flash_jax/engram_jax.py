"""Engram for DeepSeek-V4.1 in NumPy (host side: hashing + row lookup) and JAX (gate).

Port of the reference `inference/engram.py` + `Engram` in `model.py`. The hash ids depend only on token ids, so
they are computed on the host before the step; the table rows (fp8 e4m3 + e8m0 block scales) are gathered on the
host (from an mmapped file in production) and handed to the compiled step as a dense [B, L, n_cols, head_dim] array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(math.isqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    cand = start + 1
    while not _is_prime(cand) or cand in seen:
        cand += 1
    return cand


def build_compressed_token_map(tokenizer):
    """Same as the reference: token id -> compressed id (tokens that normalize alike collapse)."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        nid = key_to_new.get(key)
        if nid is None:
            nid = len(key_to_new)
            key_to_new[key] = nid
        lookup[tid] = nid
    return np.asarray(lookup, np.int64), len(key_to_new)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple  # [layer][ngram-1][head]
    n_heads: int
    head_dim: int

    @staticmethod
    def make(layer_ids, num_embeddings, max_ngram_size, n_heads, head_dim, engram_vocab_size):
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, cur = [], engram_vocab_size - 1
                for _ in range(n_heads):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return EngramLayout(
            max_ngram_size,
            tuple(layer_ids),
            tuple(num_embeddings),
            tuple(primes),
            n_heads,
            head_dim,
        )

    @property
    def n_cols(self):
        return (self.max_ngram_size - 1) * self.n_heads


def hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size):
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        v = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)


class NgramHasher:
    """Host-side hash ids for the n-grams ending at each position (same algorithm and state as the reference)."""

    DEAD = -1

    def __init__(
        self,
        layout: EngramLayout,
        token_map: np.ndarray,
        compressed_vocab_size: int,
        pad_id: int,
        max_batch: int,
        max_seq: int,
    ):
        self.layout = layout
        self.token_map = token_map
        self.pad = int(token_map[pad_id])
        flat = [[p for per in layer for p in per] for layer in layout.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], np.int64
        )  # [L, n_cols]
        self.primes = np.array(layout.primes, np.int64)  # [L, ngram-1, heads]
        self.mult = hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, compressed_vocab_size
        )  # [L, ngram]
        self.cache = np.zeros((max_batch, max_seq), np.int64)

    def __call__(self, input_ids: np.ndarray, start_pos, active=None) -> np.ndarray:
        """input_ids [B, L] -> hash ids [B, L, n_layers, n_cols]. start_pos: int or per-row [B]; active [B] bool: only active rows
        record their tokens in the history cache (inactive rows get valid-but-unused ids)."""
        b, s = input_ids.shape
        comp = self.token_map[input_ids]
        sp = np.broadcast_to(np.asarray(start_pos, np.int64), (b,))
        act = np.ones(b, bool) if active is None else np.asarray(active, bool)
        pos = np.clip(sp[:, None] + np.arange(s)[None, :], 0, self.cache.shape[1] - 1)  # [B, L]
        rows = np.nonzero(act)[0]
        if len(rows):
            self.cache[rows[:, None], pos[rows]] = comp[rows]
        blocked = np.zeros((b, s), bool)
        toks = []
        for shift in range(self.layout.max_ngram_size):
            src = np.take_along_axis(self.cache[:b], np.clip(pos - shift, 0, None), axis=1)
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)  # [B, L, ngram]
        prod = (
            toks[:, :, None, :] * self.mult
        )  # [B, L, nl, ngram]   (no int64 overflow by construction)
        rolling, hashes = prod[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            hashes.append((rolling[..., None] % self.primes[:, i - 1]))
        return np.concatenate(hashes, axis=-1) + self.offsets


def gather_rows(weight, scale, ids, block=32):
    """Dequantized table rows. weight [R, D] fp8 e4m3, scale [R, D/block] (e8m0 or float), ids [...] -> [..., D] float32."""
    w = np.asarray(weight)[ids].astype(np.float32)
    s = np.asarray(scale)[ids].astype(np.float32)
    w = w.reshape(*w.shape[:-1], -1, block) * s[..., None]
    return w.reshape(*ids.shape, -1)


def dequant_rows(w8, s8, block=32):
    """Device-side dequantization of raw Engram table bytes: w8 [..., D] uint8 (fp8 e4m3fn bit patterns), s8 [..., D/block] uint8 (e8m0
    exponents) -> float32 [..., D], bit-identical to gather_rows (fp8 value times a power of two is exact).
    """
    w = jax.lax.bitcast_convert_type(w8, jnp.float8_e4m3fn).astype(jnp.float32)
    e = s8.astype(jnp.uint32)
    s = jnp.where(
        e == 0, jnp.float32(2.0**-127), jax.lax.bitcast_convert_type(e << 23, jnp.float32)
    )  # 2**(e-127)
    return (w.reshape(*w.shape[:-1], -1, block) * s[..., None]).reshape(w8.shape)


def engram_gate(x, rows, P, hc_mult, dim, eps, clamp=1e-6):
    """x [B, L, hc, dim]; rows [B, L, n_cols, head_dim] (dequantized). P keys: wkv.weight, q_weight, k_weight."""
    kv = rows.reshape(*rows.shape[:2], -1).astype(jnp.float32) @ P["wkv.weight"].T.astype(
        jnp.float32
    )
    key, value = kv[..., : hc_mult * dim], kv[..., hc_mult * dim :]
    key = key.reshape(*key.shape[:-1], hc_mult, dim)
    weight = P["q_weight"].astype(jnp.float32) * P["k_weight"].astype(jnp.float32)
    h = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1) + eps) * jax.lax.rsqrt(
        jnp.mean(jnp.square(key), axis=-1) + eps
    )
    dot = jnp.sum(h * weight * key, axis=-1) * rstd * dim**-0.5
    gate = jax.nn.sigmoid(jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp)), dot))
    return (h + gate[..., None] * value[:, :, None, :]).astype(x.dtype)
