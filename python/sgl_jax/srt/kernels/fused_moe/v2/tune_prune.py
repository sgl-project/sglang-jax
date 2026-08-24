"""Candidate pruning for the fused_moe v2 autotuner (offline, bench-time only).

Split out of bench_v2.py so the pruning heuristic is unit-testable in isolation:
bench_v2.py runs a full kernel sweep at import time, so importing it from a test
is not possible. This module has no jax/kernel dependency -- it operates purely
on candidate objects exposing .bt/.bf/.btc/.bts/.bse (FusedMoEBlockConfig).
"""

from __future__ import annotations

import math


def prune_candidates(configs, max_configs, log=None):
    """Cut a VMEM-valid candidate list down to max_configs for benchmarking.

    Buckets by (bt, bts). Within each bucket it first SEEDS the compute-favorable
    (large bf, largest btc) corner, then fills the rest with a diversity-preserving
    Latin traversal over (bf, btc); finally it round-robins across buckets until
    max_configs are chosen.

    The seed is what fixes the historical blind spot: without it the (large bf,
    large btc) point -- the MXU-saturating optimum for compute-bound prefill --
    sank to a within-bucket depth the round-robin never reached, so the tuner
    "never tested" btc=128 with a large bf and always shipped btc=64.
    """
    if log is None:

        def log(*_a, **_k):
            return None

    if len(configs) <= max_configs:
        log(f"  tune: {len(configs)} configs (all pass VMEM filter)")
        return configs

    buckets = {}
    for cfg in configs:
        bk = (cfg.bt, cfg.bts or cfg.bt)
        buckets.setdefault(bk, []).append(cfg)
    for bk in buckets:
        # Preserve BF and BTC diversity under the max-config cap. Sorting the
        # whole bucket by BF first can spend every slot on bf=1024; naively
        # round-robining BF can then spend every slot on the same btc. Build a
        # small Latin-style traversal over (BF, BTC), rotating the starting BTC
        # for each BF -- but seed the compute-favorable corner first (below).
        by_pair = {}
        for cfg in buckets[bk]:
            by_pair.setdefault((cfg.bf, cfg.btc), []).append(cfg)
        for pair in by_pair:
            by_pair[pair].sort(key=lambda c: c.bse, reverse=True)
        ordered = []
        bf_keys = sorted({cfg.bf for cfg in buckets[bk]}, reverse=True)
        btc_keys_all = {cfg.btc for cfg in buckets[bk]}
        # Corner seeding: put the compute-favorable (large bf, largest btc) point
        # at the front so the diversity round-robin cannot evict it under a tight
        # max_configs. The MXU 128x128 systolic array wants a large compute-tile
        # M (= btc); the prefill optimum is (large bf, large btc), which the Latin
        # traversal alone buries deep (e.g. bf1024xbtc128 at within-bucket index
        # ~11 while only ~2 survive per bucket at 21 buckets / 48 slots). Seed the
        # top-2 bf (the optimum is often the 2nd-largest bf, not the max) paired
        # with the largest available btc. btc_max is bounded by bts (btc divides
        # bts), so decode buckets (small bts) only seed small btc -- adaptive, no
        # small-shape regression. Missing (bf, btc_max) pairs (e.g. bf=2048xbtc=128
        # dropped by the VMEM filter upstream) are simply skipped.
        btc_max = max(btc_keys_all)
        for bf in bf_keys[:2]:
            queue = by_pair.get((bf, btc_max), [])
            if queue:
                ordered.append(queue.pop(0))
        # Tail btc order anchored at the MXU-saturation point for this bucket's
        # token-block size (bt = bk[0]), not a fixed 32: prefill (bt>=128) then
        # also prefers large btc in the tail, while decode (small bt) collapses
        # the anchor to bt and keeps its small-btc-first order.
        anchor = max(8, min(bk[0], 128))
        btc_keys = sorted(
            btc_keys_all,
            key=lambda btc: (abs(math.log2(btc / anchor)), -btc),
        )
        round_idx = 0
        while any(by_pair[pair] for pair in by_pair):
            for bf_idx, bf in enumerate(bf_keys):
                for offset in range(len(btc_keys)):
                    btc = btc_keys[(round_idx + bf_idx + offset) % len(btc_keys)]
                    queue = by_pair.get((bf, btc), [])
                    if queue:
                        ordered.append(queue.pop(0))
                        break
            round_idx += 1
        buckets[bk] = ordered

    selected = []
    selected_keys = set()
    bucket_keys = sorted(buckets.keys(), reverse=True)
    while len(selected) < max_configs:
        made_progress = False
        for bk in bucket_keys:
            bucket = buckets[bk]
            if not bucket:
                continue
            cfg = bucket.pop(0)
            key = (cfg.bt, cfg.bf, cfg.btc, cfg.bts, cfg.bse)
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(cfg)
                made_progress = True
            if len(selected) >= max_configs:
                break
        if not made_progress:
            break

    log(
        f"  tune: {len(configs)} valid -> {len(selected)} selected "
        f"(max={max_configs}, {len(bucket_keys)} bt/bts buckets)"
    )
    return selected
