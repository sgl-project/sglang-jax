"""Unit tests for the fused_moe v2 autotuner candidate pruning (tune_prune).

Pins the fix for the historical blind spot where the compute-favorable
(large bf, large btc) corner was pruned out for prefill shapes, so the tuner
"never tested" btc=128 with a large bf and always shipped btc=64. Also guards
that decode/small shapes are NOT polluted with large btc (the seed is adaptive,
bounded by bts).

tune_prune has no jax/kernel dependency, so these run in milliseconds.
"""

from __future__ import annotations

import pathlib
import sys

_V2 = pathlib.Path(__file__).resolve().parents[1] / "srt" / "kernels" / "fused_moe" / "v2"
sys.path.insert(0, str(_V2))

from tune_prune import prune_candidates  # noqa: E402


class Cfg:
    """Minimal stand-in for FusedMoEBlockConfig (prune only reads these)."""

    def __init__(self, bt, bf, btc, bts, bse):
        self.bt, self.bf, self.btc, self.bts, self.bse = bt, bf, btc, bts, bse

    def tup(self):
        return (self.bt, self.bf, self.btc, self.bts, self.bse)

    def __repr__(self):
        return f"Cfg{self.tup()}"


# btc candidates = divisors of bts that are multiples of 8 (mirrors
# _aligned_divisors(bts, 8) in bench_v2).
def _btc_divs(bts):
    return [v for v in (128, 64, 32, 16, 8) if bts % v == 0 and v <= bts]


def _prefill_grid():
    """A GLM-5.2 ep16 16384-like candidate set: one big (bt=128,bts=128) bucket
    plus several other buckets, so total > max_configs and >1 bucket contend.
    (bf=2048, btc=128) is omitted to mimic its upstream VMEM-filter drop."""
    cfgs = []
    for bf in (2048, 1024, 512, 256, 128):
        for btc in _btc_divs(128):
            if bf == 2048 and btc == 128:
                continue  # VMEM-oversized, dropped before pruning
            cfgs.append(Cfg(128, bf, btc, 128, 1024))
    for bt, bts in ((256, 256), (128, 256), (64, 64), (32, 32)):
        for bf in (1024, 512, 256):
            for btc in _btc_divs(bts)[:3]:
                cfgs.append(Cfg(bt, bf, btc, bts, 512))
    return cfgs


def test_prefill_corner_survives_prune():
    """The compute-favorable (bt=128, bf=1024, btc=128, bts=128) corner must be
    selected — this is the exact config manual sweeps proved optimal and that the
    old anchor-32 Latin traversal buried below the survival depth."""
    cfgs = _prefill_grid()
    assert len(cfgs) > 48, "test setup must exceed the cap to exercise pruning"
    selected = prune_candidates(cfgs, 48)
    tups = {c.tup() for c in selected}
    assert (128, 1024, 128, 128, 1024) in tups, (
        f"large-bf x large-btc corner was pruned out: "
        f"{sorted(t for t in tups if t[0] == 128 and t[3] == 128)}"
    )
    # And more generally at least one btc=128 config survives.
    assert any(c.btc == 128 for c in selected)


def test_decode_small_btc_not_polluted():
    """Decode buckets (small bts) must keep small btc — the seed is bounded by
    bts (btc divides bts), so it never injects btc=128 into a bts=8 bucket."""
    cfgs = []
    # decode buckets
    for bf in (512, 1024):
        cfgs.append(Cfg(8, bf, 8, 8, 128))
        cfgs.append(Cfg(8, bf, 8, 8, 512))
    for bf in (512, 1024):
        for btc in (16, 8):
            cfgs.append(Cfg(16, bf, btc, 16, 512))
    # prefill padding to force pruning
    for bf in (2048, 1024, 512, 256, 128):
        for btc in _btc_divs(128):
            cfgs.append(Cfg(128, bf, btc, 128, 1024))
    max_configs = 12
    assert len(cfgs) > max_configs
    selected = prune_candidates(cfgs, max_configs)
    # seed is bts-bounded: no small-bts bucket ever gets btc > its bts
    for c in selected:
        assert c.btc <= c.bts, f"{c} violates btc<=bts (seed polluted a small bucket)"
    # the prefill corner still wins a slot even under the tighter cap
    assert any(c.tup() == (128, 1024, 128, 128, 1024) for c in selected)


def test_no_prune_when_under_cap():
    """Below the cap, all candidates are returned unchanged."""
    cfgs = [Cfg(128, 1024, btc, 128, 1024) for btc in (128, 64, 32)]
    selected = prune_candidates(cfgs, 48)
    assert len(selected) == len(cfgs)
    assert {c.tup() for c in selected} == {c.tup() for c in cfgs}


def test_selected_respects_cap():
    """Pruning never returns more than max_configs."""
    cfgs = _prefill_grid()
    selected = prune_candidates(cfgs, 48)
    assert len(selected) <= 48
