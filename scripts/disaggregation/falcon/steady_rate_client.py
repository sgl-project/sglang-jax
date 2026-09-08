"""Matched-arrival-rate latency checks; fail rather than silently throttle load."""

import asyncio
import math
import time

import aiohttp
from steady_client import percentile


async def run(send, rate, warmup_s, duration_s, *, max_inflight=64, on_complete=None):
    if rate <= 0 or warmup_s < 0 or duration_s <= 0 or max_inflight < 1:
        raise ValueError("invalid arrival schedule")
    origin = time.perf_counter()
    end = origin + warmup_s + duration_s
    records = []
    active = peak = 0
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
        connector=aiohttp.TCPConnector(limit=max_inflight),
        read_bufsize=2**20,
    ) as session:

        async def one(sequence, scheduled):
            nonlocal active, peak
            if active >= max_inflight:
                raise RuntimeError("client in-flight cap reached; offered rate was not sustained")
            active += 1
            peak = max(peak, active)
            begin = time.perf_counter()
            try:
                result = await send(session, sequence)
                result.update(
                    sequence=sequence,
                    start_s=begin - origin,
                    end_s=time.perf_counter() - origin,
                    scheduled_s=scheduled - origin,
                    lateness_s=max(0, begin - scheduled),
                    cohort=origin + warmup_s <= begin < end,
                )
                records.append(result)
                if on_complete:
                    on_complete(result)
            finally:
                active -= 1

        async with asyncio.TaskGroup() as group:
            for sequence in range(math.ceil((warmup_s + duration_s) * rate)):
                scheduled = origin + sequence / rate
                if scheduled >= end:
                    break
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                group.create_task(one(sequence, scheduled))
    cohort = [r for r in records if r["cohort"]]
    assert cohort, "empty measurement cohort"
    return {
        "mode": "fixed_arrival_rate_latency",
        "offered_rps": rate,
        "actual_cohort_rps": len(cohort) / duration_s,
        "warmup_s": warmup_s,
        "measurement_s": duration_s,
        "completed_total": len(records),
        "cohort_requests": len(cohort),
        "failed": 0,
        "max_inflight": max_inflight,
        "peak_client_active": peak,
        "arrival_lateness_p95_s": percentile([r["lateness_s"] for r in cohort], 0.95),
        "arrival_lateness_max_s": max(r["lateness_s"] for r in cohort),
        "ttft_p50_s": percentile([r["ttft_s"] for r in cohort], 0.5),
        "ttft_p95_s": percentile([r["ttft_s"] for r in cohort], 0.95),
        "e2e_p50_s": percentile([r["e2e_s"] for r in cohort], 0.5),
        "e2e_p95_s": percentile([r["e2e_s"] for r in cohort], 0.95),
        "drain_s": max(0, time.perf_counter() - end),
    }
