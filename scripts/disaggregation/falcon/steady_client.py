"""Closed-loop SSE load with explicit warmup, measurement, and drain boundaries.

One persistent connection pool and independent workers replenish requests without
batch barriers. Throughput counts observed token deltas in [start, end), including
requests crossing either boundary. Latency uses the start-in-window cohort,
including its drain, so slow requests are not silently censored.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class Window:
    start: float
    end: float
    tokens: int = 0
    bins: dict = field(default_factory=dict)

    def observe(self, at, delta):
        if delta < 0:
            raise AssertionError("completion token count decreased")
        if self.start <= at < self.end:
            self.tokens += delta
            second = int(at - self.start)
            self.bins[second] = self.bins.get(second, 0) + delta


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


async def run(
    url,
    payloads,
    concurrency,
    warmup_s,
    duration_s,
    *,
    on_complete=None,
    expected=None,
    request_timeout_s=600,
    stagger_s=2,
):
    if not payloads or concurrency < 1 or duration_s <= 0 or warmup_s < stagger_s:
        raise ValueError("invalid workload or measurement window")
    origin = time.perf_counter()
    window = Window(origin + warmup_s, origin + warmup_s + duration_s)
    results = []
    active = 0
    peak_active = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=request_timeout_s),
        connector=aiohttp.TCPConnector(limit=concurrency),
        read_bufsize=2**20,
    ) as session:

        async def worker(slot):
            nonlocal active, peak_active
            await asyncio.sleep(stagger_s * slot / concurrency)
            sequence = 0
            while time.perf_counter() < window.end:
                index = (slot + sequence * concurrency) % len(payloads)
                sequence += 1
                body = payloads[index]
                begin = time.perf_counter()
                active += 1
                peak_active = max(peak_active, active)
                count = 0
                first = previous = None
                itls = []
                done = False
                output_ids = None
                max_delta = 0
                try:
                    async with session.post(url, json=body) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            if not line.startswith(b"data:"):
                                continue
                            if line[5:].strip() == b"[DONE]":
                                done = True
                                break
                            data = json.loads(line[5:])
                            if "error" in data:
                                raise RuntimeError(data)
                            now = time.perf_counter()
                            new_count = int(data.get("meta_info", {}).get("completion_tokens", 0))
                            delta = new_count - count
                            window.observe(now, delta)
                            if delta:
                                max_delta = max(max_delta, delta)
                                if first is None:
                                    first = now
                                elif previous is not None:
                                    itls.append((now - previous) / delta)
                                previous = now
                            count = new_count
                            output_ids = data.get("output_ids", output_ids)
                    wanted = body["sampling_params"]["max_new_tokens"]
                    if not done or count != wanted or first is None:
                        raise AssertionError(("incomplete stream", done, count, wanted))
                    if expected is not None and output_ids != expected[index]:
                        raise AssertionError(("token mismatch", index, output_ids))
                    end = time.perf_counter()
                    result = {
                        "payload_index": index,
                        "slot": slot,
                        "sequence": sequence,
                        "start_s": begin - origin,
                        "end_s": end - origin,
                        "cohort": window.start <= begin < window.end,
                        "output": count,
                        "ttft_s": first - begin,
                        "e2e_s": end - begin,
                        "itl_s": itls,
                        "max_event_token_delta": max_delta,
                    }
                    results.append(result)
                    if on_complete:
                        on_complete(result)
                finally:
                    active -= 1

        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    cohort = [r for r in results if r["cohort"]]
    itls = [t for r in cohort for t in r["itl_s"]]
    return {
        "mode": "continuous_closed_loop",
        "concurrency": concurrency,
        "warmup_s": warmup_s,
        "measurement_s": duration_s,
        "drain_s": max(0, time.perf_counter() - window.end),
        "window_output_tokens": window.tokens,
        "output_token_per_s": window.tokens / duration_s,
        "tokens_per_second": [window.bins.get(i, 0) for i in range(int(duration_s))],
        "completed_total": len(results),
        "cohort_requests": len(cohort),
        "failed": 0,
        "peak_client_active": peak_active,
        "requests_crossing_start": sum(r["start_s"] < warmup_s < r["end_s"] for r in results),
        "requests_crossing_end": sum(
            r["start_s"] < warmup_s + duration_s < r["end_s"] for r in results
        ),
        "ttft_p50_s": percentile([r["ttft_s"] for r in cohort], 0.5),
        "ttft_p95_s": percentile([r["ttft_s"] for r in cohort], 0.95),
        "e2e_p50_s": percentile([r["e2e_s"] for r in cohort], 0.5),
        "e2e_p95_s": percentile([r["e2e_s"] for r in cohort], 0.95),
        "itl_p50_s": percentile(itls, 0.5),
        "itl_p95_s": percentile(itls, 0.95),
        "max_event_token_delta": max((r["max_event_token_delta"] for r in results), default=0),
        "exact_token_checks": len(results) if expected is not None else 0,
    }
