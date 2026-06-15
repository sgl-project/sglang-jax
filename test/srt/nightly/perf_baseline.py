"""Trailing-baseline lookup from the ci-data repo.

Returns the mean of a sweep point's metric over the last N nights, for the
trailing half of the dual-threshold perf gate. Reads the per-model CSVs at
``perf/<YYYY-MM-DD>/daily_performance_results_<model>_tp_<n>.csv`` (written by
``publish.py --data-type perf``), locating a point by its file (model + tpu_size)
and its ``(concurrency, input, output)`` row.

Degrades safely: any failure (offline, no token, insufficient history, malformed
data) returns ``None`` so the caller falls back to the absolute floor alone.
Network reads are best-effort and never raise.
"""

import csv
import io
import json
import math
import os
import urllib.request
from urllib.error import HTTPError, URLError

CI_DATA_OWNER = "pathfinder-pf"
CI_DATA_REPO = "sglang-jax-ci-data"
CI_DATA_PERF_SUBDIR = "perf"

# Trailing gate: current must be >= mean(last N nights) * (1 - tolerance);
# skipped until N nights of history exist. Placeholders pending observed noise.
TRAILING_TOLERANCE = 0.05
TRAILING_MIN_NIGHTS = 5


# Regression direction per metric: "higher" = a regression is a drop
# (throughput), "lower" = a regression is a rise (latency). Single source of
# truth for both the trailing gate (gated_metrics) and the absolute-floor gate
# (results.gate_perf_result), so a floor on a latency metric is checked the
# right way round.
METRIC_DIRECTION: dict[str, str] = {
    "in_tps": "higher",
    "out_tps": "higher",
    "ttft_ms": "lower",
    "itl_ms": "lower",
}


# Metrics the trailing gate compares, by point type. A prefill-only point
# (output<=1) has no meaningful decode throughput/ITL, so it gates prefill
# throughput + TTFT; a decode point adds output throughput + ITL.
def gated_metrics(output_len: int) -> dict[str, str]:
    if output_len <= 1:
        metrics = ("in_tps", "ttft_ms")
    else:
        metrics = ("in_tps", "out_tps", "itl_ms")
    return {m: METRIC_DIRECTION[m] for m in metrics}


# Per-process caches: a gate run reads the same date folders / CSV many times
# (every point × metric) — fetch once. Failures cache too (via the _FETCHED
# flag), so an unreachable API isn't re-hit on every lookup.
_DATE_FOLDERS_CACHE: list[str] | None = None
_DATE_FOLDERS_FETCHED = False
_CSV_ROWS_CACHE: dict[tuple[str, str], list[dict]] = {}


def _get(url: str, token: str | None, timeout: int = 30) -> bytes | None:
    headers = {"User-Agent": "sglang-jax-perf-baseline"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    # Retry once on a connection-level fault (DNS/reset/timeout); any HTTPError
    # (404, 403 rate-limit, even 5xx) is returned as-is without retry.
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError:
            return None
        except (URLError, TimeoutError, OSError):
            if attempt == 0:
                continue
            return None
    return None


def _list_date_folders(token: str | None) -> list[str]:
    """Sorted ``perf/<date>`` folder names (oldest first), or []. Cached incl. failures."""
    global _DATE_FOLDERS_CACHE, _DATE_FOLDERS_FETCHED
    if _DATE_FOLDERS_FETCHED:
        return _DATE_FOLDERS_CACHE or []
    _DATE_FOLDERS_FETCHED = True
    url = (
        f"https://api.github.com/repos/{CI_DATA_OWNER}/{CI_DATA_REPO}"
        f"/contents/{CI_DATA_PERF_SUBDIR}"
    )
    raw = _get(url, token)
    if raw is None:
        return []
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return []
    _DATE_FOLDERS_CACHE = sorted(
        i["name"] for i in items if isinstance(i, dict) and i.get("type") == "dir"
    )
    return _DATE_FOLDERS_CACHE


def _fetch_rows(date_str: str, csv_filename: str, token: str | None) -> list[dict] | None:
    """Download + parse one night's CSV into rows (cached per (date, file))."""
    key = (date_str, csv_filename)
    if key in _CSV_ROWS_CACHE:
        return _CSV_ROWS_CACHE[key]
    raw_url = (
        f"https://raw.githubusercontent.com/{CI_DATA_OWNER}/{CI_DATA_REPO}/main"
        f"/{CI_DATA_PERF_SUBDIR}/{date_str}/{csv_filename}"
    )
    raw = _get(raw_url, token)
    if raw is None:
        return None  # don't cache a transient miss — let a later lookup retry
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    _CSV_ROWS_CACHE[key] = rows
    return rows


def _fetch_metric(
    date_str: str,
    csv_filename: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    column: str,
    token: str | None,
) -> float | None:
    """Read one night's ``column`` for the matching sweep row, or None."""
    rows = _fetch_rows(date_str, csv_filename, token)
    if not rows:
        return None
    try:
        for row in rows:
            if (
                int(row["concurrency"]) == concurrency
                and int(row["input"]) == input_len
                and int(row["output"]) == output_len
            ):
                val = row.get(column)
                if val in (None, ""):
                    return None
                f = float(val)
                # Drop non-finite: a NaN in the window makes the mean NaN and
                # `value < NaN` always False (a silent pass).
                return f if math.isfinite(f) else None
    except (ValueError, TypeError, KeyError):
        return None
    return None


def fetch_trailing_baseline(
    csv_filename: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    column: str,
    n: int,
    token: str | None = None,
) -> float | None:
    """Mean of one sweep point's ``column`` over the most recent ``n`` nights.

    The point is identified by its per-model CSV file (``csv_filename``) and the
    ``(concurrency, input, output)`` row within it. Returns ``None`` (caller
    skips the trailing gate) when fewer than ``n`` nights carry a usable value,
    or on any network/parse failure. ``token`` defaults to ``$GITHUB_TOKEN``;
    the ci-data repo is public so anonymous reads also work (subject to a lower
    rate limit).
    """
    if token is None:
        token = os.getenv("GITHUB_TOKEN")

    dates = _list_date_folders(token)
    if not dates:
        return None

    # Walk newest-first, collecting up to n usable values.
    values: list[float] = []
    for date_str in reversed(dates):
        val = _fetch_metric(
            date_str, csv_filename, concurrency, input_len, output_len, column, token
        )
        if val is not None:
            values.append(val)
        if len(values) >= n:
            break

    if len(values) < n:
        return None
    return sum(values) / len(values)
