#!/usr/bin/env python3
"""
Check trend threshold for nightly CI metrics.

Fetches historical data from the CI data repo, computes sliding window
averages, and alerts via GitHub issue if metrics deviate beyond threshold.
"""

import argparse
import json
import os
import subprocess
import sys
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_OWNER = "pathfinder-pf"
DATA_REPO = "sglang-jax-ci-data"


def get_session(token=None):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    headers = {"User-Agent": "sglang-jax-ci"}
    if token:
        headers["Authorization"] = f"token {token}"
    session.headers.update(headers)
    return session


def get_date_folders(session, subdir):
    """Fetch available date folders from the data repo."""
    api_url = f"https://api.github.com/repos/{DATA_OWNER}/{DATA_REPO}/contents/{subdir}"
    resp = session.get(api_url, timeout=30)
    resp.raise_for_status()
    return sorted([item["name"] for item in resp.json() if item["type"] == "dir"])


def fetch_metric_data(session, subdir, filename, metric, filters=None):
    """Fetch all historical data for a specific metric.

    Returns a list of {"date": str, "value": float} dicts sorted by date.
    filters: optional dict of {column: value} to narrow rows before averaging
             (e.g., {"concurrency": 8, "input": 1024, "output": 1024}).
    """
    dates = get_date_folders(session, subdir)
    if not dates:
        print("No historical data found")
        return None

    all_values = []
    for date_str in dates:
        raw_url = (
            f"https://raw.githubusercontent.com/{DATA_OWNER}/{DATA_REPO}/main"
            f"/{subdir}/{date_str}/{filename}"
        )
        try:
            res = session.get(raw_url, timeout=30)
            if res.status_code == 200:
                df = pd.read_csv(StringIO(res.text))
                # Apply filters if specified (e.g., concurrency=8, input=1024)
                if filters:
                    for col, val in filters.items():
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            df = df[df[col] == val]
                if metric in df.columns:
                    df[metric] = pd.to_numeric(df[metric], errors="coerce")
                    values = df[metric].dropna().tolist()
                    if values:
                        avg_val = sum(values) / len(values)
                        all_values.append({"date": date_str, "value": avg_val})
        except Exception as e:
            print(f"Warning: Failed to fetch data for {date_str}: {e}")

    return all_values


def check_threshold(values, window, threshold_pct, metric_name, higher_is_better=True):
    """Check if recent window average deviates from baseline beyond threshold.

    Baseline is the average of all data points before the most recent `window`
    runs.  Returns (alert: bool, details: dict).
    """
    if len(values) < window + 1:
        print(f"Not enough data points ({len(values)}) for window size {window}")
        return False, {}

    recent = values[-window:]
    baseline_values = values[:-window]

    recent_avg = sum(v["value"] for v in recent) / len(recent)
    baseline_avg = sum(v["value"] for v in baseline_values) / len(baseline_values)

    if abs(baseline_avg) < 1e-9:
        print(f"Baseline average near zero ({baseline_avg}), skipping threshold check")
        return False, {}

    pct_change = ((recent_avg - baseline_avg) / abs(baseline_avg)) * 100

    # For "higher is better" metrics (accuracy, throughput), a negative change is bad.
    # For "lower is better" metrics (latency), a positive change is bad.
    if higher_is_better:
        is_regression = pct_change < -threshold_pct
    else:
        is_regression = pct_change > threshold_pct

    details = {
        "metric": metric_name,
        "baseline_avg": round(baseline_avg, 4),
        "recent_avg": round(recent_avg, 4),
        "pct_change": round(pct_change, 2),
        "threshold_pct": threshold_pct,
        "window": window,
        "is_regression": is_regression,
        "recent_dates": [v["date"] for v in recent],
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
    }

    return is_regression, details


def create_alert_issue(repo, details, token):
    """Create a GitHub issue for a threshold violation."""
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token

    title = (
        f"[CI Alert] {details['metric']} regression detected " f"({details['pct_change']:+.1f}%)"
    )

    # Check for existing open issue with same metric to avoid duplicates
    check = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            f"[CI Alert] {details['metric']} regression",
            "--json",
            "number",
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if check.returncode == 0:
        try:
            existing = json.loads(check.stdout)
            if existing:
                print(
                    f"Open alert issue already exists (#{existing[0]['number']}), skipping creation"
                )
                return
        except json.JSONDecodeError:
            pass

    body_lines = [
        "## Metric Regression Alert",
        "",
        f"**Metric:** `{details['metric']}`",
        f"**Change:** {details['pct_change']:+.2f}% (threshold: {details['threshold_pct']}%)",
        f"**Baseline average:** {details['baseline_avg']}",
        f"**Recent {details['window']}-run average:** {details['recent_avg']}",
        "",
        "### Recent runs analyzed",
        "",
    ]
    for date in details["recent_dates"]:
        body_lines.append(f"- {date}")
    body_lines.extend(
        [
            "",
            "---",
            "*Auto-generated by CI trend threshold check*",
        ]
    )
    body = "\n".join(body_lines)

    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--body",
            body,
            "--label",
            "ci",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        print(f"Created alert issue: {result.stdout.strip()}")
    else:
        print(f"Failed to create issue: {result.stderr}")


# Predefined metric configurations
METRIC_CONFIGS = {
    "gsm8k": {
        "subdir": "benchmark",
        "filename": "daily_qwen_7b_benchmark_results.csv",
        "higher_is_better": True,
    },
    "ttft_ms": {
        "subdir": "perf",
        "filename": "daily_performance_results_QWEN_7B_tp_1.csv",
        "higher_is_better": False,
        "filters": {"concurrency": 8, "input": 1024, "output": 1024},
    },
    "itl_ms": {
        "subdir": "perf",
        "filename": "daily_performance_results_QWEN_7B_tp_1.csv",
        "higher_is_better": False,
        "filters": {"concurrency": 8, "input": 1024, "output": 1024},
    },
    "in_tps": {
        "subdir": "perf",
        "filename": "daily_performance_results_QWEN_7B_tp_1.csv",
        "higher_is_better": True,
        "filters": {"concurrency": 8, "input": 1024, "output": 1024},
    },
    "out_tps": {
        "subdir": "perf",
        "filename": "daily_performance_results_QWEN_7B_tp_1.csv",
        "higher_is_better": True,
        "filters": {"concurrency": 8, "input": 1024, "output": 1024},
    },
}


def main():
    parser = argparse.ArgumentParser(description="Check trend thresholds for CI metrics")
    parser.add_argument(
        "--metric",
        required=True,
        choices=list(METRIC_CONFIGS.keys()),
        help="Metric to check",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Sliding window size (number of recent runs)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=10.0,
        help="Alert threshold as percentage deviation",
    )
    parser.add_argument(
        "--repo",
        default="sgl-project/sglang-jax",
        help="Repo to create alert issues in",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token for data repo access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check only, don't create issues",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    config = METRIC_CONFIGS[args.metric]

    print(f"Checking metric: {args.metric}")
    print(f"Window: {args.window}, Threshold: {args.threshold_pct}%")

    session = get_session(token)
    values = fetch_metric_data(
        session, config["subdir"], config["filename"], args.metric, filters=config.get("filters")
    )

    if not values:
        print("No data available, skipping check")
        return

    print(f"Fetched {len(values)} data points")

    is_regression, details = check_threshold(
        values,
        args.window,
        args.threshold_pct,
        args.metric,
        config["higher_is_better"],
    )

    # Write to GITHUB_STEP_SUMMARY when running in CI
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        status = "REGRESSION" if is_regression else "OK"
        with open(summary_file, "a") as f:
            f.write(f"\n### {args.metric}: {status}\n")
            if details:
                f.write(f"- Change: {details['pct_change']:+.2f}%\n")
                f.write(f"- Baseline: {details['baseline_avg']}\n")
                f.write(f"- Recent avg: {details['recent_avg']}\n")

    if is_regression:
        print(
            f"ALERT: {args.metric} regression detected! " f"Change: {details['pct_change']:+.2f}%"
        )
        if not args.dry_run:
            create_alert_issue(args.repo, details, token)
        else:
            print("(dry run - not creating issue)")
        sys.exit(1)
    else:
        if details:
            print(f"OK: {args.metric} change {details['pct_change']:+.2f}% within threshold")
        else:
            print("OK: Insufficient data for comparison")


if __name__ == "__main__":
    main()
