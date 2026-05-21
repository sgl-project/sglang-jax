#!/usr/bin/env python3
"""
Runner Utilization Report

Analyzes GitHub Actions job data to calculate runner utilization metrics.
Reports idle time, active time, and utilization percentage per runner label.
"""

import argparse
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

DEFAULT_LABELS_TO_IGNORE = {"self-hosted", "Linux", "X64", "ARM64"}
GITHUB_HOSTED_LABELS = {"ubuntu-latest", "ubuntu-22.04", "ubuntu-24.04"}


def run_gh_command(args, max_retries=10):
    """Run a gh CLI command with exponential backoff retry on failure."""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            wait = (2**attempt) + random.uniform(0, 1)
            print(
                f"Command failed (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s: {e.stderr.strip()}"
            )
            time.sleep(wait)


def get_workflow_runs(repo, hours=24):
    """Fetch workflow runs from the past N hours with pagination."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    output = run_gh_command(
        [
            "api",
            f"/repos/{repo}/actions/runs",
            "--paginate",
            "-f",
            "per_page=100",
            "--jq",
            ".workflow_runs[]",
        ]
    )

    runs = []
    for line in output.strip().splitlines():
        if line.strip():
            try:
                run = json.loads(line)
                created = parse_time(run.get("created_at"))
                if created and created < cutoff:
                    continue
                runs.append(run)
            except json.JSONDecodeError:
                continue

    return runs


def get_jobs_for_run(repo, run_id):
    """Fetch all jobs for a workflow run, including retried attempts."""
    jobs = []
    page = 1
    per_page = 100

    while True:
        output = run_gh_command(
            [
                "api",
                f"/repos/{repo}/actions/runs/{run_id}/jobs",
                "-f",
                "filter=all",
                "-f",
                f"per_page={per_page}",
                "-f",
                f"page={page}",
                "--jq",
                ".jobs[]",
            ]
        )

        if not output.strip():
            break

        page_jobs = []
        for line in output.strip().splitlines():
            if line.strip():
                try:
                    page_jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not page_jobs:
            break

        jobs.extend(page_jobs)

        if len(page_jobs) < per_page:
            break

        page += 1

    return jobs


def get_runners(repo, online_only=True):
    """Fetch self-hosted runners for the repo; degrades gracefully if no admin access."""
    try:
        output = run_gh_command(
            [
                "api",
                f"/repos/{repo}/actions/runners",
                "--paginate",
                "--jq",
                ".runners[]",
            ]
        )
        runners = []
        for line in output.strip().splitlines():
            if line.strip():
                try:
                    runner = json.loads(line)
                    if online_only and runner.get("status") != "online":
                        continue
                    runners.append(runner)
                except json.JSONDecodeError:
                    continue
        return runners
    except subprocess.CalledProcessError as e:
        err_str = str(e.stderr)
        if "rate limit" in err_str.lower():
            print(
                "Warning: GitHub API rate limit hit while listing runners; runner count will be unavailable."
            )
            return []
        if "403" in err_str or "Must have admin rights" in err_str:
            print(
                "Note: No admin access to list runners; runner count will be estimated from job data."
            )
            return []
        raise


def parse_time(time_str):
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime."""
    if not time_str:
        return None
    # Handle both 'Z' suffix and '+00:00' offset forms
    time_str = time_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        return None


def calculate_concurrency_metrics(jobs, window_start, window_end, num_runners):
    """
    Use a sweep-line algorithm to compute peak and average concurrent jobs,
    saturation percentage, and peak queue depth.
    """
    events = []
    for job in jobs:
        start = parse_time(job.get("started_at"))
        end = parse_time(job.get("completed_at"))
        if start and end and start < end:
            # Clamp to the reporting window
            start = max(start, window_start)
            end = min(end, window_end)
            if start < end:
                events.append((start, +1))
                events.append((end, -1))

    if not events:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_pct": 0.0,
            "peak_queue": 0,
        }

    # At equal timestamps, process end events (delta=-1) before start events
    # (delta=+1) to avoid transiently inflating the concurrent count.
    events.sort(key=lambda e: (e[0], -e[1]))

    current = 0
    peak = 0
    # Weighted sum of (concurrent_count * duration_seconds) for average
    weighted_sum = 0.0
    prev_time = window_start

    for event_time, delta in events:
        if event_time > prev_time:
            duration = (event_time - prev_time).total_seconds()
            weighted_sum += current * duration
            prev_time = event_time
        current += delta
        peak = max(peak, current)

    # Final segment to window_end
    if prev_time < window_end:
        duration = (window_end - prev_time).total_seconds()
        weighted_sum += current * duration

    window_seconds = (window_end - window_start).total_seconds()
    avg_concurrent = weighted_sum / window_seconds if window_seconds > 0 else 0.0

    saturation_pct = (avg_concurrent / num_runners * 100.0) if num_runners > 0 else 0.0
    peak_queue = max(0, peak - num_runners)

    return {
        "peak_concurrent": peak,
        "avg_concurrent": avg_concurrent,
        "saturation_pct": saturation_pct,
        "peak_queue": peak_queue,
    }


_NON_GPU_WORKFLOW_HINTS = (
    "lint",
    "release",
    "runner utilization",
    "summize",  # CI Trend Summary (nightly-test-summize)
)


def _likely_no_gpu_jobs(workflow_name):
    """Return True if this workflow name suggests it has no GPU/TPU jobs."""
    name_lower = workflow_name.lower()
    return any(hint in name_lower for hint in _NON_GPU_WORKFLOW_HINTS)


def calculate_utilization(repo, hours=24, runner_filter=None):
    """
    Main calculation: fetch runs and jobs, aggregate per-label utilization,
    concurrency metrics, and host-level busy time.
    """
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)

    print(f"Fetching workflow runs for {repo} in the past {hours} hours...")
    runs = get_workflow_runs(repo, hours=hours)
    print(f"Found {len(runs)} workflow runs.")

    # Fetch runners once; fall back gracefully
    runners_list = get_runners(repo, online_only=False)

    # Build label -> runner count map
    label_runner_count = defaultdict(int)
    for runner in runners_list:
        labels = {lbl["name"] for lbl in runner.get("labels", [])}
        for lbl in labels:
            if lbl not in DEFAULT_LABELS_TO_IGNORE and lbl not in GITHUB_HOSTED_LABELS:
                label_runner_count[lbl] += 1

    fetch_failures = 0
    total_runs = len(runs)

    # Per-label: list of (started_at, completed_at) intervals
    label_intervals = defaultdict(list)
    # Per-label: all jobs (for concurrency sweep)
    label_jobs = defaultdict(list)

    def fetch_run_jobs(run):
        if _likely_no_gpu_jobs(run.get("name", "")):
            return run["id"], []
        try:
            return run["id"], get_jobs_for_run(repo, run["id"])
        except Exception as e:
            print(f"Warning: failed to fetch jobs for run {run['id']}: {e}")
            return run["id"], None

    print("Fetching job details (parallel, max 4 workers)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_run_jobs, run): run for run in runs}
        for future in as_completed(futures):
            try:
                run_id, jobs = future.result()
            except Exception as e:
                print(f"Warning: unexpected error fetching jobs: {e}")
                fetch_failures += 1
                continue
            if jobs is None:
                fetch_failures += 1
                continue
            for job in jobs:
                if job.get("status") != "completed":
                    continue
                labels = set(job.get("labels", []))
                # Strip generic labels to identify the meaningful runner label
                meaningful = labels - DEFAULT_LABELS_TO_IGNORE - GITHUB_HOSTED_LABELS
                if not meaningful:
                    continue
                if runner_filter and not any(runner_filter in lbl for lbl in meaningful):
                    continue

                started = parse_time(job.get("started_at"))
                completed = parse_time(job.get("completed_at"))
                if not started or not completed or completed <= started:
                    continue
                # Clamp to window
                eff_start = max(started, window_start)
                eff_end = min(completed, window_end)
                if eff_start >= eff_end:
                    continue

                for lbl in meaningful:
                    label_intervals[lbl].append((eff_start, eff_end))
                    label_jobs[lbl].append(job)

    fetch_failure_pct = (fetch_failures / total_runs * 100.0) if total_runs > 0 else 0.0
    window_seconds = (window_end - window_start).total_seconds()

    results = {}
    all_labels = set(label_intervals.keys()) | set(label_runner_count.keys())

    for label in sorted(all_labels):
        if runner_filter and runner_filter not in label:
            continue
        intervals = label_intervals.get(label, [])
        num_runners = label_runner_count.get(label, 0)

        # Merge overlapping intervals to compute total busy time
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        busy_seconds = sum((e - s).total_seconds() for s, e in merged)

        # Capacity: num_runners * window_seconds (0 if unknown)
        capacity_seconds = num_runners * window_seconds if num_runners > 0 else 0.0
        utilization_pct = (
            (busy_seconds / capacity_seconds * 100.0) if capacity_seconds > 0 else None
        )

        # Total job seconds (sum of all individual job durations, not merged)
        total_job_seconds = sum((e - s).total_seconds() for s, e in intervals)

        concurrency = calculate_concurrency_metrics(
            label_jobs.get(label, []),
            window_start,
            window_end,
            num_runners if num_runners > 0 else 1,
        )

        results[label] = {
            "num_runners": num_runners,
            "busy_seconds": busy_seconds,
            "capacity_seconds": capacity_seconds,
            "utilization_pct": utilization_pct,
            "total_job_seconds": total_job_seconds,
            "job_count": len(label_jobs.get(label, [])),
            "concurrency": concurrency,
        }

    return results, fetch_failure_pct


def format_report(results, hours, fetch_failure_pct=0.0):
    """Format the utilization results as a Markdown report."""
    lines = []
    lines.append(f"# Runner Utilization Report (Past {hours} Hours)")
    lines.append("")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"_Generated at {now_utc}_")
    lines.append("")

    if fetch_failure_pct > 0:
        lines.append(
            f"> **Warning:** {fetch_failure_pct:.1f}% of workflow runs could not be fetched (API errors). Results may be incomplete."
        )
        lines.append("")

    if not results:
        lines.append("No runner activity found in the specified time window.")
        return "\n".join(lines)

    # Utilization table
    lines.append("## Utilization by Runner Label")
    lines.append("")
    lines.append("| Runner Label | Runners | Jobs | Busy Time | Capacity | Utilization |")
    lines.append("|---|---|---|---|---|---|")

    for label, data in sorted(results.items()):
        num_runners = data["num_runners"] or "N/A"
        job_count = data["job_count"]
        busy_h = data["busy_seconds"] / 3600
        cap_h = data["capacity_seconds"] / 3600 if data["capacity_seconds"] else None
        util = data["utilization_pct"]

        busy_str = f"{busy_h:.1f}h"
        cap_str = f"{cap_h:.1f}h" if cap_h is not None else "N/A"
        util_str = f"{util:.1f}%" if util is not None else "N/A"

        lines.append(
            f"| `{label}` | {num_runners} | {job_count} | {busy_str} | {cap_str} | {util_str} |"
        )

    lines.append("")

    # Concurrency analysis
    lines.append("## Concurrency Analysis")
    lines.append("")
    lines.append("| Runner Label | Peak Concurrent | Avg Concurrent | Saturation | Peak Queue |")
    lines.append("|---|---|---|---|---|")

    for label, data in sorted(results.items()):
        conc = data["concurrency"]
        peak = conc["peak_concurrent"]
        avg = conc["avg_concurrent"]
        sat = conc["saturation_pct"]
        queue = conc["peak_queue"]

        lines.append(f"| `{label}` | {peak} | {avg:.2f} | {sat:.1f}% | {queue} |")

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    has_recommendation = False
    for label, data in sorted(results.items()):
        util = data["utilization_pct"]
        conc = data["concurrency"]

        if util is not None and util > 80:
            lines.append(
                f"- **`{label}`**: High utilization ({util:.1f}%). Consider adding more runners to reduce queue time."
            )
            has_recommendation = True
        elif util is not None and util < 10 and data["job_count"] > 0:
            lines.append(
                f"- **`{label}`**: Low utilization ({util:.1f}%). Runner capacity may be over-provisioned."
            )
            has_recommendation = True

        if conc["peak_queue"] > 0:
            lines.append(
                f"- **`{label}`**: Peak queue depth of {conc['peak_queue']} job(s) detected. Jobs had to wait for a free runner."
            )
            has_recommendation = True

    if not has_recommendation:
        lines.append(
            "No immediate action required. All runner labels appear within normal utilization bounds."
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate runner utilization report")
    parser.add_argument(
        "--repo", default="sgl-project/sglang-jax", help="GitHub repository (owner/name)"
    )
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours (default: 24)")
    parser.add_argument(
        "--filter", type=str, dest="filter", help="Filter runner labels (substring match)"
    )
    parser.add_argument("--output", type=str, help="Output file path (default: stdout)")
    args = parser.parse_args()

    results, fetch_failure_pct = calculate_utilization(
        repo=args.repo,
        hours=args.hours,
        runner_filter=args.filter,
    )

    report = format_report(results, hours=args.hours, fetch_failure_pct=fetch_failure_pct)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Write to GitHub Actions step summary if available
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(report)
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
