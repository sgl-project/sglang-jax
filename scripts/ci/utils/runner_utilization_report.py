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
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

DEFAULT_LABELS_TO_IGNORE = {"self-hosted", "Linux", "X64", "ARM64"}
GITHUB_HOSTED_LABELS = {"ubuntu-latest", "ubuntu-22.04", "ubuntu-24.04"}


_NON_RETRYABLE_PATTERNS = ("401", "403", "404", "422", "not found", "must have admin")


def run_gh_command(args, max_retries=5):
    """Run a gh CLI command with exponential backoff retry on transient failure."""
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
            err = str(e.stderr)
            err_lower = err.lower()
            if "rate limit" in err_lower:
                pass  # fall through to backoff
            elif any(p in err_lower for p in _NON_RETRYABLE_PATTERNS):
                raise
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            print(
                f"Command failed (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s: {e.stderr.strip()}"
            )
            time.sleep(wait)


def get_workflow_runs(repo, hours=24):
    """Fetch workflow runs from the past N hours with manual pagination and early stop."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    runs = []
    page = 1
    per_page = 100

    while True:
        output = run_gh_command(
            [
                "api",
                f"/repos/{repo}/actions/runs",
                "--method",
                "GET",
                "-f",
                f"per_page={per_page}",
                "-f",
                f"page={page}",
                "-f",
                "sort=created",
                "-f",
                "direction=desc",
                "--jq",
                ".workflow_runs[]",
            ]
        )

        if not output.strip():
            break

        page_count = 0
        hit_cutoff = False
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            page_count += 1
            try:
                run = json.loads(line)
                created = parse_time(run.get("created_at"))
                if created and created < cutoff:
                    hit_cutoff = True
                    continue
                runs.append(run)
            except json.JSONDecodeError:
                continue

        if hit_cutoff or page_count < per_page:
            break

        page += 1

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
                "--method",
                "GET",
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


def get_runners(repo, online_only=False):
    """Fetch self-hosted runners for the repo; degrades gracefully if no admin access."""
    try:
        output = run_gh_command(
            [
                "api",
                f"/repos/{repo}/actions/runners",
                "--method",
                "GET",
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

        # Infer labels from runner name when labels array is empty
        for runner in runners:
            labels = {lbl["name"] for lbl in runner.get("labels", [])}
            meaningful = labels - DEFAULT_LABELS_TO_IGNORE - GITHUB_HOSTED_LABELS
            if not meaningful:
                name = runner.get("name", "")
                m = re.match(r"^(arc-runner-v6e-\d+)-", name)
                if m:
                    inferred = m.group(1)
                elif "cpu-runner" in name or name.startswith("runnerdeploy-"):
                    inferred = "arc-runner-cpu"
                else:
                    inferred = None
                if inferred:
                    runner.setdefault("labels", []).append({"name": inferred})

        return runners
    except subprocess.CalledProcessError as e:
        err_str = str(e.stderr)
        if "rate limit" in err_str.lower():
            print(
                "Warning: GitHub API rate limit hit while listing runners; runner count will be unavailable."
            )
            return []
        if "403" in err_str or "must have admin" in err_str.lower():
            print(
                "Note: No admin access to list runners; runner count will be estimated from job data."
            )
            return []
        raise


def get_fleet_status(runners):
    """Aggregate per-label fleet status from runner list."""
    status = defaultdict(lambda: {"total": 0, "online": 0, "offline": 0, "busy": 0, "idle": 0})
    for runner in runners:
        labels = {lbl["name"] for lbl in runner.get("labels", [])}
        meaningful = labels - DEFAULT_LABELS_TO_IGNORE - GITHUB_HOSTED_LABELS
        for lbl in meaningful:
            s = status[lbl]
            s["total"] += 1
            if runner.get("status") == "online":
                s["online"] += 1
                if runner.get("busy"):
                    s["busy"] += 1
                else:
                    s["idle"] += 1
            else:
                s["offline"] += 1
    return dict(status)


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
            "saturation_pct": None,
            "peak_queue": None,
        }

    events.sort(key=lambda e: (e[0], e[1]))

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

    saturation_pct = (avg_concurrent / num_runners * 100.0) if num_runners > 0 else None
    peak_queue = max(0, peak - num_runners) if num_runners > 0 else None

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


def _percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * p / 100)
    return sorted_values[min(idx, len(sorted_values) - 1)]


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

    fleet_status = get_fleet_status(runners_list)

    # Build label -> runner count map
    label_runner_count = defaultdict(int)
    for runner in runners_list:
        labels = {lbl["name"] for lbl in runner.get("labels", [])}
        for lbl in labels:
            if lbl not in DEFAULT_LABELS_TO_IGNORE and lbl not in GITHUB_HOSTED_LABELS:
                label_runner_count[lbl] += 1

    fetch_failures = 0
    total_runs = len(runs)

    label_intervals = defaultdict(list)
    label_jobs = defaultdict(list)
    label_queue_waits = defaultdict(list)
    label_job_durations = defaultdict(list)
    label_conclusions = defaultdict(list)
    all_jobs_enriched = []
    workflow_stats = defaultdict(
        lambda: defaultdict(lambda: {"job_count": 0, "total_duration": 0.0})
    )
    hourly_buckets = defaultdict(int)

    def fetch_run_jobs(run):
        wf_name = run.get("name", "")
        run_url = run.get("html_url", "")
        if _likely_no_gpu_jobs(wf_name):
            return run["id"], wf_name, run_url, []
        try:
            return run["id"], wf_name, run_url, get_jobs_for_run(repo, run["id"])
        except Exception as e:
            print(f"Warning: failed to fetch jobs for run {run['id']}: {e}")
            return run["id"], wf_name, run_url, None

    print("Fetching job details (parallel, max 4 workers)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_run_jobs, run): run for run in runs}
        for future in as_completed(futures):
            try:
                run_id, wf_name, run_url, jobs = future.result()
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
                created = parse_time(job.get("created_at"))
                if not started or not completed or completed <= started:
                    continue

                wait_s = 0.0
                if created and started > created:
                    wait_s = (started - created).total_seconds()
                    for lbl in meaningful:
                        label_queue_waits[lbl].append(wait_s)

                eff_start = max(started, window_start)
                eff_end = min(completed, window_end)
                if eff_start >= eff_end:
                    continue

                for lbl in meaningful:
                    label_intervals[lbl].append((eff_start, eff_end))
                    label_jobs[lbl].append(job)

                duration_s = (eff_end - eff_start).total_seconds()
                for lbl in meaningful:
                    label_job_durations[lbl].append(duration_s)
                    label_conclusions[lbl].append(job.get("conclusion", "unknown"))
                    workflow_stats[wf_name][lbl]["job_count"] += 1
                    workflow_stats[wf_name][lbl]["total_duration"] += duration_s

                all_jobs_enriched.append(
                    {
                        "workflow": wf_name,
                        "job_name": job.get("name", ""),
                        "duration": duration_s,
                        "wait": wait_s,
                        "conclusion": job.get("conclusion", "unknown"),
                        "label": ", ".join(sorted(meaningful)),
                        "url": job.get("html_url", run_url),
                    }
                )

                hourly_buckets[started.hour] += 1

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
        total_job_seconds = sum((e - s).total_seconds() for s, e in intervals)

        capacity_seconds = num_runners * window_seconds if num_runners > 0 else 0.0
        utilization_pct = (
            (total_job_seconds / capacity_seconds * 100.0) if capacity_seconds > 0 else None
        )

        concurrency = calculate_concurrency_metrics(
            label_jobs.get(label, []),
            window_start,
            window_end,
            num_runners,
        )

        waits = sorted(label_queue_waits.get(label, []))
        durations = sorted(label_job_durations.get(label, []))
        conclusions = label_conclusions.get(label, [])
        success_count = conclusions.count("success")
        failure_count = conclusions.count("failure")
        cancelled_count = conclusions.count("cancelled")
        total_concluded = len(conclusions)

        results[label] = {
            "num_runners": num_runners,
            "busy_seconds": busy_seconds,
            "capacity_seconds": capacity_seconds,
            "utilization_pct": utilization_pct,
            "total_job_seconds": total_job_seconds,
            "job_count": len(label_jobs.get(label, [])),
            "concurrency": concurrency,
            "avg_queue_wait": sum(waits) / len(waits) if waits else 0.0,
            "max_queue_wait": max(waits) if waits else 0.0,
            "wait_p50": _percentile(waits, 50),
            "wait_p95": _percentile(waits, 95),
            "duration_p50": _percentile(durations, 50),
            "duration_p95": _percentile(durations, 95),
            "duration_p99": _percentile(durations, 99),
            "duration_max": max(durations) if durations else 0.0,
            "success_count": success_count,
            "failure_count": failure_count,
            "cancelled_count": cancelled_count,
            "total_concluded": total_concluded,
        }

    return {
        "per_label": results,
        "fleet_status": fleet_status,
        "fetch_failure_pct": fetch_failure_pct,
        "all_jobs": all_jobs_enriched,
        "workflow_stats": dict(workflow_stats),
        "hourly_buckets": dict(hourly_buckets),
    }


def _format_duration(seconds):
    """Format seconds into a human-readable string (e.g. '2m 30s', '1h 5m')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_report(report_data, hours):
    """Format the utilization results as a Markdown report."""
    results = report_data["per_label"]
    fleet_status = report_data["fleet_status"]
    fetch_failure_pct = report_data["fetch_failure_pct"]
    all_jobs = report_data["all_jobs"]
    workflow_stats = report_data["workflow_stats"]
    hourly_buckets = report_data["hourly_buckets"]

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

    if not results and not fleet_status:
        lines.append("No runner activity found in the specified time window.")
        return "\n".join(lines)

    # Fleet Status
    lines.append("## Fleet Status")
    lines.append("")
    if fleet_status:
        lines.append("| Runner Label | Total | Online | Offline | Busy | Idle |")
        lines.append("|---|---|---|---|---|---|")
        for label, s in sorted(fleet_status.items()):
            lines.append(
                f"| `{label}` | {s['total']} | {s['online']} | {s['offline']} | {s['busy']} | {s['idle']} |"
            )
    else:
        lines.append("No runner data available (admin token required).")
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
    lines.append(
        "| Runner Label | Peak Concurrent | Avg Concurrent | Saturation | Peak Queue | Wait P50 | Wait P95 | Wait Max |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for label, data in sorted(results.items()):
        conc = data["concurrency"]
        peak = conc["peak_concurrent"]
        avg = conc["avg_concurrent"]
        sat = conc["saturation_pct"]
        queue = conc["peak_queue"]
        sat_str = f"{sat:.1f}%" if sat is not None else "N/A"
        queue_str = str(queue) if queue is not None else "N/A"
        wait_p50 = _format_duration(data["wait_p50"])
        wait_p95 = _format_duration(data["wait_p95"])
        max_wait = _format_duration(data["max_queue_wait"])

        lines.append(
            f"| `{label}` | {peak} | {avg:.2f} | {sat_str} | {queue_str} | {wait_p50} | {wait_p95} | {max_wait} |"
        )

    lines.append("")

    # Job Duration
    lines.append("## Job Duration")
    lines.append("")
    lines.append("| Runner Label | Jobs | P50 | P95 | P99 | Max |")
    lines.append("|---|---|---|---|---|---|")

    for label, data in sorted(results.items()):
        job_count = data["job_count"]
        p50 = _format_duration(data["duration_p50"])
        p95 = _format_duration(data["duration_p95"])
        p99 = _format_duration(data["duration_p99"])
        dmax = _format_duration(data["duration_max"])
        lines.append(f"| `{label}` | {job_count} | {p50} | {p95} | {p99} | {dmax} |")

    lines.append("")

    # Top 10 Workflows by Runner Time
    lines.append("## Top 10 Workflows by Runner Time")
    lines.append("")
    lines.append("| # | Workflow | Jobs | Total Time | Avg Duration | Runner Labels |")
    lines.append("|---|---|---|---|---|---|")

    # Aggregate across labels per workflow
    wf_aggregates = {}
    for wf_name, label_data in workflow_stats.items():
        total_jobs = sum(v["job_count"] for v in label_data.values())
        total_duration = sum(v["total_duration"] for v in label_data.values())
        labels_used = sorted(label_data.keys())
        wf_aggregates[wf_name] = {
            "total_jobs": total_jobs,
            "total_duration": total_duration,
            "labels": labels_used,
        }

    sorted_wfs = sorted(wf_aggregates.items(), key=lambda x: x[1]["total_duration"], reverse=True)
    for rank, (wf_name, agg) in enumerate(sorted_wfs[:10], start=1):
        avg_duration = agg["total_duration"] / agg["total_jobs"] if agg["total_jobs"] > 0 else 0.0
        labels_str = ", ".join(agg["labels"])
        lines.append(
            f"| {rank} | {wf_name} | {agg['total_jobs']} | {_format_duration(agg['total_duration'])} | {_format_duration(avg_duration)} | {labels_str} |"
        )

    lines.append("")

    # Top 10 Slowest Jobs
    lines.append("## Top 10 Slowest Jobs")
    lines.append("")
    lines.append("| # | Workflow | Job | Duration | Wait | Runner Label | Link |")
    lines.append("|---|---|---|---|---|---|---|")

    sorted_jobs = sorted(all_jobs, key=lambda j: j["duration"], reverse=True)
    for rank, job in enumerate(sorted_jobs[:10], start=1):
        lines.append(
            f"| {rank} | {job['workflow']} | {job['job_name']} | {_format_duration(job['duration'])} | {_format_duration(job['wait'])} | {job['label']} | [Run]({job['url']}) |"
        )

    lines.append("")

    # Job Success Rate
    lines.append("## Job Success Rate")
    lines.append("")
    lines.append("| Runner Label | Total | Success | Failure | Cancelled | Success Rate |")
    lines.append("|---|---|---|---|---|---|")

    for label, data in sorted(results.items()):
        total = data["total_concluded"]
        success = data["success_count"]
        failure = data["failure_count"]
        cancelled = data["cancelled_count"]
        rate_str = f"{success / total * 100:.1f}%" if total > 0 else "N/A"
        lines.append(f"| `{label}` | {total} | {success} | {failure} | {cancelled} | {rate_str} |")

    lines.append("")

    # Failed Jobs (max 20)
    lines.append("## Failed Jobs")
    lines.append("")
    failed_jobs = [j for j in all_jobs if j["conclusion"] == "failure"]
    if failed_jobs:
        failed_jobs_sorted = sorted(failed_jobs, key=lambda j: j["duration"], reverse=True)
        lines.append("| Workflow | Job | Duration | Runner Label | Link |")
        lines.append("|---|---|---|---|---|")
        for job in failed_jobs_sorted[:20]:
            lines.append(
                f"| {job['workflow']} | {job['job_name']} | {_format_duration(job['duration'])} | {job['label']} | [Run]({job['url']}) |"
            )
    else:
        lines.append("No failed jobs in this period.")

    lines.append("")

    # Hourly Distribution
    lines.append("## Hourly Distribution (UTC)")
    lines.append("")
    active_hours = [(h, hourly_buckets[h]) for h in range(24) if hourly_buckets.get(h, 0) > 0]
    if active_hours:
        lines.append("| Hour | Jobs Started |")
        lines.append("|---|---|")
        for h, count in active_hours:
            lines.append(f"| {h:02d}:00 | {count} |")
    else:
        lines.append("No jobs started in this period.")

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    has_recommendation = False

    # Grand total across all workflows for dominance check
    grand_total_duration = sum(agg["total_duration"] for agg in wf_aggregates.values())

    for label, data in sorted(results.items()):
        util = data["utilization_pct"]
        conc = data["concurrency"]
        total_concluded = data["total_concluded"]
        failure_count = data["failure_count"]

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

        if conc["peak_queue"] is not None and conc["peak_queue"] > 0:
            lines.append(
                f"- **`{label}`**: Peak queue depth of {conc['peak_queue']} job(s) detected. Jobs had to wait for a free runner."
            )
            has_recommendation = True

        if total_concluded > 5 and failure_count / total_concluded > 0.2:
            lines.append(
                f"- **`{label}`**: High failure rate ({failure_count}/{total_concluded} jobs, {failure_count / total_concluded * 100:.1f}%). Investigate job failures."
            )
            has_recommendation = True

        if data["wait_p95"] > 600:
            lines.append(
                f"- **`{label}`**: Long queue wait detected (P95 wait = {_format_duration(data['wait_p95'])}). Consider scaling runners."
            )
            has_recommendation = True

    # Fleet offline runner warnings
    for label, s in sorted(fleet_status.items()):
        if s["offline"] > 0:
            lines.append(
                f"- **`{label}`**: {s['offline']} runner(s) are offline. Check runner health."
            )
            has_recommendation = True

    # Workflow dominance warning
    if grand_total_duration > 0:
        for wf_name, agg in sorted_wfs:
            if agg["total_duration"] > 0.5 * grand_total_duration:
                lines.append(
                    f"- **Workflow `{wf_name}`** accounts for more than 50% of total runner time ({_format_duration(agg['total_duration'])} of {_format_duration(grand_total_duration)}). Review if this is expected."
                )
                has_recommendation = True

    if not has_recommendation:
        lines.append(
            "No immediate action required. All runner labels appear within normal utilization bounds."
        )

    lines.append("")
    return "\n".join(lines)


def format_slack_summary(report_data, hours):
    """Format a compact plain-text summary for Slack (no markdown tables)."""
    results = report_data["per_label"]
    fleet_status = report_data["fleet_status"]
    recommendations = []

    lines = [f"*Runner Utilization Report (Past {hours} Hours)*", ""]

    for label in sorted(set(list(results.keys()) + list(fleet_status.keys()))):
        parts = [f"• *{label}*:"]
        fs = fleet_status.get(label)
        if fs:
            parts.append(f"{fs['total']} runners ({fs['online']} online, {fs['busy']} busy)")
        data = results.get(label)
        if data:
            parts.append(f"{data['job_count']} jobs")
            util = data["utilization_pct"]
            if util is not None:
                parts.append(f"utilization {util:.1f}%")
            if data["wait_p95"] > 0:
                parts.append(f"wait P95 {_format_duration(data['wait_p95'])}")
            rate = (
                f"{data['success_count'] / data['total_concluded'] * 100:.0f}%"
                if data["total_concluded"] > 0
                else "N/A"
            )
            parts.append(f"success rate {rate}")
        lines.append(" | ".join(parts))

    for label, data in sorted(results.items()):
        if data["wait_p95"] > 600:
            recommendations.append(
                f"⚠ {label}: P95 queue wait {_format_duration(data['wait_p95'])}"
            )
        if data["total_concluded"] > 5 and data["failure_count"] / data["total_concluded"] > 0.2:
            recommendations.append(
                f"⚠ {label}: {data['failure_count']}/{data['total_concluded']} jobs failed"
            )

    if recommendations:
        lines.append("")
        lines.extend(recommendations)
    else:
        lines.append("")
        lines.append("✓ No issues detected")

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
    parser.add_argument("--slack-summary", type=str, help="Write Slack summary to this file")
    args = parser.parse_args()

    if args.hours < 1:
        parser.error("--hours must be a positive integer")

    report_data = calculate_utilization(
        repo=args.repo,
        hours=args.hours,
        runner_filter=args.filter,
    )

    report = format_report(report_data, hours=args.hours)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    if args.slack_summary:
        slack = format_slack_summary(report_data, hours=args.hours)
        with open(args.slack_summary, "w") as f:
            f.write(slack)
        print(f"Slack summary written to {args.slack_summary}")

    # Write to GitHub Actions step summary if available
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(report)
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
