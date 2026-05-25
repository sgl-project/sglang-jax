#!/usr/bin/env python3
"""Runner Utilization Report — analyzes GitHub Actions job data for runner metrics."""

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
_NON_GPU_HINTS = ("lint", "release", "runner utilization", "summize")


def run_gh_command(args, max_retries=5):
    """Run a gh CLI command with exponential backoff retry on transient failure."""
    for attempt in range(max_retries):
        try:
            return subprocess.run(["gh"] + args, capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            err_lower = str(e.stderr).lower()
            if "rate limit" not in err_lower and any(
                p in err_lower for p in _NON_RETRYABLE_PATTERNS
            ):
                raise
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1}/{max_retries} in {wait:.1f}s: {e.stderr.strip()}")
            time.sleep(wait)
    raise RuntimeError("run_gh_command: exhausted retries without raising")


def _paginated_gh_api(endpoint, params=None, jq_filter=None, stop_fn=None):
    """Generic paginated GitHub API fetcher."""
    page, per_page, all_items = 1, 100, []
    while True:
        args = [
            "api",
            endpoint,
            "--method",
            "GET",
            "-f",
            f"per_page={per_page}",
            "-f",
            f"page={page}",
        ]
        for k, v in (params or {}).items():
            args.extend(["-f", f"{k}={v}"])
        if jq_filter:
            args.extend(["--jq", jq_filter])
        output = run_gh_command(args)
        if not output.strip():
            break
        page_count, stop = 0, False
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            page_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stop_fn and stop_fn(item):
                stop = True
                break
            all_items.append(item)
        if stop or page_count < per_page:
            break
        page += 1
    return all_items


def get_workflow_runs(repo, hours=24):
    """Fetch workflow runs from the past N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return _paginated_gh_api(
        f"/repos/{repo}/actions/runs",
        params={"sort": "created", "direction": "desc"},
        jq_filter=".workflow_runs[]",
        stop_fn=lambda r: (t := parse_time(r.get("created_at"))) is not None and t < cutoff,
    )


def get_jobs_for_run(repo, run_id):
    """Fetch all jobs for a workflow run, including retried attempts."""
    return _paginated_gh_api(
        f"/repos/{repo}/actions/runs/{run_id}/jobs",
        params={"filter": "all"},
        jq_filter=".jobs[]",
    )


def _meaningful_labels(label_dicts):
    """Extract meaningful labels from a list of {"name": ...} dicts."""
    return {d["name"] for d in label_dicts} - DEFAULT_LABELS_TO_IGNORE - GITHUB_HOSTED_LABELS


def get_runners(repo, online_only=False):
    """Fetch self-hosted runners; degrades gracefully without admin access."""
    try:
        runners = _paginated_gh_api(
            f"/repos/{repo}/actions/runners",
            jq_filter=".runners[]",
        )
        if online_only:
            runners = [r for r in runners if r.get("status") == "online"]
        for runner in runners:
            if not _meaningful_labels(runner.get("labels", [])):
                name = runner.get("name", "")
                m = re.match(r"^(arc-runner-v6e-\d+)-", name)
                inferred = (
                    m.group(1)
                    if m
                    else (
                        "arc-runner-cpu"
                        if ("cpu-runner" in name or name.startswith("runnerdeploy-"))
                        else None
                    )
                )
                if inferred:
                    runner.setdefault("labels", []).append({"name": inferred})
        return runners
    except subprocess.CalledProcessError as e:
        err_str = str(e.stderr)
        if "rate limit" in err_str.lower():
            print("Warning: GitHub API rate limit hit; runner count unavailable.")
        elif "403" in err_str or "must have admin" in err_str.lower():
            print("Note: No admin access to list runners; count estimated from job data.")
        else:
            raise
        return []


def parse_time(time_str):
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime."""
    if not time_str:
        return None
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def calculate_concurrency_metrics(jobs, window_start, window_end, num_runners):
    """Sweep-line algorithm for peak/avg concurrent jobs, saturation, queue depth."""
    events = []
    for job in jobs:
        s, e = parse_time(job.get("started_at")), parse_time(job.get("completed_at"))
        if s and e and s < e:
            s, e = max(s, window_start), min(e, window_end)
            if s < e:
                events += [(s, +1), (e, -1)]
    if not events:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_pct": None,
            "peak_queue": None,
        }
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    weighted_sum = 0.0
    prev_time = window_start
    for event_time, delta in events:
        if event_time > prev_time:
            weighted_sum += current * (event_time - prev_time).total_seconds()
            prev_time = event_time
        current += delta
        peak = max(peak, current)
    if prev_time < window_end:
        weighted_sum += current * (window_end - prev_time).total_seconds()
    ws = (window_end - window_start).total_seconds()
    avg = weighted_sum / ws if ws > 0 else 0.0
    return {
        "peak_concurrent": peak,
        "avg_concurrent": avg,
        "saturation_pct": (avg / num_runners * 100) if num_runners > 0 else None,
        "peak_queue": max(0, peak - num_runners) if num_runners > 0 else None,
    }


def _percentile(sorted_values, p):
    """Floor-index percentile: P95 equals max for samples < 20."""
    if not sorted_values:
        return 0.0
    return sorted_values[min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)]


def _format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = seconds / 60
    return f"{m:.1f}m" if m < 60 else f"{m / 60:.1f}h"


def calculate_utilization(repo, hours=24, runner_filter=None):
    """Fetch runs/jobs, aggregate per-label utilization and concurrency metrics."""
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)
    print(f"Fetching workflow runs for {repo} in the past {hours} hours...")
    runs = get_workflow_runs(repo, hours=hours)
    print(f"Found {len(runs)} workflow runs.")

    runners_list = get_runners(repo, online_only=False)

    # Fleet status aggregation
    fleet_status = defaultdict(
        lambda: {"total": 0, "online": 0, "offline": 0, "busy": 0, "idle": 0}
    )
    label_runner_count = defaultdict(int)
    for runner in runners_list:
        meaningful = _meaningful_labels(runner.get("labels", []))
        for lbl in meaningful:
            label_runner_count[lbl] += 1
            fs = fleet_status[lbl]
            fs["total"] += 1
            if runner.get("status") == "online":
                fs["online"] += 1
                fs["busy" if runner.get("busy") else "idle"] += 1
            else:
                fs["offline"] += 1
    fleet_status = dict(fleet_status)

    fetch_failures, total_runs = 0, len(runs)
    label_intervals = defaultdict(list)
    label_jobs = defaultdict(list)
    label_queue_waits = defaultdict(list)
    label_job_durations = defaultdict(list)
    label_conclusions = defaultdict(list)
    failed_jobs = []
    workflow_stats = defaultdict(
        lambda: defaultdict(lambda: {"job_count": 0, "total_duration": 0.0})
    )

    def fetch_run_jobs(run):
        wf_name, run_url = run.get("name", ""), run.get("html_url", "")
        if any(h in wf_name.lower() for h in _NON_GPU_HINTS):
            return run["id"], wf_name, run_url, []
        try:
            return run["id"], wf_name, run_url, get_jobs_for_run(repo, run["id"])
        except Exception as exc:
            print(f"Warning: failed to fetch jobs for run {run['id']}: {exc}")
            return run["id"], wf_name, run_url, None

    print("Fetching job details (parallel, max 4 workers)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_run_jobs, run): run for run in runs}
        for future in as_completed(futures):
            try:
                run_id, wf_name, run_url, jobs = future.result()
            except Exception as exc:
                print(f"Warning: unexpected error fetching jobs: {exc}")
                fetch_failures += 1
                continue
            if jobs is None:
                fetch_failures += 1
                continue
            for job in jobs:
                if job.get("status") != "completed":
                    continue
                meaningful = (
                    set(job.get("labels", [])) - DEFAULT_LABELS_TO_IGNORE - GITHUB_HOSTED_LABELS
                )
                if not meaningful:
                    continue
                if runner_filter and not any(runner_filter in l for l in meaningful):
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
                duration_s = (eff_end - eff_start).total_seconds()
                conclusion = job.get("conclusion", "unknown")
                for lbl in meaningful:
                    label_intervals[lbl].append((eff_start, eff_end))
                    label_jobs[lbl].append(job)
                    label_job_durations[lbl].append(duration_s)
                    label_conclusions[lbl].append(conclusion)
                    workflow_stats[wf_name][lbl]["job_count"] += 1
                    workflow_stats[wf_name][lbl]["total_duration"] += duration_s
                if conclusion == "failure":
                    failed_jobs.append(
                        {
                            "workflow": wf_name,
                            "job_name": job.get("name", ""),
                            "duration": duration_s,
                            "label": ", ".join(sorted(meaningful)),
                            "url": job.get("html_url", run_url),
                        }
                    )

    fetch_failure_pct = (fetch_failures / total_runs * 100) if total_runs > 0 else 0.0
    window_seconds = (window_end - window_start).total_seconds()

    results = {}
    for label in sorted(set(label_intervals) | set(label_runner_count)):
        if runner_filter and runner_filter not in label:
            continue
        intervals = label_intervals.get(label, [])
        num_runners = label_runner_count.get(label, 0)
        total_job_s = sum((e - s).total_seconds() for s, e in intervals)
        cap_s = num_runners * window_seconds if num_runners > 0 else 0.0
        waits = sorted(label_queue_waits.get(label, []))
        durations = sorted(label_job_durations.get(label, []))
        conclusions = label_conclusions.get(label, [])
        results[label] = {
            "num_runners": num_runners,
            "utilization_pct": (total_job_s / cap_s * 100) if cap_s > 0 else None,
            "job_count": len(label_jobs.get(label, [])),
            "concurrency": calculate_concurrency_metrics(
                label_jobs.get(label, []),
                window_start,
                window_end,
                num_runners,
            ),
            "wait_p95": _percentile(waits, 95),
            "duration_p50": _percentile(durations, 50),
            "duration_p95": _percentile(durations, 95),
            "success_count": conclusions.count("success"),
            "failure_count": conclusions.count("failure"),
            "total_concluded": len(conclusions),
        }

    wf_aggregates = {}
    for wf_name, label_data in workflow_stats.items():
        wf_aggregates[wf_name] = {
            "total_jobs": sum(v["job_count"] for v in label_data.values()),
            "total_duration": sum(v["total_duration"] for v in label_data.values()),
            "labels": sorted(label_data.keys()),
        }

    return {
        "per_label": results,
        "fleet_status": fleet_status,
        "fetch_failure_pct": fetch_failure_pct,
        "failed_jobs": failed_jobs,
        "workflow_stats": dict(workflow_stats),
        "wf_aggregates": wf_aggregates,
    }


def _generate_recommendations(results, fleet_status, wf_aggregates):
    """Generate recommendation strings shared by full report and Slack summary."""
    recs = []
    grand_total = sum(a["total_duration"] for a in wf_aggregates.values())
    for label, d in sorted(results.items()):
        util, conc = d["utilization_pct"], d["concurrency"]
        tc, fc = d["total_concluded"], d["failure_count"]
        if util is not None and util > 80:
            recs.append(
                f"**`{label}`**: High utilization ({util:.1f}%). Consider adding more runners."
            )
        elif util is not None and util < 10 and d["job_count"] > 0:
            recs.append(f"**`{label}`**: Low utilization ({util:.1f}%). May be over-provisioned.")
        if conc["peak_queue"] is not None and conc["peak_queue"] > 0:
            recs.append(
                f"**`{label}`**: Peak queue depth {conc['peak_queue']}. Jobs waited for runners."
            )
        if tc > 5 and fc / tc > 0.2:
            recs.append(f"**`{label}`**: High failure rate ({fc}/{tc}, {fc / tc * 100:.1f}%).")
        if d["wait_p95"] > 600:
            recs.append(
                f"**`{label}`**: Long queue wait (P95 = {_format_duration(d['wait_p95'])}). Consider scaling."
            )
    for label, s in sorted(fleet_status.items()):
        if s["offline"] > 0:
            recs.append(f"**`{label}`**: {s['offline']} runner(s) offline.")
    if grand_total > 0:
        for wf, a in sorted(
            wf_aggregates.items(), key=lambda x: x[1]["total_duration"], reverse=True
        ):
            if a["total_duration"] > 0.5 * grand_total:
                pct = a["total_duration"] / grand_total * 100
                recs.append(
                    f"Workflow **`{wf}`** accounts for {pct:.0f}% of runner time "
                    f"({_format_duration(a['total_duration'])} of {_format_duration(grand_total)})."
                )
    return recs


def format_report(report_data, hours):
    """Format utilization results as a Markdown report (5 sections)."""
    results = report_data["per_label"]
    fleet_status = report_data["fleet_status"]
    wf_aggregates = report_data["wf_aggregates"]
    failed_jobs = report_data["failed_jobs"]
    lines = [f"# Runner Utilization Report (Past {hours} Hours)", ""]
    lines.append(f"_Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    if report_data["fetch_failure_pct"] > 0:
        lines.append(
            f"> **Warning:** {report_data['fetch_failure_pct']:.1f}% of runs could not be fetched."
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

    # Utilization Summary
    lines.append("## Utilization Summary")
    lines.append("")
    lines.append(
        "| Runner Label | Runners | Jobs | Utilization | Peak Conc "
        "| Queue P95 | Duration P50 | Duration P95 | Success Rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, d in sorted(results.items()):
        nr = d["num_runners"] or "N/A"
        util = f"{d['utilization_pct']:.1f}%" if d["utilization_pct"] is not None else "N/A"
        tc = d["total_concluded"]
        rate = f"{d['success_count'] / tc * 100:.1f}%" if tc > 0 else "N/A"
        lines.append(
            f"| `{label}` | {nr} | {d['job_count']} | {util} "
            f"| {d['concurrency']['peak_concurrent']} | {_format_duration(d['wait_p95'])} "
            f"| {_format_duration(d['duration_p50'])} | {_format_duration(d['duration_p95'])} | {rate} |"
        )
    lines.append("")

    # Top 10 Workflows
    lines.append("## Top 10 Workflows by Runner Time")
    lines.append("")
    lines.append("| # | Workflow | Jobs | Total Time | Avg Duration | Runner Labels |")
    lines.append("|---|---|---|---|---|---|")
    sorted_wfs = sorted(wf_aggregates.items(), key=lambda x: x[1]["total_duration"], reverse=True)
    for rank, (wf, a) in enumerate(sorted_wfs[:10], start=1):
        avg = a["total_duration"] / a["total_jobs"] if a["total_jobs"] > 0 else 0
        lines.append(
            f"| {rank} | {wf} | {a['total_jobs']} | {_format_duration(a['total_duration'])} "
            f"| {_format_duration(avg)} | {', '.join(a['labels'])} |"
        )
    lines.append("")

    # Failed Jobs (max 5)
    lines.append("## Failed Jobs")
    lines.append("")
    if failed_jobs:
        lines.append("| Workflow | Job | Duration | Runner Label | Link |")
        lines.append("|---|---|---|---|---|")
        for j in sorted(failed_jobs, key=lambda j: j["duration"], reverse=True)[:5]:
            lines.append(
                f"| {j['workflow']} | {j['job_name']} | {_format_duration(j['duration'])} "
                f"| {j['label']} | [Run]({j['url']}) |"
            )
    else:
        lines.append("No failed jobs in this period.")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    recs = _generate_recommendations(results, fleet_status, wf_aggregates)
    for r in recs:
        lines.append(f"- {r}")
    if not recs:
        lines.append("No immediate action required. All runners within normal bounds.")
    lines.append("")
    return "\n".join(lines)


def format_slack_summary(report_data, hours, run_url=""):
    """Format a compact Slack mrkdwn summary."""
    results = report_data["per_label"]
    fleet_status = report_data["fleet_status"]
    wf_aggregates = report_data["wf_aggregates"]
    lines = [f"*Runner Utilization Report (Past {hours} Hours)*"]
    lines.append("")
    lines.append("*Fleet Status*")
    if fleet_status:
        for label, s in sorted(fleet_status.items()):
            lines.append(
                f"• `{label}`: {s['total']} runners ({s['online']} online, {s['busy']} busy)"
            )
    else:
        lines.append("Fleet data unavailable (admin token required)")
    if not results:
        lines += ["", "No runner activity in this period."]
        if run_url:
            lines += ["", f"<{run_url}|View full report>"]
        return "\n".join(lines)

    lines.append("")
    lines.append("*Utilization*")
    lines.append(
        "• "
        + " | ".join(
            (
                f"`{l}`: {d['utilization_pct']:.1f}%"
                if d["utilization_pct"] is not None
                else f"`{l}`: N/A"
            )
            for l, d in sorted(results.items())
        )
    )
    lines.append("")
    lines.append("*Queue Wait (P95)*")
    lines.append(
        "• "
        + " | ".join(
            f"`{l}`: {_format_duration(d['wait_p95'])}" for l, d in sorted(results.items())
        )
    )
    lines.append("")
    lines.append("*Success Rate*")
    rate_parts = []
    for l, d in sorted(results.items()):
        if d["total_concluded"] > 0:
            p = f"`{l}`: {d['success_count'] / d['total_concluded'] * 100:.1f}%"
            if d["failure_count"] > 0:
                p += f" ({d['failure_count']} failures)"
            rate_parts.append(p)
        else:
            rate_parts.append(f"`{l}`: N/A")
    lines.append("• " + " | ".join(rate_parts))

    recs = _generate_recommendations(results, fleet_status, wf_aggregates)
    lines.append("")
    if recs:
        lines.append("*Recommendations*")
        for r in recs:
            lines.append(f"• {r.replace('**`', '*`').replace('`**', '`*')}")
    else:
        lines.append("No issues detected.")
    if run_url:
        lines += ["", f"<{run_url}|View full report>"]
    text = "\n".join(lines)
    # Slack Block Kit section text has a 3000 character limit
    if len(text) > 2900:
        text = text[:2900] + "\n… (truncated)"
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate runner utilization report")
    parser.add_argument("--repo", default="sgl-project/sglang-jax", help="GitHub repository")
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    parser.add_argument("--filter", type=str, help="Filter runner labels (substring match)")
    parser.add_argument("--output", type=str, help="Output file path (default: stdout)")
    parser.add_argument("--slack-output", type=str, help="Slack summary output file path")
    parser.add_argument("--run-url", type=str, default="", help="Actions run URL for Slack link")
    args = parser.parse_args()
    if args.hours < 1:
        parser.error("--hours must be a positive integer")

    report_data = calculate_utilization(args.repo, args.hours, args.filter)
    report = format_report(report_data, args.hours)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    if args.slack_output:
        summary = format_slack_summary(report_data, args.hours, args.run_url)
        with open(args.slack_output, "w") as f:
            f.write(summary)
        print(f"Slack summary written to {args.slack_output}")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(report)
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
