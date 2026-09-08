"""Summarize explicitly selected Falcon PD runs; never select away failed runs."""

import json
import os
from pathlib import Path
import statistics


def summarize(root, runs):
    reports = {}
    for kind, filename in [
        ("correctness", "summary.json"),
        ("stream", "stream-summary.json"),
        ("fault", "fault-summary.json"),
        ("eos", "eos-summary.json"),
        ("performance", "performance-summary.json"),
    ]:
        path = root / "pd-results" / runs[kind] / filename
        reports[kind] = json.loads(path.read_text())
    if "mixed" in runs:
        reports["mixed"] = json.loads(
            (root / "pd-results" / runs["mixed"] / "mixed-summary.json").read_text()
        )
    performance = reports["performance"]
    rows = {}
    for measurement in performance["measurements"]:
        tag = measurement["tag"]
        if tag not in {f"{repeat}-{group}" for repeat in range(3) for group in ("B1", "PD")}:
            continue
        key = (measurement["input"], measurement["output"], measurement["concurrency"])
        rows.setdefault(key, {})[tag] = measurement
    metrics = [
        "output_token_per_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "itl_p50_s",
        "itl_p95_s",
        "e2e_p95_s",
    ]
    comparisons = []
    for key, measurements in sorted(rows.items()):
        if len(measurements) != 6:
            raise ValueError(f"incomplete repeated A/B: {key}")
        item = dict(zip(("input", "output", "concurrency"), key))
        item["metrics"] = {}
        for metric in metrics:
            baseline = [measurements[f"{repeat}-B1"][metric] for repeat in range(3)]
            candidate = [measurements[f"{repeat}-PD"][metric] for repeat in range(3)]
            changes = [(new / old - 1) * 100 for old, new in zip(baseline, candidate)]
            item["metrics"][metric] = {
                "baseline": baseline,
                "candidate": candidate,
                "baseline_median": statistics.median(baseline),
                "candidate_median": statistics.median(candidate),
                "paired_percent_changes": changes,
                "median_percent_change": statistics.median(changes),
                "nonoverlapping_ranges": min(candidate) > max(baseline)
                or max(candidate) < min(baseline),
            }
        comparisons.append(item)
    return {
        "runs": runs,
        "statuses": {k: v["status"] for k, v in reports.items()},
        "comparisons": comparisons,
        "measurement_count": len(performance["measurements"]),
        "profiles": performance["profiles"],
        "ablations": [m for m in performance["measurements"] if m["tag"] in ("P-only", "D-only")],
        "limitations": [
            "Single-pod results do not measure cross-host networking.",
            "Fault injection delays completion visibility, not physical DMA.",
            "Pool peaks are sampled at one-second intervals.",
            "XProf analysis is required before claiming physical transfer overlap.",
        ],
    }


def markdown(summary):
    lines = [
        "# Raiden PD overlap validation",
        "",
        "Selected runs: `" + json.dumps(summary["runs"]) + "`.",
        "",
        "Statuses: `" + json.dumps(summary["statuses"]) + "`.",
        "",
        "Three paired repetitions; percentages below are median paired PD/B1 changes. "
        "Positive throughput is an improvement; positive latency is a regression.",
        "",
        "| Input/output/concurrency | Output tok/s B1 → PD | Throughput Δ | TTFT p95 Δ | ITL p95 Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        m = row["metrics"]
        throughput = m["output_token_per_s"]
        lines.append(
            f'| {row["input"]}/{row["output"]}/{row["concurrency"]} | '
            f'{throughput["baseline_median"]:.1f} → {throughput["candidate_median"]:.1f} | '
            f'{throughput["median_percent_change"]:+.1f}% | '
            f'{m["ttft_p95_s"]["median_percent_change"]:+.1f}% | '
            f'{m["itl_p95_s"]["median_percent_change"]:+.1f}% |'
        )
    lines += ["", "Limitations:", ""] + ["- " + item for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    params = json.loads(os.environ["FALCON_ANALYSIS_PARAMS_JSON"])
    result = summarize(Path(os.environ["ARTIFACT_LOCAL_DIR"]), params["runs"])
    destination = Path(os.environ["RESULT_LOCAL_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2))
    (destination / "report.md").write_text(markdown(result))
