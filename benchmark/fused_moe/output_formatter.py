"""Output formatting for benchmark results (CSV and Markdown)."""

import csv
import io
from typing import List

from benchmark.fused_moe.benchmark_runner import BenchmarkResult


def format_as_csv(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as CSV.

    CSV Schema:
        implementation,scenario,num_tokens,ep_size,tp_size,num_experts,
        num_experts_per_tok,latency_mean_ms,latency_std_ms,latency_p50_ms,
        latency_p95_ms,latency_p99_ms,latency_min_ms,latency_max_ms,
        max_load,min_load,avg_load,max_imbalance,throughput_tok_per_sec

    Args:
        results: List of benchmark results

    Returns:
        CSV formatted string
    """
    header = [
        "implementation",
        "scenario",
        "num_tokens",
        "ep_size",
        "tp_size",
        "num_experts",
        "num_experts_per_tok",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_min_ms",
        "latency_max_ms",
        "max_load",
        "min_load",
        "avg_load",
        "max_imbalance",
        "throughput_tok_per_sec",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                r.implementation,
                r.scenario,
                r.num_tokens,
                r.ep_size,
                r.tp_size,
                r.num_experts,
                r.num_experts_per_tok,
                f"{r.latency_mean:.4f}",
                f"{r.latency_std:.4f}",
                f"{r.latency_p50:.4f}",
                f"{r.latency_p95:.4f}",
                f"{r.latency_p99:.4f}",
                f"{r.latency_min:.4f}",
                f"{r.latency_max:.4f}",
                r.max_load,
                r.min_load,
                f"{r.avg_load:.2f}",
                f"{r.max_imbalance:.2f}",
                f"{r.throughput:.2f}",
            ]
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


def format_as_markdown(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as Markdown table.

    Groups by scenario and num_tokens, shows side-by-side comparison.

    Args:
        results: List of benchmark results

    Returns:
        Markdown formatted string
    """
    if not results:
        return "# MoE Benchmark Results\n\nNo results to display.\n"

    # Group results by (scenario, num_tokens)
    grouped = {}
    for r in results:
        key = (r.scenario, r.num_tokens)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.implementation] = r

    lines = []
    lines.append("# MoE Benchmark Results\n")

    # Add configuration info from first result
    first_result = results[0]
    lines.append(
        f"**Configuration:** {first_result.num_experts} experts, "
        f"top-{first_result.num_experts_per_tok}, "
        f"EP={first_result.ep_size}, TP={first_result.tp_size}\n"
    )

    # Create tables for each scenario
    for (scenario, num_tokens), impls in sorted(grouped.items()):
        lines.append(f"\n## Scenario: {scenario}, Tokens: {num_tokens}\n")

        # Table header
        lines.append("| Metric | Fused MoE | EP MoE | Speedup |")
        lines.append("|--------|-----------|--------|---------|")

        fused = impls.get("fused")
        epmoe = impls.get("epmoe")

        if fused and epmoe:
            speedup = epmoe.latency_mean / fused.latency_mean

            lines.append(
                f"| Mean Latency (ms) | {fused.latency_mean:.4f} | "
                f"{epmoe.latency_mean:.4f} | {speedup:.2f}x |"
            )
            lines.append(
                f"| P95 Latency (ms) | {fused.latency_p95:.4f} | " f"{epmoe.latency_p95:.4f} | - |"
            )
            lines.append(
                f"| P99 Latency (ms) | {fused.latency_p99:.4f} | " f"{epmoe.latency_p99:.4f} | - |"
            )
            lines.append(
                f"| Throughput (tok/s) | {fused.throughput:.2f} | " f"{epmoe.throughput:.2f} | - |"
            )
            lines.append(
                f"| Max Imbalance | {fused.max_imbalance:.2f}x | "
                f"{epmoe.max_imbalance:.2f}x | - |"
            )
        elif fused:
            lines.append(f"| Mean Latency (ms) | {fused.latency_mean:.4f} | N/A | - |")
            lines.append(f"| P95 Latency (ms) | {fused.latency_p95:.4f} | N/A | - |")
            lines.append(f"| Throughput (tok/s) | {fused.throughput:.2f} | N/A | - |")
            lines.append(f"| Max Imbalance | {fused.max_imbalance:.2f}x | N/A | - |")
        elif epmoe:
            lines.append(f"| Mean Latency (ms) | N/A | {epmoe.latency_mean:.4f} | - |")
            lines.append(f"| P95 Latency (ms) | N/A | {epmoe.latency_p95:.4f} | - |")
            lines.append(f"| Throughput (tok/s) | N/A | {epmoe.throughput:.2f} | - |")
            lines.append(f"| Max Imbalance | N/A | {epmoe.max_imbalance:.2f}x | - |")

    return "\n".join(lines)


def save_results(
    results: List[BenchmarkResult],
    output_file: str,
    output_format: str = "both",
):
    """
    Save benchmark results to files.

    Args:
        results: List of benchmark results
        output_file: Base output file path (without extension)
        output_format: "csv", "markdown", or "both"
    """
    if output_format in ("csv", "both"):
        csv_content = format_as_csv(results)
        csv_path = f"{output_file}.csv"
        with open(csv_path, "w") as f:
            f.write(csv_content)
        print(f"CSV results saved to {csv_path}")

    if output_format in ("markdown", "both"):
        md_content = format_as_markdown(results)
        md_path = f"{output_file}.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Markdown results saved to {md_path}")
