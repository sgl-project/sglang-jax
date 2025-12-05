"""Analyze server logs to extract token count information.

Usage:
    # Analyze from stdin
    cat server.log | python analyze_server_log.py

    # Or analyze from file
    python analyze_server_log.py server.log
"""

import re
import sys
from collections import defaultdict


def parse_prefill_line(line):
    """Parse prefill batch log line."""
    # Example: Prefill batch. #new-seq: 8, #new-token: 65536, #cached-token: 0, token usage: 0.21, #running-req: 8, #queue-req: 8,
    match = re.search(
        r"#new-seq:\s*(\d+).*#new-token:\s*(\d+).*#cached-token:\s*(\d+).*token usage:\s*([\d.]+).*#running-req:\s*(\d+).*#queue-req:\s*(\d+)",
        line,
    )
    if match:
        return {
            "type": "prefill",
            "new_seq": int(match.group(1)),
            "new_token": int(match.group(2)),
            "cached_token": int(match.group(3)),
            "token_usage": float(match.group(4)),
            "running_req": int(match.group(5)),
            "queue_req": int(match.group(6)),
        }
    return None


def parse_decode_line(line):
    """Parse decode batch log line."""
    # Example: Decode batch. #running-req: 8, #token: 286848, token usage: 0.96, gen throughput (token/s): 232.27, #queue-req: 0,
    match = re.search(
        r"#running-req:\s*(\d+).*#token:\s*(\d+).*token usage:\s*([\d.]+).*gen throughput.*:\s*([\d.]+).*#queue-req:\s*(\d+)",
        line,
    )
    if match:
        return {
            "type": "decode",
            "running_req": int(match.group(1)),
            "token": int(match.group(2)),
            "token_usage": float(match.group(3)),
            "gen_throughput": float(match.group(4)),
            "queue_req": int(match.group(5)),
        }
    return None


def main():
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], "r") as f:
            lines = f.readlines()
    else:
        # Read from stdin
        lines = sys.stdin.readlines()

    print("=" * 80)
    print("SERVER LOG ANALYSIS")
    print("=" * 80)

    prefill_logs = []
    decode_logs = []

    for line in lines:
        if "Prefill batch" in line:
            data = parse_prefill_line(line)
            if data:
                prefill_logs.append(data)
        elif "Decode batch" in line:
            data = parse_decode_line(line)
            if data:
                decode_logs.append(data)

    # Analyze prefill logs
    if prefill_logs:
        print(f"\nPREFILL BATCHES: {len(prefill_logs)} found")
        print("-" * 80)
        print(
            f"{'#':>4} {'NewSeq':>8} {'NewTok':>10} {'CacheTok':>10} {'Usage':>8} {'RunReq':>8} {'QueueReq':>10}"
        )
        print("-" * 80)

        total_new_tokens = 0
        for i, log in enumerate(prefill_logs[:20]):  # Show first 20
            print(
                f"{i+1:4d} {log['new_seq']:8d} {log['new_token']:10d} {log['cached_token']:10d} "
                f"{log['token_usage']:8.2f} {log['running_req']:8d} {log['queue_req']:10d}"
            )
            total_new_tokens += log["new_token"]

        if len(prefill_logs) > 20:
            print(f"... and {len(prefill_logs) - 20} more")

        print(f"\nPrefill Statistics:")
        print(f"  Total new tokens across all prefills: {total_new_tokens:,}")
        print(f"  Average tokens per prefill: {total_new_tokens / len(prefill_logs):,.0f}")
        if prefill_logs:
            print(f"  First prefill new tokens: {prefill_logs[0]['new_token']:,}")

    # Analyze decode logs
    if decode_logs:
        print(f"\n{'=' * 80}")
        print(f"DECODE BATCHES: {len(decode_logs)} found")
        print("-" * 80)
        print(
            f"{'#':>4} {'RunReq':>8} {'#Token':>12} {'Usage':>8} {'Throughput':>12} {'QueueReq':>10}"
        )
        print("-" * 80)

        for i, log in enumerate(decode_logs[:20]):  # Show first 20
            print(
                f"{i+1:4d} {log['running_req']:8d} {log['token']:12,d} {log['token_usage']:8.2f} "
                f"{log['gen_throughput']:12.2f} {log['queue_req']:10d}"
            )

        if len(decode_logs) > 20:
            print(f"... and {len(decode_logs) - 20} more")

        print(f"\nDecode Statistics:")
        tokens_by_running_req = defaultdict(list)
        for log in decode_logs:
            tokens_by_running_req[log["running_req"]].append(log["token"])

        for running_req in sorted(tokens_by_running_req.keys()):
            tokens = tokens_by_running_req[running_req]
            avg_token = sum(tokens) / len(tokens)
            min_token = min(tokens)
            max_token = max(tokens)
            print(
                f"  Running {running_req:2d} requests: avg={avg_token:10.0f}, min={min_token:10,d}, max={max_token:10,d} tokens ({len(tokens)} samples)"
            )
            if running_req > 0:
                avg_per_req = avg_token / running_req
                print(f"    → Avg per request: {avg_per_req:,.0f} tokens")

    # Key findings
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print("=" * 80)

    if prefill_logs and decode_logs:
        first_prefill_tokens = prefill_logs[0]["new_token"]
        first_decode_tokens = decode_logs[0]["token"] if decode_logs else 0

        print(f"\n1. First prefill batch:")
        print(f"   New tokens: {first_prefill_tokens:,}")
        print(f"   New sequences: {prefill_logs[0]['new_seq']}")
        print(f"   Average per sequence: {first_prefill_tokens / prefill_logs[0]['new_seq']:,.0f}")

        print(f"\n2. Decode batch token count:")
        print(f"   Token count: {first_decode_tokens:,}")
        print(f"   Running requests: {decode_logs[0]['running_req']}")
        if decode_logs[0]["running_req"] > 0:
            print(
                f"   Average per request: {first_decode_tokens / decode_logs[0]['running_req']:,.0f}"
            )

        print(
            f"\n3. Ratio (decode/prefill): {first_decode_tokens / first_prefill_tokens if first_prefill_tokens > 0 else 0:.2f}x"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
