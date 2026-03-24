#!/bin/bash
# Setup and run benchmarks on a GKE TPU v7x-8 pod.
#
# Prerequisites:
#   - gcloud, kubectl, xpk installed and authenticated
#   - Cluster 'xpk-cluster' already created (see skill /exec-gke-tpu)
#
# Usage:
#   bash scripts/gke_tpu7x/run_on_pod.sh <POD_NAME> <WORKLOAD_NAME> <script> [args...]
#
# Examples:
#   # Smoke test
#   bash scripts/gke_tpu7x/run_on_pod.sh \
#     test-workload-slice-job-0-0-2spbs test-workload \
#     scripts/gke_tpu7x/smoke_test.py
#
#   # Benchmark fused_moe
#   bash scripts/gke_tpu7x/run_on_pod.sh \
#     test-workload-slice-job-0-0-2spbs test-workload \
#     benchmark/moe/bench_fused_moe.py \
#     --num-experts 8 --top-k 2 --hidden-size 2048 --intermediate-size 512 \
#     --num-tokens 64 128 --iters 3 --warmup-iters 1

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <POD_NAME> <WORKLOAD_NAME> <script> [args...]"
    exit 1
fi

POD="$1"
WORKLOAD="$2"
SCRIPT="$3"
shift 3
ARGS=("$@")

C1="${WORKLOAD}-1"
C2="${WORKLOAD}-2"

echo "=== Syncing code to pod ==="
for C in "$C1" "$C2"; do
    echo "  -> $C"
    kubectl cp scripts/gke_tpu7x/launcher.py "$POD":/tmp/launcher.py -c "$C"
done

echo ""
echo "=== Launching on both containers ==="
echo "  Script: $SCRIPT ${ARGS[*]:-}"
echo ""

# Worker container in background
kubectl exec "$POD" -c "$C2" -- python3 -u /tmp/launcher.py "$SCRIPT" "${ARGS[@]}" 2>&1 &
BGPID=$!

# Main container in foreground
kubectl exec "$POD" -c "$C1" -- python3 -u /tmp/launcher.py "$SCRIPT" "${ARGS[@]}" 2>&1
RC=$?

echo ""
echo "=== Main process exit code: $RC ==="
kill $BGPID 2>/dev/null || true
wait $BGPID 2>/dev/null || true
exit $RC
