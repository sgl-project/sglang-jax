#!/bin/bash
# Usage:
#   ./gke-debug.sh start    — 启动调试 pods
#   ./gke-debug.sh stop     — 停止并清理
#   ./gke-debug.sh status   — 查看 pod 状态
#   ./gke-debug.sh ssh [N]  — exec 进入 worker N (默认 0)
#   ./gke-debug.sh setup    — 在所有 worker 上安装 sgl-jax 环境
#   ./gke-debug.sh run CMD  — 在所有 worker 上并行执行命令
#   ./gke-debug.sh serve    — 在所有 worker 上启动 sgl-jax server
#   ./gke-debug.sh logs [N] — 查看 worker N 日志
#
# 前置条件:
#   gcloud container clusters get-credentials tpuv6e-256-node \
#     --region=us-east5 --project=poc-tpu-partner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECTL="/opt/homebrew/share/google-cloud-sdk/bin/kubectl"
JOB_NAME="debug-v6e16"
NUM_WORKERS=4

# ---- 可配置参数 ----
GIT_REMOTE="primatrix"
GIT_REMOTE_URL="https://github.com/primatrix/sglang-jax.git"
GIT_BRANCH="feat/mimo-v2-flash"
MODEL_PATH="/models/MiMo-V2-Flash"

get_pod_name() {
  local index=$1
  $KUBECTL get pods -l "batch.kubernetes.io/job-completion-index=${index},job-name=${JOB_NAME}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

wait_pods_ready() {
  echo "Waiting for all ${NUM_WORKERS} pods to be Running..."
  for i in $(seq 0 $((NUM_WORKERS - 1))); do
    while true; do
      phase=$($KUBECTL get pods -l "batch.kubernetes.io/job-completion-index=${i},job-name=${JOB_NAME}" \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
      if [ "$phase" = "Running" ]; then
        pod=$(get_pod_name $i)
        echo "  Worker $i ($pod): Running"
        break
      fi
      sleep 2
    done
  done
  echo "All pods ready!"
}

case "${1:-help}" in
  start)
    echo "Starting debug pods..."
    $KUBECTL apply -f "${SCRIPT_DIR}/debug-v6e-16.yaml"
    wait_pods_ready
    echo ""
    echo "To enter worker 0:  $0 ssh"
    echo "To setup env:       $0 setup"
    ;;

  stop)
    echo "Stopping debug pods..."
    $KUBECTL delete -f "${SCRIPT_DIR}/debug-v6e-16.yaml" --ignore-not-found
    echo "Done."
    ;;

  status)
    $KUBECTL get pods -l "job-name=${JOB_NAME}" -o wide
    ;;

  ssh)
    worker=${2:-0}
    pod=$(get_pod_name $worker)
    if [ -z "$pod" ]; then
      echo "Error: Worker $worker pod not found. Run '$0 status' to check."
      exit 1
    fi
    echo "Entering worker $worker ($pod)..."
    exec $KUBECTL exec -it "$pod" -- bash
    ;;

  setup)
    echo "Setting up sgl-jax environment on all workers..."
    echo "Branch: ${GIT_REMOTE}/${GIT_BRANCH}"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
      pod=$(get_pod_name $i)
      echo "=== Setting up worker $i ($pod) ==="
      $KUBECTL exec "$pod" -- bash -c "
        set -euxo pipefail
        export PATH=\"\$HOME/.local/bin:\$PATH\"

        # Clone or update repo
        if [ ! -d /workspace/sgl-jax/.git ]; then
          mkdir -p /workspace
          git clone ${GIT_REMOTE_URL} /workspace/sgl-jax
        fi

        cd /workspace/sgl-jax

        # Fetch and checkout our branch
        git remote add ${GIT_REMOTE} ${GIT_REMOTE_URL} 2>/dev/null || true
        git fetch ${GIT_REMOTE} ${GIT_BRANCH}
        git checkout ${GIT_REMOTE}/${GIT_BRANCH} -- python/

        # Create venv and install
        if [ ! -f .venv/bin/activate ]; then
          uv venv --python 3.12
        fi
        source .venv/bin/activate
        uv pip install -e 'python[all,multimodal]'

        echo '=== Worker setup complete ==='
      " &
    done
    wait
    echo "All workers setup complete!"
    ;;

  serve)
    HEAD_ADDR="debug-v6e16-0.debug-v6e16-svc:10011"
    echo "Starting sgl-jax server on all workers..."
    echo "Model: ${MODEL_PATH}"
    echo "Head addr: ${HEAD_ADDR}"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
      pod=$(get_pod_name $i)
      echo "=== Starting server on worker $i ($pod) ==="
      $KUBECTL exec "$pod" -- bash -c "
        set -euo pipefail
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd /workspace/sgl-jax
        source .venv/bin/activate

        python -u -m sgl_jax.launch_server \
          --model-path ${MODEL_PATH} \
          --trust-remote-code \
          --tp-size 16 --ep-size 16 \
          --moe-backend epmoe \
          --nnodes ${NUM_WORKERS} --node-rank ${i} \
          --dist-init-addr ${HEAD_ADDR} \
          --host 0.0.0.0 --port 30271 \
          --page-size 128 \
          --context-length 16384 \
          --disable-radix-cache \
          --chunked-prefill-size 2048 \
          --dtype bfloat16 \
          --mem-fraction-static 0.68 \
          --disable-precompile --skip-server-warmup \
          --log-level info
      " &
    done
    echo ""
    echo "Server starting in background on all workers."
    echo "Use '$0 logs [N]' to check progress."
    echo "Press Ctrl+C to stop all."
    wait
    ;;

  run)
    shift
    CMD="$*"
    echo "Running on all workers: $CMD"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
      pod=$(get_pod_name $i)
      echo "--- Worker $i ($pod) ---"
      $KUBECTL exec "$pod" -- bash -c "$CMD" &
    done
    wait
    ;;

  logs)
    worker=${2:-0}
    pod=$(get_pod_name $worker)
    $KUBECTL logs "$pod" -f
    ;;

  help|*)
    echo "Usage: $0 {start|stop|status|ssh [N]|setup|serve|run CMD|logs [N]}"
    echo ""
    echo "  start   - Create debug pods (sleep infinity)"
    echo "  stop    - Delete debug pods"
    echo "  status  - Show pod status"
    echo "  ssh [N] - Exec into worker N (default: 0)"
    echo "  setup   - Install sgl-jax env on all workers (branch: ${GIT_BRANCH})"
    echo "  serve   - Launch sgl-jax server on all workers"
    echo "  run CMD - Run command on all workers in parallel"
    echo "  logs [N]- Stream logs from worker N (default: 0)"
    ;;
esac
