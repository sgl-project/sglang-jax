#!/usr/bin/env bash
# Deploy MiMo-V2-Flash SWA PD E2E validation on the poc-tpu-partner GKE cluster.
#
# Default target:
#   project: poc-tpu-partner
#   cluster: tpuv7x-64-node
#   region:  us-central1
#
# Usage:
#   scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh
#   scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh --dry-run
#   scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh --delete-only
#
# Common overrides:
#   BRANCH=epic/mimo-pd-disggragation \
#   GSM_Q=128 GSM_PAR=64 \
#   BENCH_INPUT_LEN=4096 BENCH_OUTPUT_LEN=1024 \
#   scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/mimo_swa_pd_e2e_job.yaml"

PROJECT="${PROJECT:-poc-tpu-partner}"
CLUSTER="${CLUSTER:-tpuv7x-64-node}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-mimo-swa-pd-e2e}"
GENERATED="${GENERATED:-/tmp/${JOB_NAME}.yaml}"

GH_REPO="${GH_REPO:-https://github.com/sgl-project/sglang-jax.git}"
BRANCH="${BRANCH:-epic/mimo-pd-disggragation}"
IMAGE="${IMAGE:-us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-default}"

MODEL_BUCKET="${MODEL_BUCKET:-inference-model-storage-poc-tpu-hns}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2-Flash}"
RAIDEN_CACHE="${RAIDEN_CACHE:-/models/raiden-cache/raiden-v7x-jax0.10.2.tar.gz}"

TPU_ACCELERATOR="${TPU_ACCELERATOR:-tpu7x}"
TPU_TOPOLOGY="${TPU_TOPOLOGY:-2x2x1}"
TPU_CHIPS="${TPU_CHIPS:-4}"
EPHEMERAL_STORAGE="${EPHEMERAL_STORAGE:-30Gi}"

CAP="${CAP:-16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
ROUTER_MAX_CONCURRENT_REQUESTS="${ROUTER_MAX_CONCURRENT_REQUESTS:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.2}"
PRECOMPILE_BS_PADDINGS="${PRECOMPILE_BS_PADDINGS:-1 4 8 16 32 64}"
PRECOMPILE_TOKEN_PADDINGS="${PRECOMPILE_TOKEN_PADDINGS:-512 1024 2048 4096}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM_Q="${GSM_Q:-128}"
GSM_PAR="${GSM_PAR:-64}"
GSM_MAXTOK="${GSM_MAXTOK:-512}"
GSM_MIN_ACC="${GSM_MIN_ACC:-0.60}"
GSM_TIMEOUT_SECONDS="${GSM_TIMEOUT_SECONDS:-3600}"

RUN_LONG_BENCH="${RUN_LONG_BENCH:-1}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-4096}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-4}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-2}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-0}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS:-7200}"
KEEPALIVE_SECONDS="${KEEPALIVE_SECONDS:-3600}"

usage() {
  sed -n '1,28p' "$0" | sed 's/^# \{0,1\}//'
}

cleanup() {
  kubectl delete job "$JOB_NAME" --ignore-not-found=true
  kubectl delete svc "${JOB_NAME}-headless-svc" --ignore-not-found=true
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

render() {
  sed \
    -e "s|<JOB_NAME>|$(escape_sed "$JOB_NAME")|g" \
    -e "s|<GH_REPO>|$(escape_sed "$GH_REPO")|g" \
    -e "s|<BRANCH>|$(escape_sed "$BRANCH")|g" \
    -e "s|<IMAGE>|$(escape_sed "$IMAGE")|g" \
    -e "s|<SERVICE_ACCOUNT>|$(escape_sed "$SERVICE_ACCOUNT")|g" \
    -e "s|<MODEL_BUCKET>|$(escape_sed "$MODEL_BUCKET")|g" \
    -e "s|<MODEL_PATH>|$(escape_sed "$MODEL_PATH")|g" \
    -e "s|<RAIDEN_CACHE>|$(escape_sed "$RAIDEN_CACHE")|g" \
    -e "s|<TPU_ACCELERATOR>|$(escape_sed "$TPU_ACCELERATOR")|g" \
    -e "s|<TPU_TOPOLOGY>|$(escape_sed "$TPU_TOPOLOGY")|g" \
    -e "s|<TPU_CHIPS>|$(escape_sed "$TPU_CHIPS")|g" \
    -e "s|<EPHEMERAL_STORAGE>|$(escape_sed "$EPHEMERAL_STORAGE")|g" \
    -e "s|<CAP>|$(escape_sed "$CAP")|g" \
    -e "s|<MAX_RUNNING_REQUESTS>|$(escape_sed "$MAX_RUNNING_REQUESTS")|g" \
    -e "s|<ROUTER_MAX_CONCURRENT_REQUESTS>|$(escape_sed "$ROUTER_MAX_CONCURRENT_REQUESTS")|g" \
    -e "s|<CONTEXT_LENGTH>|$(escape_sed "$CONTEXT_LENGTH")|g" \
    -e "s|<CHUNKED_PREFILL_SIZE>|$(escape_sed "$CHUNKED_PREFILL_SIZE")|g" \
    -e "s|<MAX_PREFILL_TOKENS>|$(escape_sed "$MAX_PREFILL_TOKENS")|g" \
    -e "s|<MEM_FRACTION_STATIC>|$(escape_sed "$MEM_FRACTION_STATIC")|g" \
    -e "s|<SWA_FULL_TOKENS_RATIO>|$(escape_sed "$SWA_FULL_TOKENS_RATIO")|g" \
    -e "s|<PRECOMPILE_BS_PADDINGS>|$(escape_sed "$PRECOMPILE_BS_PADDINGS")|g" \
    -e "s|<PRECOMPILE_TOKEN_PADDINGS>|$(escape_sed "$PRECOMPILE_TOKEN_PADDINGS")|g" \
    -e "s|<RUN_GSM8K>|$(escape_sed "$RUN_GSM8K")|g" \
    -e "s|<GSM_Q>|$(escape_sed "$GSM_Q")|g" \
    -e "s|<GSM_PAR>|$(escape_sed "$GSM_PAR")|g" \
    -e "s|<GSM_MAXTOK>|$(escape_sed "$GSM_MAXTOK")|g" \
    -e "s|<GSM_MIN_ACC>|$(escape_sed "$GSM_MIN_ACC")|g" \
    -e "s|<GSM_TIMEOUT_SECONDS>|$(escape_sed "$GSM_TIMEOUT_SECONDS")|g" \
    -e "s|<RUN_LONG_BENCH>|$(escape_sed "$RUN_LONG_BENCH")|g" \
    -e "s|<BENCH_INPUT_LEN>|$(escape_sed "$BENCH_INPUT_LEN")|g" \
    -e "s|<BENCH_OUTPUT_LEN>|$(escape_sed "$BENCH_OUTPUT_LEN")|g" \
    -e "s|<BENCH_NUM_PROMPTS>|$(escape_sed "$BENCH_NUM_PROMPTS")|g" \
    -e "s|<BENCH_CONCURRENCY>|$(escape_sed "$BENCH_CONCURRENCY")|g" \
    -e "s|<BENCH_WARMUP_REQUESTS>|$(escape_sed "$BENCH_WARMUP_REQUESTS")|g" \
    -e "s|<BENCH_TIMEOUT_SECONDS>|$(escape_sed "$BENCH_TIMEOUT_SECONDS")|g" \
    -e "s|<KEEPALIVE_SECONDS>|$(escape_sed "$KEEPALIVE_SECONDS")|g" \
    "$TEMPLATE" > "$GENERATED"
}

MODE="${1:-apply}"
case "$MODE" in
  apply|--apply) ;;
  --dry-run) ;;
  --delete-only) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

export USE_GKE_GCLOUD_AUTH_PLUGIN=True
gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$PROJECT"

render
echo "Rendered manifest: $GENERATED"

if [[ "$MODE" == "--delete-only" ]]; then
  cleanup
  exit 0
fi

if [[ "$MODE" == "--dry-run" ]]; then
  kubectl apply --dry-run=client --validate=false -f "$GENERATED"
  exit 0
fi

cleanup
kubectl apply --validate=false -f "$GENERATED"

cat <<EOF

Submitted ${JOB_NAME}.

Monitor:
  kubectl get pods -l job-name=${JOB_NAME} -w
  kubectl logs ${JOB_NAME}-0 -c jax-tpu -f
  kubectl logs ${JOB_NAME}-1 -c jax-tpu -f

Fetch logs after the driver starts:
  kubectl logs ${JOB_NAME}-0 -c jax-tpu > /tmp/${JOB_NAME}-prefill.log
  kubectl logs ${JOB_NAME}-1 -c jax-tpu > /tmp/${JOB_NAME}-decode.log

Cleanup:
  JOB_NAME=${JOB_NAME} scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh --delete-only
EOF
