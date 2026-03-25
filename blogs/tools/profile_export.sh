#!/usr/bin/env bash
# profile_export.sh — 运行 profiling (可选)、打包、上传 GCS
#
# 用法:
#   # 仅打包并上传已有的 profile 目录
#   bash blogs/tools/profile_export.sh ./profile_test_moe
#
#   # 运行 sglang-jax kernel profiling + 打包 + 上传
#   bash blogs/tools/profile_export.sh --run --num-experts 256 --top-k 8 \
#       --hidden-size 5120 --intermediate-size 2048 --num-tokens 256 --iters 3
#
#   # 运行 tpu-inference kernel profiling + 打包 + 上传
#   bash blogs/tools/profile_export.sh --run-tpu-inference --num-experts 256 --top-k 8 \
#       --hidden-size 8192 --intermediate-size 2048 --scoring-fn sigmoid --num-tokens 256 --iters 3
#
set -euo pipefail

GCS_MOUNT="/inference-models/profile"
GCS_BUCKET="gs://inference-model-storage-sgl/profile"

if [[ "${1:-}" == "--run" ]]; then
    shift
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PROFILE_DIR="./profile_moe_${TIMESTAMP}"

    echo "==> Running sglang-jax kernel profiling, output: ${PROFILE_DIR}"
    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
        python -m benchmark.moe.bench_fused_moe \
        --profile \
        --profile-dir "${PROFILE_DIR}" \
        "$@"

elif [[ "${1:-}" == "--run-tpu-inference" ]]; then
    shift
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PROFILE_DIR="./profile_tpu_inf_moe_${TIMESTAMP}"

    echo "==> Running tpu-inference kernel profiling, output: ${PROFILE_DIR}"
    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
        python -m benchmark.moe.bench_tpu_inference_fused_moe \
        --profile \
        --profile-dir "${PROFILE_DIR}" \
        "$@"

else
    PROFILE_DIR="${1:?Usage: $0 <profile-dir> | $0 --run [bench args...] | $0 --run-tpu-inference [bench args...]}"
fi

if [[ ! -d "${PROFILE_DIR}" ]]; then
    echo "ERROR: ${PROFILE_DIR} does not exist"
    exit 1
fi

DIRNAME=$(basename "${PROFILE_DIR}")
TARBALL="${DIRNAME}.tar.gz"

echo "==> Compressing ${PROFILE_DIR} -> ${TARBALL}"
tar czf "${TARBALL}" "${PROFILE_DIR}"
du -sh "${TARBALL}"

echo "==> Copying to GCS mount: ${GCS_MOUNT}/${TARBALL}"
mkdir -p "${GCS_MOUNT}"
cp "${TARBALL}" "${GCS_MOUNT}/${TARBALL}"

echo ""
echo "=========================================="
echo "  Done! Download from your local machine:"
echo "=========================================="
echo ""
echo "  gcloud storage cp ${GCS_BUCKET}/${TARBALL} ./"
echo "  tar xzf ${TARBALL}"
echo "  xprof --logdir=./${DIRNAME} --port 9001"
echo ""
