#!/usr/bin/env bash
# profile_export.sh — 运行 profiling (可选)、打包、上传 GCS
#
# 用法:
#   # 仅打包并上传已有的 profile 目录
#   bash blogs/tools/profile_export.sh ./profile_test_moe
#
#   # 运行 profiling + 打包 + 上传 (传额外参数给 bench_fused_moe)
#   bash blogs/tools/profile_export.sh --run --num-experts 256 --top-k 8 \
#       --hidden-size 5120 --intermediate-size 2048 --num-tokens 256 --iters 3
#
set -euo pipefail

GCS_MOUNT="/inference-models/profile"
GCS_BUCKET="gs://inference-model-storage-sgl/profile"

if [[ "${1:-}" == "--run" ]]; then
    shift
    # 用时间戳生成唯一目录名
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PROFILE_DIR="./profile_moe_${TIMESTAMP}"

    echo "==> Running profiling, output: ${PROFILE_DIR}"
    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
        python -m benchmark.moe.bench_fused_moe \
        --profile \
        --profile-dir "${PROFILE_DIR}" \
        "$@"
else
    PROFILE_DIR="${1:?Usage: $0 <profile-dir> | $0 --run [bench args...]}"
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
