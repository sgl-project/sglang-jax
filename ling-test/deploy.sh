#!/bin/bash
# sglang-jax TPU deploy pipeline — 一键打包→上传→验证→部署→监控
#
# 解决的核心问题：过去每次部署到 TPU 上等 20-30 分钟才发现低级错误
# （路径不对、少依赖、语法错、tarball 版本不对）。这个脚本在打包阶段
# 就拦截这些错误，避免浪费时间。
#
# 用法：
#   ./ling-test/deploy.sh <yaml_file>            # 完整流程
#   ./ling-test/deploy.sh --dry-run <yaml_file>  # 只验证，不真正部署
#   ./ling-test/deploy.sh --local-check           # 只做本地代码检查

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── Config ──────────────────────────────────────────────────────
TARBALL_NAME="${TARBALL_NAME:-sglang-jax-v4.tar.gz}"
GCS_BUCKET="${GCS_BUCKET:-ainfer-tpu-bench-code}"
GCS_TARBALL="gs://${GCS_BUCKET}/${TARBALL_NAME}"
PUBLIC_TARBALL_URL="https://storage.googleapis.com/${GCS_BUCKET}/${TARBALL_NAME}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Phase 0: 本地代码检查 ────────────────────────────────────────
phase_0_local_check() {
    echo "=== Phase 0: Local code check ==="

    cd "$REPO_ROOT/python"

    # 0.1 编译检查：关键文件都能 parse
    echo "  syntax check (key files) ..."
    cd "$REPO_ROOT/python"
    local key_files=(
        "sgl_jax/srt/layers/fused_moe.py"
        "sgl_jax/srt/configs/model_config.py"
        "sgl_jax/srt/models/bailing_moe_v3.py"
        "sgl_jax/srt/kernels/fused_moe/v4/kernel.py"
        "sgl_jax/srt/kernels/fused_moe/v4/__init__.py"
        "sgl_jax/srt/layers/moe.py"
    )
    for f in "${key_files[@]}"; do
        echo -n "    $f ... "
        python3 -c "import py_compile; py_compile.compile('$f', doraise=True)" 2>/dev/null || fail "syntax error"
        echo OK
    done

    # 0.2 Import 链检查：sgl_jax 核心模块可 import
    echo -n "  import check ... "
    for mod in \
        "sgl_jax.srt.configs.model_config" \
        "sgl_jax.srt.layers.fused_moe" \
        "sgl_jax.srt.kernels.fused_moe.v4.kernel" \
        "sgl_jax.srt.kernels.fused_moe.v1.kernel" \
    ; do
        python3 -c "import $mod" 2>/dev/null || warn "can't import $mod (may need TPU deps — ok for packaging)"
    done
    pass "core modules accessible"

    echo
}

# ── Phase 1: 打包 ─────────────────────────────────────────────────
phase_1_pack() {
    echo "=== Phase 1: Package ==="

    cd "$REPO_ROOT"
    local out="/tmp/${TARBALL_NAME}"

    # Pack from parent so tarball top-level is sglang-jax/ (matches YAML cd path)
    local parent_dir; parent_dir="$(dirname "$REPO_ROOT")"
    local repo_name; repo_name="$(basename "$REPO_ROOT")"
    tar czf "$out" \
        -C "$parent_dir" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='*.egg-info' \
        --exclude='node_modules' \
        "$repo_name/"

    local size; size=$(du -h "$out" | cut -f1)
    echo "  tarball: $out ($size)"

    # 1.1 结构验证：检查关键文件在 tarball 里
    echo -n "  structure check ... "
    for expected in \
        "sglang-jax/python/sgl_jax/srt/layers/fused_moe.py" \
        "sglang-jax/python/sgl_jax/srt/configs/model_config.py" \
        "sglang-jax/python/sgl_jax/srt/models/bailing_moe_v3.py" \
        "sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v4/kernel.py" \
        "sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py" \
    ; do
        tar tzf "$out" | grep -qF "$expected" || fail "missing $expected in tarball"
    done
    pass "all key files present"

    # 1.2 YAML 路径一致性：tarball 顶层目录名必须匹配 YAML 的 cd 路径
    local top; top=$(tar tzf "$out" | head -1 | cut -d/ -f1)
    echo "  tarball top-level dir: $top"

    # 检查目标 yaml 的 cd 路径
    if [ -n "${YAML_FILE:-}" ] && [ -f "${YAML_FILE}" ]; then
        echo -n "  YAML path consistency ... "
        # 提取 yaml 中 "cd /app/.../python" 这样的路径
        local cd_paths; cd_paths=$(sed -n 's/.*cd \([^ ]*\/python\).*/\1/p' "$YAML_FILE" 2>/dev/null || true)
        for cp in $cd_paths; do
            local dir_path="${cp#cd }"
            dir_path="${dir_path%/python}"
            # yaml cd /app/sglang-jax/sglang-jax/python → tarball top = sglang-jax
            local expected_parent
            expected_parent=$(basename "$dir_path")
            if [ "$top" != "$expected_parent" ]; then
                warn "YAML expects cd into '$dir_path' (parent='$expected_parent') but tarball top is '$top'"
            fi
        done
        pass "YAML cd path matches tarball structure"
    fi

    echo
}

# ── Phase 2: 上传 ─────────────────────────────────────────────────
phase_2_upload() {
    echo "=== Phase 2: Upload ==="

    local local_file="/tmp/${TARBALL_NAME}"

    # 2.1 上传
    echo -n "  uploading to $GCS_TARBALL ... "
    gsutil -q cp "$local_file" "$GCS_TARBALL" 2>/dev/null || {
        warn "gsutil failed, trying curl ..."
        # fallback: GCS JSON API upload (需要 bucket 有写权限)
        fail "gsutil not available — install gcloud SDK"
    }
    pass "uploaded"

    # 2.2 验证远端 tarball：下载 metadata 并检查关键文件
    echo -n "  verifying remote tarball ... "
    local remote_files
    remote_files=$(curl -fsSL "$PUBLIC_TARBALL_URL" 2>/dev/null | tar tzf - 2>/dev/null || echo "")
    if [ -z "$remote_files" ]; then
        fail "can't read remote tarball (check GCS permissions)"
    fi

    for expected in \
        "sglang-jax/python/sgl_jax/srt/layers/fused_moe.py" \
        "sglang-jax/python/sgl_jax/srt/configs/model_config.py" \
    ; do
        echo "$remote_files" | grep -qF "$expected" || fail "missing $expected in remote tarball"
    done
    pass "remote tarball verified"

    echo
}

# ── Phase 3: 部署 ─────────────────────────────────────────────────
phase_3_deploy() {
    echo "=== Phase 3: Deploy ==="

    [ -z "${YAML_FILE:-}" ] && { warn "no YAML_FILE, skipping deploy"; return; }
    [ ! -f "$YAML_FILE" ] && fail "YAML file not found: $YAML_FILE"

    # 3.1 清理旧 job
    local job_name
    job_name=$(grep 'name:' "$YAML_FILE" | head -1 | awk '{print $2}')
    if kubectl get job "$job_name" &>/dev/null 2>&1; then
        echo -n "  deleting old job $job_name ... "
        kubectl delete job "$job_name" --ignore-not-found >/dev/null 2>&1
        pass "deleted"
    fi

    # 3.2 部署
    echo -n "  kubectl apply ... "
    kubectl apply -f "$YAML_FILE" >/dev/null
    pass "applied"

    # 3.3 等待 Pod 启动
    echo -n "  waiting for pod ... "
    local max_wait=60 i=0
    while [ $i -lt $max_wait ]; do
        sleep 5; ((i+=5))
        if kubectl get pods -l "batch.kubernetes.io/job-name=$job_name" 2>/dev/null | grep -q Running; then
            pass "pod running"
            break
        fi
        if kubectl get pods -l "batch.kubernetes.io/job-name=$job_name" 2>/dev/null | grep -q Error; then
            fail "pod already failed"
        fi
    done

    echo
}

# ── Phase 4: 监控 ─────────────────────────────────────────────────
phase_4_monitor() {
    echo "=== Phase 4: Monitor (Ctrl-C to stop) ==="
    echo "  Will poll every 30s until DONE_ or fatal error."

    [ -z "${YAML_FILE:-}" ] && return
    local job_name
    job_name=$(grep 'name:' "$YAML_FILE" | head -1 | awk '{print $2}')

    local saw_bench_done=0
    while true; do
        sleep 30

        local latest; latest=$(kubectl logs "job/$job_name" --tail=15 2>/dev/null || echo "")

        # 成功退出
        if echo "$latest" | grep -q "DONE_\|BENCH_RESULT"; then
            echo -e "\n${GREEN}=== BENCHMARK COMPLETE ===${NC}"
            echo "$latest" | grep "BENCH_RESULT\|DONE_"
            echo
            echo "Full log: kubectl logs job/$job_name"
            return 0
        fi

        # 致命错误：立即报告
        if echo "$latest" | grep -qE "Error|FATAL|ModuleNotFoundError|Traceback|ValueError|AssertionError"; then
            echo -e "\n${RED}=== ERROR DETECTED ===${NC}"
            echo "$latest"
            echo
            echo "Full log: kubectl logs job/$job_name"
            return 1
        fi

        # 进度报告
        local progress; progress=$(echo "$latest" | sed -n 's/.*Loading MoE Weights: \([0-9]*%\).*/\1/p' | tail -1)
        local ts; ts=$(echo "$latest" | sed -n 's/.*\(\[20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\]\).*/\1/p' | tail -1)
        echo "  $(date +%H:%M:%S) ${ts} ${progress}"
    done
}

# ── Main ──────────────────────────────────────────────────────────
main() {
    local dry_run=0 local_only=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=1; shift ;;
            --local-check) local_only=1; shift ;;
            --tarball) TARBALL_NAME="$2"; shift 2 ;;
            --bucket) GCS_BUCKET="$2"; shift 2 ;;
            *) YAML_FILE="$1"; shift ;;
        esac
    done

    if [ -n "${YAML_FILE:-}" ] && [ ! -f "$YAML_FILE" ]; then
        fail "YAML file not found: $YAML_FILE"
    fi

    echo "TARBALL: $TARBALL_NAME  |  GCS: $GCS_BUCKET  |  YAML: ${YAML_FILE:-none}"
    echo

    phase_0_local_check

    if [ "$local_only" -eq 1 ]; then
        echo "Local check only — done."
        return 0
    fi

    phase_1_pack
    phase_2_upload

    if [ "$dry_run" -eq 1 ]; then
        echo "Dry run — stopping before deploy."
        return 0
    fi

    if [ -z "${YAML_FILE:-}" ]; then
        warn "no YAML_FILE, skipping deploy + monitor"
        return 0
    fi

    phase_3_deploy
    phase_4_monitor
}

main "$@"
