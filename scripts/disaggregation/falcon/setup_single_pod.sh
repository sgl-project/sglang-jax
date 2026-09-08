#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export TMPDIR=/tmp/tpu_logs/tmp HF_HOME=/tmp/tpu_logs/huggingface
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export ALLOW_MULTIPLE_LIBTPU_LOAD=true
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 TPU_PROCESS_BOUNDS=1,1,1
unset TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS TPU_MESH_CONTROLLER_ADDRESS TPU_MESH_CONTROLLER_PORT
export NO_PROXY='*' no_proxy='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PD_OUT=/tmp/tpu_logs/pd-results
export MODEL_PATH=/models/Qwen3-30B-A3B-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
mkdir -p "$TMPDIR" "$HF_HOME" "$PD_OUT"
cleanup() {
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then tail -n 80 "$PD_OUT"/*.log 2>/dev/null || true; fi
  printf '%s\n' "$rc" > "$PD_OUT/runner-exit-code.txt"
  mkdir -p "$ARTIFACT_LOCAL_DIR/pd-results"
  cp -a "$PD_OUT/." "$ARTIFACT_LOCAL_DIR/pd-results/"
  exit "$rc"
}
trap cleanup EXIT
cd /workspace/sglang-jax
cp /tmp/pd-payload/manifest.json "$PD_OUT/source-manifest.json"
cp /tmp/pd-payload/source.patch "$PD_OUT/source.patch"
python3 -m pip install -q torch torchvision 'torchcodec>=0.16.0' --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q -e 'python[tpu]' 'jax==0.11.1' 'jaxlib==0.11.1' 'flax==0.12.9' 'libtpu==0.0.46.1' 'tpu-info==0.7.1' 'protobuf>=6.33.5' imageio-ffmpeg decord
WHEEL=/raiden-wheel/tpu_raiden_jax-0.0.1.dev20260907100311-cp312-cp312-manylinux_2_31_x86_64.whl
(cd /raiden-wheel && sha256sum -c SHA256SUMS)
printf '%s  %s\n' f2650941afb6a41ac86bfde1ceb2af8216586b63f98a4d4a71369000d88882cc "$WHEEL" | sha256sum -c -
cp /raiden-wheel/SHA256SUMS "$PD_OUT/wheel-SHA256SUMS"
for item in /raiden-wheel/*manifest*.json; do [ ! -f "$item" ] || cp "$item" "$PD_OUT/"; done
python3 -m pip install --no-deps --force-reinstall "$WHEEL"
python3 -m pip freeze > "$PD_OUT/environment.txt"
python3 - <<'PY'
from importlib.metadata import version
for package, expected in [('jax', '0.11.1'), ('jaxlib', '0.11.1'), ('libtpu', '0.0.46.1'), ('flax', '0.12.9')]:
    assert version(package) == expected, (package, version(package))
from sgl_jax.raiden import preload_raiden
preload_raiden()
PY
mkdir -p /tmp/pd-fault-hooks
cp scripts/disaggregation/falcon/fault_sitecustomize.py /tmp/pd-fault-hooks/sitecustomize.py
python3 scripts/disaggregation/falcon/session_supervisor.py
