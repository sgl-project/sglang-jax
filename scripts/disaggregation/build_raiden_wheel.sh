#!/usr/bin/env bash
set -euo pipefail

: "${RAIDEN_SRC:?set RAIDEN_SRC to a tpu-sync checkout}"
: "${RAIDEN_CACHE_ROOT:?set RAIDEN_CACHE_ROOT to a persistent cache directory}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH="$SCRIPT_DIR/patches/tpu-sync/jax-0.11.1.patch"
PYTHON_BIN=${PYTHON_BIN:-python3.12}
JAX_VERSION=0.11.1
JAXLIB_VERSION=0.11.1
LIBTPU_VERSION=0.0.46.1
RAIDEN_COMMIT=6d43141191210b73359d22743be619532ad587dc
CACHE_DIR=${RAIDEN_CACHE_ROOT}/${RAIDEN_COMMIT}-jax${JAX_VERSION}-jaxlib${JAXLIB_VERSION}-libtpu${LIBTPU_VERSION}

test "$(git -C "${RAIDEN_SRC}" rev-parse HEAD)" = "${RAIDEN_COMMIT}"
if ! git -C "${RAIDEN_SRC}" apply --reverse --check "$PATCH" 2>/dev/null; then
  git -C "${RAIDEN_SRC}" apply --check "$PATCH"
  git -C "${RAIDEN_SRC}" apply "$PATCH"
fi

if [[ -s "${CACHE_DIR}/READY" && -s "${CACHE_DIR}/SHA256SUMS" ]]; then
  (
    cd "${CACHE_DIR}"
    sha256sum -c SHA256SUMS
  )
  printf 'Using cached Raiden wheel: %s\n' "${CACHE_DIR}/$(<"${CACHE_DIR}/READY")"
  exit 0
fi

grep -q "jax==${JAX_VERSION}" "${RAIDEN_SRC}/requirements.txt"
grep -q "jaxlib==${JAXLIB_VERSION}" "${RAIDEN_SRC}/requirements.txt"
grep -q "libtpu==${LIBTPU_VERSION}" "${RAIDEN_SRC}/requirements.txt"

export HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12}
(
  cd "${RAIDEN_SRC}"
  ./build.sh jax //ci/wheel:raiden_jax_wheel
)

shopt -s nullglob
wheels=("${RAIDEN_SRC}"/bazel-bin/ci/wheel/tpu_raiden_jax-*.whl)
if (( ${#wheels[@]} != 1 )); then
  printf 'Expected one Raiden wheel, found %d\n' "${#wheels[@]}" >&2
  exit 1
fi

wheel=${wheels[0]}
listing=$(${PYTHON_BIN} -m zipfile -l "${wheel}")
grep -q '_tpu_raiden_jax.*\.so' <<<"${listing}"
grep -q '_kv_cache_manager_ffi.*\.so' <<<"${listing}"

mkdir -p "${CACHE_DIR}"
cp "${wheel}" "${CACHE_DIR}/"
(
  cd "${CACHE_DIR}"
  sha256sum "$(basename "${wheel}")" > SHA256SUMS
  sha256sum -c SHA256SUMS
  basename "${wheel}" > READY
)
printf 'Cached Raiden wheel: %s\n' "${CACHE_DIR}/$(<"${CACHE_DIR}/READY")"
