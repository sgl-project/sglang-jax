#!/usr/bin/env bash
# Diagnostic experiment for sgl-project/sglang-jax#1216:
#   FAILED_PRECONDITION crash when Engine B (devices [2,3]) inits in
#   the same process as Engine A (devices [0,1]) on tpu-v6e-4.
#
# Hypothesis under test:
#   Engine B's *own* init path triggers `deserialize_executable` from the
#   persistent compilation cache (entries written by Engine A during its
#   init), and that deserialize call is what corrupts libtpu state.
#
# What this script does:
#   1. Enable JAX persistent compilation cache logging.
#   2. Run only test_01_multi_engine_modes_cache_miss.
#   3. Save the full log, then extract the cache-relevant events partitioned
#      by Engine A vs Engine B phase.
#
# Outputs (under $LOG_DIR, default /tmp):
#   - cache_log.full.log          : full stdout/stderr
#   - cache_log.events.txt        : grep-extracted cache_key / cache_hit /
#                                   cache_miss / deserialize lines
#   - cache_log.summary.txt       : counts + last-event-before-crash

set -uo pipefail

LOG_DIR="${LOG_DIR:-/tmp}"
XLA_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/xla-cache}"
TEST_TARGET="test.srt.rl.test_multi_engines_in_one_process.TestMultiEnginesInOneProcess.test_01_multi_engine_modes_cache_miss"

FULL_LOG="$LOG_DIR/cache_log.full.log"
EVENTS_LOG="$LOG_DIR/cache_log.events.txt"
SUMMARY_LOG="$LOG_DIR/cache_log.summary.txt"

echo "=== Cache-log diagnostic experiment ==="
echo "JAX_COMPILATION_CACHE_DIR : $XLA_CACHE_DIR"
echo "Cache dir size before run : $(du -sh "$XLA_CACHE_DIR" 2>/dev/null || echo 'missing')"
echo "Cache file count before   : $(find "$XLA_CACHE_DIR" -type f 2>/dev/null | wc -l)"
echo "Full log                  : $FULL_LOG"
echo

export JAX_COMPILATION_CACHE_DIR="$XLA_CACHE_DIR"
# Log every cache key, cache hit/miss, and compile event JAX makes.
export JAX_DEBUG_LOG_MODULES="jax._src.compilation_cache,jax._src.cache_key,jax._src.compiler,jax._src.dispatch"
# Surface absl/glog from the C++ side too.
export TF_CPP_MIN_LOG_LEVEL=0
# Keep CI guard on so test_02 stays skipped.
export SGLANG_JAX_IS_IN_CI=true

python -m unittest "$TEST_TARGET" >"$FULL_LOG" 2>&1
RC=$?

echo "Exit code: $RC"
echo "Cache dir size after run  : $(du -sh "$XLA_CACHE_DIR" 2>/dev/null || echo 'missing')"
echo "Cache file count after    : $(find "$XLA_CACHE_DIR" -type f 2>/dev/null | wc -l)"
echo

# Extract cache-related events (case-insensitive on key terms).
grep -nE "cache[_ ]?(hit|miss)|cache_key|deserializ|writing .*compilation|reading .*compilation|Compiling module|Persistent compilation cache" \
    "$FULL_LOG" >"$EVENTS_LOG" || true

# Build a small summary: hit/miss counts, last events before the FAILED_PRECONDITION line.
{
    echo "# Counts"
    echo -n "cache hits     : "; grep -ciE "cache[_ ]?hit" "$EVENTS_LOG" || true
    echo -n "cache misses   : "; grep -ciE "cache[_ ]?miss" "$EVENTS_LOG" || true
    echo -n "deserialize    : "; grep -ciE "deserializ" "$EVENTS_LOG" || true
    echo -n "compile starts : "; grep -ciE "Compiling module" "$EVENTS_LOG" || true
    echo
    echo "# First and last cache event"
    head -1 "$EVENTS_LOG" 2>/dev/null
    tail -1 "$EVENTS_LOG" 2>/dev/null
    echo
    echo "# Last 30 cache events before FAILED_PRECONDITION"
    if grep -q "FAILED_PRECONDITION" "$FULL_LOG"; then
        FAIL_LINE=$(grep -nE "FAILED_PRECONDITION" "$FULL_LOG" | head -1 | cut -d: -f1)
        awk -v limit="$FAIL_LINE" -F: '$1 < limit' "$EVENTS_LOG" | tail -30
    else
        echo "(no FAILED_PRECONDITION found in log — test may have passed or crashed differently)"
    fi
    echo
    echo "# Crash traceback (first 40 lines around FAILED_PRECONDITION)"
    grep -nE "FAILED_PRECONDITION|Traceback|program continuator" "$FULL_LOG" | head -40
} >"$SUMMARY_LOG"

echo "=== Summary ($SUMMARY_LOG) ==="
cat "$SUMMARY_LOG"

exit "$RC"
