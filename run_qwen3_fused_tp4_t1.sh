#!/usr/bin/env bash
set -euo pipefail
cd ~/sky_workdir/sgl-jax
source ~/sky_workdir/sgl-jax/.venv/bin/activate
PORT=30161
LOG=~/sky_workdir/server_qwen3_main_fused_tp4_30161.log
RESP=/tmp/qwen3_main_fused_tp4_30161_resp.json
MODEL=/models/Qwen3-30B-A3B
rm -f "$LOG" "$RESP"
pkill -f "sgl_jax.launch_server.*${PORT}" >/dev/null 2>&1 || true
sleep 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u -m sgl_jax.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 --ep-size 4 --moe-backend fused \
  --enable-single-process \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length 2048 \
  --max-total-tokens 4096 \
  --max-prefill-tokens 512 \
  --mem-fraction-static 0.3 \
  --disable-precompile --skip-server-warmup \
  --log-level info > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
for i in $(seq 1 900); do
  if grep -q "Uvicorn running on" "$LOG"; then echo "READY ${i}s"; break; fi
  if ! kill -0 "$PID" >/dev/null 2>&1; then echo "EXITED_EARLY"; tail -n 240 "$LOG"; exit 1; fi
  sleep 1
done
REQ='{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"Say one short sentence about AI."}],"max_tokens":1,"temperature":0.7}'
set +e
curl --max-time 180 -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" > "$RESP"
RC=$?
set -e
echo "CURL_RC=$RC"; wc -c "$RESP" || true; head -c 800 "$RESP"; echo
if kill -0 "$PID" >/dev/null 2>&1; then echo "SERVER_ALIVE"; else echo "SERVER_DIED"; fi
kill "$PID" >/dev/null 2>&1 || true
sleep 2
grep -n "POST /v1/chat/completions\|Traceback\|ValueError\|TypeError\|Anomalies" "$LOG" | tail -n 80 || true