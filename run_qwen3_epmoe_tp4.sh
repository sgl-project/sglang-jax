#!/usr/bin/env bash
set -euo pipefail
cd ~/sky_workdir/sgl-jax
source ~/sky_workdir/sgl-jax/.venv/bin/activate
PORT=30170
LOG=~/sky_workdir/server_qwen3_epmoe_tp4.log
RESP=/tmp/qwen3_epmoe_tp4_resp.json
MODEL=/models/Qwen3-30B-A3B
rm -f "$LOG" "$RESP"
pkill -f "sgl_jax.launch_server.*${PORT}" >/dev/null 2>&1 || true
sleep 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -u -m sgl_jax.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 --ep-size 4 --moe-backend epmoe \
  --enable-single-process \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code \
  --context-length 2048 \
  --max-total-tokens 4096 \
  --max-prefill-tokens 512 \
  --mem-fraction-static 0.3 \
  --log-level info > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
# precompile 约需 300s，最多等 1800s
for i in $(seq 1 1800); do
  if grep -q "Uvicorn running on" "$LOG"; then echo "READY ${i}s"; break; fi
  if ! kill -0 "$PID" >/dev/null 2>&1; then echo "EXITED_EARLY"; tail -n 240 "$LOG"; exit 1; fi
  sleep 1
done
REQ='{"model":"/models/Qwen3-30B-A3B","messages":[{"role":"user","content":"Hello, what is 1+1?"},{"role":"assistant","content":"2"},{"role":"user","content":"What is Paris known for?"}],"max_tokens":30,"temperature":0.7}'
set +e
curl --max-time 300 -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" > "$RESP"
RC=$?
set -e
echo "CURL_RC=$RC"; wc -c "$RESP" || true; head -c 800 "$RESP"; echo
if kill -0 "$PID" >/dev/null 2>&1; then echo "SERVER_ALIVE"; else echo "SERVER_DIED"; fi
kill "$PID" >/dev/null 2>&1 || true
sleep 2
grep -n "POST /v1/chat/completions\|Traceback\|ValueError\|TypeError\|Anomalies\|Precompile finished" "$LOG" | tail -n 30 || true
