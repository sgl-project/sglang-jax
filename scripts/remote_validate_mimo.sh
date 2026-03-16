#!/bin/bash
set -euo pipefail

if [[ "${SKYPILOT_NODE_RANK:-0}" != "0" ]]; then
  exit 0
fi

cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python

echo "[phase] wait-ready"
for i in $(seq 1 600); do
  out="$(curl -sf http://127.0.0.1:30271/get_model_info || true)"
  if printf '%s' "$out" | grep -q '"is_ready":true'; then
    echo "attempt=$i is_ready=true"
    break
  fi
  echo "attempt=$i is_ready=false"
  sleep 2
done

echo "[phase] benchmark-b4"
python -m sgl_jax.bench_one_batch_server \
  --model-path /models/MiMo-V2-Flash \
  --base-url http://127.0.0.1:30271 \
  --batch-size 4 \
  --input-len 472 \
  --output-len 128 \
  --skip-server-info \
  --skip-flush-cache

echo "[phase] health-after-benchmark"
curl -sf http://127.0.0.1:30271/get_model_info

system_msg=$(
  cat <<'EOF'
You are MiMo, an AI assistant developed by Xiaomi.

Today is March 12, 2026 Thursday. Your knowledge cutoff date is December 2024.
EOF
)

echo "[phase] gpqa"
python test/srt/run_eval.py \
  --eval-name gpqa \
  --base-url http://127.0.0.1:30271 \
  --num-examples 10 \
  --num-threads 1 \
  --temperature 0.8 \
  --top-p 0.95 \
  --system-message "$system_msg" \
  --max-tokens 64 \
  --request-timeout 120 \
  --max-retries 2
