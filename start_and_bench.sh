# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source .venv/bin/activate

python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B --trust-remote-code --host 0.0.0.0 --port 8000 \
  --device tpu --tp-size 1 --nnodes 1 --log-level info --node-rank 0 \
  --dist-init-addr 0.0.0.0:10011 --dtype bfloat16 \
  --mem-fraction-static 0.7 --max-prefill-tokens 2048 --chunked-prefill-size -1 \
  --precompile-token-paddings 1024 2048 4096 8192 16384 32768 \
  --precompile-bs-paddings 1 2 4 8 12 16 24 32 48 64 \
  --max-running-requests 128 --page-size 64 --attention-backend fa \
  --skip-server-warmup --enable-scoring-cache \
  --multi-item-extend-batch-size 128 --disable-overlap-schedule \
  --multi-item-score-from-cache-v2-items-per-step 64 \
  --multi-item-score-label-only-logprob --multi-item-score-fastpath-log-metrics &
SERVER_PID=$!

echo "Waiting for server to start..."
timeout=300
elapsed=0
while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 5
    elapsed=$((elapsed+5))
    if [ "$elapsed" -ge "$timeout" ]; then
        echo "Server failed to start within time."
        kill $SERVER_PID
        exit 1
    fi
done
echo "Server is up!"

python benchmark.py

echo "Killing server..."
kill $SERVER_PID
