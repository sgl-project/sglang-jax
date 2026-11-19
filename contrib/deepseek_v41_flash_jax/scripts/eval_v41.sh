#!/bin/bash
# Quality eval for the V4.1 JAX server, same harness/settings as the V4 Flash baseline (lm-eval chat completions, temperature 1.0, 8192 max new tokens).
# Usage (on host 0): eval_v41.sh <tasks:limit,...> [concurrency]   e.g. "gsm8k_cot_llama:500,ifeval:541,mmlu_pro:150"
MODEL=DeepSeek-V4.1-Flash; URL=http://localhost:8000/v1/chat/completions; CONC=${2:-64}
source ~/.evalenv/bin/activate; mkdir -p ~/results/quality_v41
for spec in $(echo "${1:-gsm8k_cot_llama:500}" | tr ',' ' '); do
  TASK=${spec%%:*}; LIM=${spec##*:}
  echo "== $TASK limit $LIM start $(date -u +%T)"
  lm_eval --model local-chat-completions \
    --model_args "model=$MODEL,base_url=$URL,num_concurrent=$CONC,max_retries=2,timeout=3600,tokenized_requests=False" \
    --tasks $TASK --apply_chat_template --limit $LIM \
    --gen_kwargs "temperature=1.0,top_p=1.0,max_gen_toks=8192,seed=None" \
    --output_path ~/results/quality_v41/${MODEL}_${TASK}.json --log_samples 2>&1 | tail -n 14
  echo "== $TASK done $(date -u +%T)"
done
