#!/bin/bash

HOST="${HOST:-localhost}"
PORT="${PORT:-30271}"
MODEL="${MODEL:-default}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-32000}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

DEFAULT_PROMPT='Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.\nLet'"'"'s think step by step and output the final answer within \\boxed{}.'
PROMPT="${1:-$DEFAULT_PROMPT}"

curl -sN "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d "$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [
    {"role": "user", "content": "${PROMPT}"}
  ],
  "temperature": ${TEMP},
  "top_p": ${TOP_P},
  "max_tokens": ${MAX_TOKENS},
  "chat_template_kwargs": {"enable_thinking": ${ENABLE_THINKING}},
  "stream": true
}
EOF
)" 2>/dev/null \
  | grep --line-buffered '^data: ' \
  | sed -u 's/^data: //' \
  | grep -v '^\[DONE\]' \
  | python3 -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        delta = obj["choices"][0]["delta"]
        content = delta.get("content") or delta.get("reasoning_content") or ""
        if content:
            print(content, end="", flush=True)
    except:
        pass
print()
'
