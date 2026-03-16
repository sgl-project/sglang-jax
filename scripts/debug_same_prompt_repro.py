import argparse
import time

import requests
from transformers import AutoTokenizer

from scripts.debug_gpqa_hang_repro import SYSTEM_PROMPT, chat_ids


def find_same_prompt(tokenizer, target_tokens: int) -> tuple[int, int, str]:
    for common_n in range(1, 400):
        prefix = "common " * common_n
        for tail_n in range(1, 1400):
            user = (prefix + ("target " * tail_n)).strip()
            n = len(chat_ids(tokenizer, SYSTEM_PROMPT, user))
            if n == target_tokens:
                return common_n, tail_n, user
            if n > target_tokens:
                break
    raise RuntimeError(f"no synthetic prompt with {target_tokens} tokens found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/MiMo-V2-Flash")
    parser.add_argument("--base-url", default="http://127.0.0.1:30271/v1/chat/completions")
    parser.add_argument("--target-tokens", type=int, default=659)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    common_n, tail_n, user = find_same_prompt(tok, args.target_tokens)
    prompt_tokens = len(chat_ids(tok, SYSTEM_PROMPT, user))
    print(
        {
            "common_n": common_n,
            "tail_n": tail_n,
            "prompt_tokens": prompt_tokens,
            "user_chars": len(user),
        },
        flush=True,
    )

    session = requests.Session()
    session.trust_env = False
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
    }

    for rep in (1, 2):
        started = time.time()
        print({"phase": "start", "rep": rep}, flush=True)
        resp = session.post(args.base_url, json=payload, timeout=args.timeout)
        body = resp.json()
        print(
            {
                "phase": "done",
                "rep": rep,
                "status": resp.status_code,
                "elapsed_s": round(time.time() - started, 3),
                "body_keys": sorted(body.keys()),
                "text": ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:80],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
