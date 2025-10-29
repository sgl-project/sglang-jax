"""
Usage:
python3 run_curl.py --base-url=http://127.0.0.1:8100 --return-logprob=true --top-logprobs-num 3   --logprob-start-len 1
"""

import argparse
import json

import requests


def run_curl(args):
    base_url = args.base_url

    # Request payload matching the curl command
    payload = {
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": getattr(args, "max_new_tokens", 10),
            "frequency_penalty": getattr(args, "frequency_penalty", 0.0),
            "presence_penalty": getattr(args, "presence_penalty", 0.0),
            "min_new_tokens": getattr(args, "min_new_tokens", 0),
        },
        "text": args.text,
        "return_logprob": getattr(args, "return_logprob", False),
        "top_logprobs_num": getattr(args, "top_logprobs_num", 0),
        "token_ids_logprob": getattr(args, "token_ids_logprob", None),
        "logprob_start_len": getattr(args, "logprob_start_len", -1),
    }

    headers = {"Content-Type": "application/json"}

    print(f"Sending request to {base_url}/generate")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(f"{base_url}/generate", json=payload, headers=headers, timeout=30)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            # print(f"{json.dumps(result,indent=4, ensure_ascii=False)}")
            return result
        else:
            print(f"Error Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
    )
    parser.add_argument(
        "--text",
        type=str,
    )
    parser.add_argument(
        "--temperature",
        type=int,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
    )
    parser.add_argument(
        "--return-logprob",
        type=bool,
    )
    parser.add_argument(
        "--top-logprobs-num",
        type=int,
    )
    parser.add_argument(
        "--token-ids-logprob",
        type=list,
    )
    parser.add_argument(
        "--logprob-start-len",
        type=int,
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
    )
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--min-new-tokens", type=int)
    args = parser.parse_args()

    run_curl(args)
