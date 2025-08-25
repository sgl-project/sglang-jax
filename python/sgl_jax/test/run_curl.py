"""
Usage:
python3 run_curl.py --base-url=http://127.0.0.1:8100 --return-logprob=true --top-logprobs-num 3   --logprob-start-len 1
"""

import argparse
import json
import time

import requests


def run_curl(args):
    base_url = args.base_url

    # Request payload matching the curl command
    payload = {
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        "text": args.text,
        "return_logprob": args.return_logprob,
        "top_logprobs_num": args.top_logprobs_num,
        "token_ids_logprob": args.token_ids_logprob,
        "logprob_start_len": args.logprob_start_len,
    }

    headers = {"Content-Type": "application/json"}

    print(f"Sending request to {base_url}/generate")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(
            f"{base_url}/generate", json=payload, headers=headers, timeout=30
        )

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
        default=None,
    )
    parser.add_argument(
        "--text",
        type=str,
        default="the capital of France is",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--return-logprob",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--top-logprobs-num",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--token-ids-logprob",
        type=list,
        default=None,
    )
    parser.add_argument(
        "--logprob-start-len",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    run_curl(args)
