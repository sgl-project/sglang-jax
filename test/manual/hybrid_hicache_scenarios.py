"""Run a frozen staged HTTP request sequence and retain raw results/snapshots.

This drives pressure and abort; it does not manufacture L2/lifecycle events or
claim the target path ran. Audit the diagnostic server's actual method records.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def validate_manifest(manifest):
    if manifest.get("return_logprob") is not True:
        raise ValueError("manifest must freeze return_logprob=true")
    sampling = manifest.get("sampling_params", {})
    if sampling.get("no_stop_trim") is not True or sampling.get("skip_special_tokens") is not False:
        raise ValueError("manifest must freeze no_stop_trim=true and skip_special_tokens=false")
    cases = {case["rid"]: case for case in manifest["cases"]}
    if len(cases) != len(manifest["cases"]):
        raise ValueError("duplicate case RID")
    rids = [rid for stage in manifest["stages"] for rid in stage["rids"]]
    if len(rids) != len(set(rids)) or set(rids) != set(cases):
        raise ValueError("stages must schedule each frozen RID exactly once")
    for stage in manifest["stages"]:
        if not stage["rids"]:
            raise ValueError("each stage needs at least one RID")
        abort = stage.get("abort_when_running")
        if abort is not None and (abort not in stage["rids"] or cases[abort]["kind"] != "abort"):
            raise ValueError("abort_when_running must name an abort case in the same stage")
    for case in cases.values():
        if not case.get("input_ids") or any(type(token) is not int for token in case["input_ids"]):
            raise ValueError("input_ids must be a frozen nonempty integer list")
        if type(case.get("dp_rank")) is not int or case["dp_rank"] < 0:
            raise ValueError("dp_rank must be a nonnegative integer")
        if type(case.get("max_new_tokens")) is not int or case["max_new_tokens"] < 1:
            raise ValueError("max_new_tokens must be a positive integer")
    return cases


def run(manifest, base_url, raw, observations, save, *, timeout=600):
    cases = validate_manifest(manifest)

    def state(label):
        response = requests.get(base_url + "/get_server_info", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        observations.append({"label": label, "wall_time_ns": time.time_ns(), "server_info": data})
        save()
        return data

    def generate(case):
        response = requests.post(
            base_url + "/generate",
            json={
                "rid": case["rid"],
                "input_ids": case["input_ids"],
                "dp_rank": case["dp_rank"],
                "stream": False,
                "return_logprob": True,
                "sampling_params": {
                    "temperature": 0,
                    "sampling_seed": 3,
                    "max_new_tokens": case["max_new_tokens"],
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    for index, stage in enumerate(manifest["stages"]):
        state(f"stage-{index}-before")
        if stage.get("flush_before", False):
            response = requests.post(base_url + "/flush_cache", timeout=timeout)
            observations.append(
                {
                    "label": f"stage-{index}-flush",
                    "http_status": response.status_code,
                    "body": response.text,
                }
            )
            save()
            response.raise_for_status()
            state(f"stage-{index}-after-flush")
        with ThreadPoolExecutor(max_workers=len(stage["rids"])) as executor:
            futures = {rid: executor.submit(generate, cases[rid]) for rid in stage["rids"]}
            abort = stage.get("abort_when_running")
            abort_error = None
            try:
                if abort is not None:
                    deadline = time.monotonic() + timeout
                    while True:
                        info = state(f"stage-{index}-abort-poll")
                        running = [
                            rid
                            for item in info.get("internal_states", [])
                            for rid in item.get("running_batch_rids", [])
                        ]
                        if abort in running:
                            response = requests.post(
                                base_url + "/abort_request", json={"rid": abort}, timeout=timeout
                            )
                            observations.append(
                                {
                                    "label": f"stage-{index}-abort-sent",
                                    "rid": abort,
                                    "http_status": response.status_code,
                                    "body": response.text,
                                }
                            )
                            save()
                            response.raise_for_status()
                            break
                        if futures[abort].done() or time.monotonic() >= deadline:
                            raise RuntimeError(
                                f"{abort}: no active request observed before completion/deadline; abort inconclusive"
                            )
                        time.sleep(0.1)
            except Exception as exc:
                abort_error = exc
            for rid, future in futures.items():
                try:
                    raw["results"].append(future.result())
                except Exception as exc:
                    raw["results"].append(
                        {"meta_info": {"id": rid}, "error": f"{type(exc).__name__}: {exc}"}
                    )
                save()
        state(f"stage-{index}-after")
        if abort_error is not None:
            raise abort_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()
    frozen = args.manifest.read_bytes()
    manifest = json.loads(frozen)
    validate_manifest(manifest)
    if args.check_manifest:
        print("Manifest checked; no HTTP requests made.")
        return
    if args.output.exists():
        parser.error("output exists; use a fresh run path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "manifest_sha256": hashlib.sha256(frozen).hexdigest(),
        "return_logprob": True,
        "sampling_params": {"no_stop_trim": True, "skip_special_tokens": False},
        "results": [],
    }
    observations = []

    def save():
        args.output.write_text(json.dumps(raw, indent=2) + "\n")
        args.output.with_suffix(".observations.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )

    save()
    run(manifest, args.base_url.rstrip("/"), raw, observations, save, timeout=args.timeout)


if __name__ == "__main__":
    main()
