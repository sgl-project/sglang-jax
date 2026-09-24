"""Capture ordinary frozen FULL+SWA requests and attach operator evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def capture(manifest, evidence, post):
    """Submit only normal cases; keep each server response unchanged except events."""
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest has no cases")
    rids = [case.get("rid") for case in cases]
    if len(rids) != len(set(rids)):
        raise ValueError("manifest has duplicate RID")
    normal = [case for case in cases if case.get("kind") == "normal"]
    if not normal:
        raise ValueError("manifest has no ordinary cases to capture")
    for case in normal:
        rid = case["rid"]
        ids = case.get("input_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(i, bool) or not isinstance(i, int) for i in ids)
        ):
            raise ValueError(f"{rid}: input_ids must be nonempty integer list")
        if (
            isinstance(case.get("dp_rank"), bool)
            or not isinstance(case.get("dp_rank"), int)
            or case["dp_rank"] < 0
        ):
            raise ValueError(f"{rid}: invalid dp_rank")
        limit = case.get("max_new_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{rid}: invalid max_new_tokens")

    responses = []
    for case in normal:
        payload = {
            "rid": case["rid"],
            "input_ids": case["input_ids"],
            "dp_rank": case["dp_rank"],
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": case["max_new_tokens"],
                "sampling_seed": 3,
            },
        }
        response = post(payload)
        if not isinstance(response, dict):
            raise TypeError(f"{case['rid']}: /generate returned non-object response")
        responses.append(response)

    if callable(evidence):
        evidence = evidence()
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be RID-to-events mapping")
    results = []
    for case, response in zip(normal, responses, strict=True):
        rid = case["rid"]
        if rid not in evidence:
            raise ValueError(f"{rid}: missing evidence list")
        events = evidence.get(rid)
        if not isinstance(events, list):
            raise TypeError(f"{rid}: evidence must be event list")
        # The checker enforces L2 evidence for on/on_repeat; off captures
        # legitimately use the same manifest with empty transfer evidence.
        for event in events:
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("source"), str)
                or not event["source"].strip()
            ):
                raise ValueError(f"{rid}: evidence event requires source reference")
        results.append({**response, "events": events})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence", type=Path, required=True, help="RID-to-events JSON from this run"
    )
    parser.add_argument("--base-url", help="Server URL for live capture")
    parser.add_argument(
        "--from-raw",
        type=Path,
        help="Offline assembly of an earlier raw capture, with no HTTP requests",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Keep unmodified HTTP results even if evidence assembly fails",
    )
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if args.from_raw:
        raw_file = json.loads(args.from_raw.read_text())
        if raw_file.get("manifest_sha256") != digest:
            parser.error("raw capture manifest_sha256 mismatch")
        raw_iter = iter(raw_file.get("results", []))

        def post(_payload):
            try:
                return next(raw_iter)
            except StopIteration as exc:
                raise ValueError("raw capture has fewer ordinary responses than manifest") from exc

    else:
        if not args.base_url or not args.raw_output:
            parser.error("live capture requires --base-url and --raw-output")
    raw = []
    if not args.from_raw:

        def post(payload):
            response = requests.post(
                f"{args.base_url.rstrip('/')}/generate", json=payload, timeout=300
            )
            response.raise_for_status()
            data = response.json()
            raw.append(data)
            args.raw_output.write_text(
                json.dumps({"manifest_sha256": digest, "results": raw}, indent=2) + "\n"
            )
            return data

    results = capture(manifest, lambda: json.loads(args.evidence.read_text()), post)
    args.output.write_text(
        json.dumps({"manifest_sha256": digest, "results": results}, indent=2) + "\n"
    )
    print(
        f"Captured {len(results)} ordinary requests; add manual retract/abort results before checking"
    )


if __name__ == "__main__":
    main()
