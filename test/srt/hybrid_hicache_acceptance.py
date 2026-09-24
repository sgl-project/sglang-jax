"""Check frozen-manifest FULL+SWA serving captures without importing JAX."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

RUNS = ("off", "off_repeat", "on", "on_repeat")
TRANSFER = ("d2h", "device_evict", "slot_overwrite", "h2d")
OUTPUT_TOKEN_PARAMS = {"no_stop_trim": True, "skip_special_tokens": False}


def preserves_output_tokens(params):
    return isinstance(params, dict) and all(
        params.get(name) is value for name, value in OUTPUT_TOKEN_PARAMS.items()
    )


def check(manifest, runs, *, report=None):
    """Return violations; optionally record logprob differences without a tolerance."""
    errors = []
    details = {}
    if report is not None:
        report["logprobs"] = details
    if manifest.get("return_logprob") is not True:
        errors.append("manifest must freeze return_logprob: true")
    if not preserves_output_tokens(manifest.get("sampling_params")):
        errors.append(f"manifest sampling_params must freeze {OUTPUT_TOKEN_PARAMS}")
    cases = manifest.get("cases", [])
    expected = [case.get("rid") for case in cases]
    if not expected or any(not isinstance(rid, str) or not rid for rid in expected):
        errors.append("manifest needs nonempty string RIDs")
    if len(set(expected)) != len(expected):
        errors.append("manifest has duplicate RIDs")
    indexed = {}

    for run_name in RUNS:
        run = runs.get(run_name)
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            errors.append(f"{run_name}: missing results list")
            continue
        if run.get("return_logprob") is not True:
            errors.append(f"{run_name}: return_logprob must match frozen manifest true")
        if not preserves_output_tokens(run.get("sampling_params")):
            errors.append(f"{run_name}: sampling_params must preserve {OUTPUT_TOKEN_PARAMS}")
        found = [r.get("meta_info", {}).get("id") for r in run["results"] if isinstance(r, dict)]
        counts = Counter(found)
        for rid in expected:
            if counts[rid] == 0:
                errors.append(f"{run_name}: missing RID {rid}")
            elif counts[rid] > 1:
                errors.append(f"{run_name}: duplicate RID {rid}")
        for rid in counts.keys() - set(expected):
            errors.append(f"{run_name}: unexpected RID {rid}")
        indexed[run_name] = {
            r.get("meta_info", {}).get("id"): r for r in run["results"] if isinstance(r, dict)
        }

    for case in cases:
        rid = case.get("rid")
        kind = case.get("kind")
        if kind not in ("normal", "retract", "abort"):
            errors.append(f"manifest {rid}: invalid kind {kind}")
            continue
        if kind != "abort" and case.get("expected_finish_reason") not in (
            "stop",
            "length",
        ):
            errors.append(f"manifest {rid}: expected_finish_reason must be stop or length")
        if isinstance(case.get("dp_rank"), bool) or not isinstance(case.get("dp_rank"), int):
            errors.append(f"manifest {rid}: invalid dp_rank")
        if not isinstance(case.get("max_new_tokens"), int) or case["max_new_tokens"] < 1:
            errors.append(f"manifest {rid}: invalid max_new_tokens")
            continue
        outputs = {}
        logprobs = {}
        detail = details[rid] = {"runs": {}, "comparisons": {}}
        for run_name in RUNS:
            result = indexed.get(run_name, {}).get(rid)
            if result is None:
                detail["runs"][run_name] = {"status": "invalid", "reason": "missing RID"}
                continue
            label = f"{run_name}/{rid}"
            meta = result.get("meta_info", {})
            ids = result.get("output_ids")
            if result.get("error"):
                errors.append(f"{label}: unexpected error {result['error']}")
            if meta.get("dp_rank") != case.get("dp_rank"):
                errors.append(
                    f"{label}: actual dp_rank {meta.get('dp_rank')} != requested {case.get('dp_rank')}"
                )
            if not isinstance(ids, list) or any(
                isinstance(v, bool) or not isinstance(v, int) for v in ids
            ):
                errors.append(f"{label}: output_ids must be complete integer list")
                detail["runs"][run_name] = {"status": "invalid", "reason": "invalid output_ids"}
                continue
            values, error = _output_logprobs(meta.get("output_token_logprobs"), ids)
            if error:
                errors.append(f"{label}: {error}")
                detail["runs"][run_name] = {"status": "invalid", "reason": error}
            else:
                logprobs[run_name] = values
                detail["runs"][run_name] = {"status": "valid", "token_count": len(values)}
            reason = meta.get("finish_reason")
            reason_type = reason.get("type") if isinstance(reason, dict) else None
            if kind == "abort":
                if reason_type != "abort":
                    errors.append(f"{label}: expected abort terminal reason")
                if not _has_event(result, "cancelled", case.get("dp_rank")):
                    errors.append(f"{label}: missing cancelled evidence")
                if not _has_event(result, "cleanup", case.get("dp_rank")):
                    errors.append(f"{label}: missing cleanup evidence")
                continue
            if reason_type not in ("stop", "length"):
                errors.append(f"{label}: missing normal terminal reason")
            if reason_type != case.get("expected_finish_reason"):
                errors.append(
                    f"{label}: expected terminal {case.get('expected_finish_reason')}, got {reason_type}"
                )
            if not ids:
                errors.append(f"{label}: empty normal output")
            if meta.get("completion_tokens") != len(ids):
                errors.append(f"{label}: completion_tokens differs from output_ids length")
            if len(ids) > case["max_new_tokens"]:
                errors.append(f"{label}: output_ids exceed max_new_tokens")
            if reason_type == "length" and len(ids) != case["max_new_tokens"]:
                errors.append(f"{label}: length termination before max_new_tokens")
            if kind == "retract":
                for event in ("retract", "reschedule", "complete"):
                    if not _has_event(result, event, case.get("dp_rank")):
                        errors.append(f"{label}: missing {event} evidence")
            if run_name.startswith("on"):
                for component in case.get("l2_components", []):
                    if component not in ("FULL", "SWA"):
                        errors.append(f"manifest {rid}: invalid L2 component {component}")
                        continue
                    for event in TRANSFER:
                        if not _has_event(result, event, case.get("dp_rank"), component):
                            errors.append(f"{label}: missing {component} {event} evidence")
            outputs[run_name] = (ids, reason)
        for a, b in (("off", "off_repeat"), ("on", "on_repeat"), ("off", "on")):
            if a in outputs and b in outputs and outputs[a] != outputs[b]:
                errors.append(f"{rid}: {a} differs from {b} in full output_ids or terminal reason")
            comparison = {"status": "not_compared"}
            if kind == "abort":
                comparison["reason"] = "abort outputs may differ"
            elif a not in logprobs or b not in logprobs:
                comparison["reason"] = "missing or invalid logprobs"
            elif outputs[a][0] != outputs[b][0]:
                comparison["reason"] = "output_ids differ"
            else:
                differences = [abs(x - y) for x, y in zip(logprobs[a], logprobs[b], strict=True)]
                changed = [i for i, value in enumerate(differences) if value != 0]
                comparison = {
                    "status": "compared",
                    "token_count": len(differences),
                    "different_count": len(changed),
                    "max_abs_diff": max(differences, default=0),
                    "mean_abs_diff": sum(differences) / len(differences) if differences else 0,
                    "first_different_index": changed[0] if changed else None,
                }
            detail["comparisons"][f"{a}/{b}"] = comparison
    return errors


def _output_logprobs(entries, ids):
    if not isinstance(entries, list):
        return None, "missing output_token_logprobs list"
    if len(entries) != len(ids):
        return None, "output_token_logprobs length differs from output_ids"
    values = []
    for index, (entry, token) in enumerate(zip(entries, ids, strict=True)):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None, f"invalid output_token_logprobs entry at index {index}"
        value, token_id, _text = entry
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return None, f"output_token_logprobs must be finite at index {index}"
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id != token:
            return None, f"output_token_logprobs token_id differs from output_ids at index {index}"
        values.append(value)
    return values, None


def _has_event(result, event_type, rank, component=None):
    return any(
        isinstance(event, dict)
        and event.get("type") == event_type
        and event.get("dp_rank") == rank
        and (component is None or event.get("component") == component)
        and isinstance(event.get("source"), str)
        and bool(event["source"].strip())
        for event in result.get("events", [])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for name in RUNS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, help="Write per-RID validity and logprob differences"
    )
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    runs = {}
    for name in RUNS:
        run = json.loads(getattr(args, name).read_text())
        if run.get("manifest_sha256") != manifest_hash:
            parser.error(f"{name}: manifest_sha256 does not match frozen manifest")
        runs[name] = run
    report = {}
    errors = check(manifest, runs, report=report)
    report.update(manifest_sha256=manifest_hash, errors=errors, status="FAIL" if errors else "PASS")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Per-RID logprob report: {args.report}")
    else:
        for rid, detail in report["logprobs"].items():
            print(f"logprobs {rid}: {json.dumps(detail, sort_keys=True)}")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(manifest['cases'])} RIDs in four runs; manifest sha256={manifest_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
