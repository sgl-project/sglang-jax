from __future__ import annotations

import ipaddress
import random
import urllib.parse
import uuid
from typing import Any


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def generate_bootstrap_room() -> int:
    return random.randint(0, 2**63 - 1)


def get_request_batch_size(request: dict[str, Any]) -> int | None:
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


def _expand_identity_field(value: Any, batch_size: int) -> list[str]:
    """Expand a scalar id into ``batch_size`` aligned per-item ids.

    Mirrors ``GenerateReqInput._normalize_rid``'s ``"{rid}_{i}"`` scheme so an
    already-list value is left untouched and a scalar becomes per-item unique.
    """

    if isinstance(value, list):
        return value
    return [f"{value}_{i}" for i in range(batch_size)]


def ensure_request_identity_fields(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    modified_request = request_data.copy()
    batch_size = get_request_batch_size(modified_request)
    rid = modified_request.get("rid")
    disagg_transfer_id = modified_request.get("disagg_transfer_id")

    if rid is None and disagg_transfer_id is None:
        if batch_size is None:
            rid = uuid.uuid4().hex
            disagg_transfer_id = rid
        else:
            rid = [uuid.uuid4().hex for _ in range(batch_size)]
            disagg_transfer_id = list(rid)
    elif rid is None:
        rid = disagg_transfer_id
    elif disagg_transfer_id is None:
        disagg_transfer_id = rid

    # For batched requests GenerateReqInput expands a scalar rid into per-item
    # ids ("{rid}_{i}") but indexes disagg_transfer_id as-is, so a scalar
    # transfer id would be shared by every element and the prefill side would
    # register colliding uuid:kv entries. Expand both to aligned per-item lists
    # here so each element carries a unique, P/D-consistent transfer id.
    if batch_size is not None:
        rid = _expand_identity_field(rid, batch_size)
        disagg_transfer_id = _expand_identity_field(disagg_transfer_id, batch_size)

    modified_request["rid"] = rid
    modified_request["disagg_transfer_id"] = disagg_transfer_id
    return modified_request


def inject_bootstrap_fields(
    request_data: dict[str, Any],
    *,
    prefill_server: str,
    bootstrap_port: int | None,
    bootstrap_host_override: str | None = None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(prefill_server)
    hostname = bootstrap_host_override or maybe_wrap_ipv6_address(
        parsed.hostname or ""
    )
    room = generate_bootstrap_room()
    modified_request = ensure_request_identity_fields(request_data)

    batch_size = get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [room + i for i in range(batch_size)],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": room,
            }
        )
    return modified_request
