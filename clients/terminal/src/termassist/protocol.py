"""Client-side message builders and envelope helpers.

No Pydantic — just dict construction and basic validation.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _envelope(msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": msg_type,
        "id": _new_id(),
        "timestamp": _now_ms(),
        "payload": payload,
    }


def make_prompt(
    session_id: str | None,
    project_id: str,
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    return _envelope(
        "prompt",
        {
            "session_id": session_id,
            "project_id": project_id,
            "content": content,
        },
    )


def make_approval_response(
    ref_id: str,
    decision: str,
    edited_command: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ref_id": ref_id, "decision": decision, "edited_command": edited_command
    }
    return _envelope("approval_response", payload)


def make_abort(session_id: str) -> dict[str, Any]:
    return _envelope("abort", {"session_id": session_id})


def make_command(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return _envelope("command", {"name": name, "args": args or {}})


def make_pair_request(pairing_code: str, device_info: dict[str, Any]) -> dict[str, Any]:
    return _envelope("pair_request", {"pairing_code": pairing_code, "device_info": device_info})


def make_project_select(project_id: str) -> dict[str, Any]:
    return _envelope("project_select", {"project_id": project_id})


def make_project_list() -> dict[str, Any]:
    return _envelope("project_list", {})


def make_project_upsert(project_id: str, project: dict[str, Any]) -> dict[str, Any]:
    return _envelope("project_upsert", {"project_id": project_id, "project": project})


def make_session_resume(session_id: str) -> dict[str, Any]:
    return _envelope("session_resume", {"session_id": session_id})


def parse_message(raw: str) -> dict[str, Any]:
    """Parse and minimally validate a gateway message envelope.

    Returns a dict with at least 'type' and 'payload' keys.
    Raises ValueError if the envelope is malformed.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(msg, dict):
        raise ValueError("Message must be a JSON object")

    for field in ("type", "id", "timestamp", "payload"):
        if field not in msg:
            raise ValueError(f"Missing required field: {field!r}")

    if not isinstance(msg["type"], str):
        raise ValueError("Field 'type' must be a string")

    if not isinstance(msg["payload"], dict):
        raise ValueError("Field 'payload' must be an object")

    return msg
