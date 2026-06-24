"""Pydantic models for every message type in the gateway protocol."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


def new_id() -> str:
    return str(uuid.uuid4())


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    media_type: str
    data: str  # base64


class FileContent(BaseModel):
    type: Literal["file"]
    filename: str
    media_type: str
    data: str  # base64


class FileRefContent(BaseModel):
    type: Literal["file_ref"]
    path: str  # relative to project root_path


ContentBlock = Annotated[
    TextContent | ImageContent | FileContent | FileRefContent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Client → Gateway payloads
# ---------------------------------------------------------------------------


class PromptPayload(BaseModel):
    session_id: str | None
    content: list[ContentBlock]
    agent_name: str | None = None


class ApprovalResponsePayload(BaseModel):
    ref_id: str
    decision: Literal["yes", "no", "yes_session", "yes_always", "edit"]
    edited_command: str | None = None


class AbortPayload(BaseModel):
    session_id: str


class CommandPayload(BaseModel):
    name: str
    args: dict[str, Any] = {}


class PairRequestPayload(BaseModel):
    pairing_code: str
    device_info: dict[str, Any] = {}


class ProjectListPayload(BaseModel):
    pass


class ProjectSelectPayload(BaseModel):
    project_id: str


class ProjectUpsertPayload(BaseModel):
    project_id: str
    project: dict[str, Any]


class ProjectDeletePayload(BaseModel):
    project_id: str


class AgentUpsertPayload(BaseModel):
    name: str
    agent: dict[str, Any]


class AgentDeletePayload(BaseModel):
    name: str


class AgentListPayload(BaseModel):
    pass


class SessionResumePayload(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Client → Gateway envelopes
# ---------------------------------------------------------------------------


class PromptEnvelope(BaseModel):
    type: Literal["prompt"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: PromptPayload


class ApprovalResponseEnvelope(BaseModel):
    type: Literal["approval_response"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: ApprovalResponsePayload


class AbortEnvelope(BaseModel):
    type: Literal["abort"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: AbortPayload


class CommandEnvelope(BaseModel):
    type: Literal["command"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: CommandPayload


class PairRequestEnvelope(BaseModel):
    type: Literal["pair_request"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: PairRequestPayload


class ProjectListEnvelope(BaseModel):
    type: Literal["project_list"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: ProjectListPayload = Field(default_factory=ProjectListPayload)


class ProjectSelectEnvelope(BaseModel):
    type: Literal["project_select"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: ProjectSelectPayload


class ProjectUpsertEnvelope(BaseModel):
    type: Literal["project_upsert"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: ProjectUpsertPayload


class ProjectDeleteEnvelope(BaseModel):
    type: Literal["project_delete"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: ProjectDeletePayload


class AgentUpsertEnvelope(BaseModel):
    type: Literal["agent_upsert"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: AgentUpsertPayload


class AgentDeleteEnvelope(BaseModel):
    type: Literal["agent_delete"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: AgentDeletePayload


class AgentListEnvelope(BaseModel):
    type: Literal["agent_list"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: AgentListPayload = Field(default_factory=AgentListPayload)


class SessionResumeEnvelope(BaseModel):
    type: Literal["session_resume"]
    id: str = Field(default_factory=new_id)
    timestamp: int = Field(default_factory=now_ms)
    payload: SessionResumePayload


ClientMessage = Annotated[
    PromptEnvelope
    | ApprovalResponseEnvelope
    | AbortEnvelope
    | CommandEnvelope
    | PairRequestEnvelope
    | ProjectListEnvelope
    | ProjectSelectEnvelope
    | ProjectUpsertEnvelope
    | ProjectDeleteEnvelope
    | AgentUpsertEnvelope
    | AgentDeleteEnvelope
    | AgentListEnvelope
    | SessionResumeEnvelope,
    Field(discriminator="type"),
]

_client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str) -> ClientMessage:
    return _client_message_adapter.validate_json(raw)


# ---------------------------------------------------------------------------
# Gateway → Client message constructors
# ---------------------------------------------------------------------------


def _envelope(msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": msg_type,
        "id": new_id(),
        "timestamp": now_ms(),
        "payload": payload,
    }


def make_session_start(
    auth_mode: str,
    default_model_name: str,
    available_backends: list[str],
) -> dict[str, Any]:
    return _envelope(
        "session_start",
        {
            "auth": auth_mode,
            "device_id": None,
            "gateway_version": "1.0.0",
            "default_model": {
                "backend": "local",
                "model_name": default_model_name,
            },
            "available_backends": available_backends,
            "capabilities": ["streaming", "tools", "images", "file_attachments"],
        },
    )


def make_stream_chunk(
    session_id: str,
    ref_id: str,
    delta: str,
    role: str,
) -> dict[str, Any]:
    return _envelope(
        "stream_chunk",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "delta": delta,
            "role": role,
        },
    )


def make_stream_end(
    session_id: str,
    ref_id: str,
    aborted: bool,
    usage: dict[str, Any] | None,
    context_limit: int = 32000,
    session_budget: dict[str, int] | None = None,
    finish_reason: str | None = None,
    tool_calls_made: int = 0,
    duration_ms: int | None = None,
    tokens_used: int | None = None,
    tool_call_count: int | None = None,
) -> dict[str, Any]:
    usage_out: dict[str, Any] = {}
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = prompt_tokens + completion_tokens
        usage_out = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_budget": session_budget
            or {
                "used": total,
                "limit": context_limit,
            },
        }
    elif session_budget:
        usage_out = {"context_budget": session_budget}
    result: dict[str, Any] = {
        "session_id": session_id,
        "ref_id": ref_id,
        "aborted": aborted,
        "usage": usage_out,
        "finish_reason": finish_reason,
        "tool_calls_made": tool_calls_made,
    }
    if duration_ms is not None:
        result["duration_ms"] = duration_ms
    if tokens_used is not None:
        result["tokens_used"] = tokens_used
    if tool_call_count is not None:
        result["tool_call_count"] = tool_call_count
    return _envelope("stream_end", result)


def make_error(
    ref_id: str | None,
    code: str,
    message: str,
    recoverable: bool = True,
) -> dict[str, Any]:
    return _envelope(
        "error",
        {
            "ref_id": ref_id,
            "code": code,
            "message": message,
            "recoverable": recoverable,
        },
    )


def make_approval_request(
    session_id: str,
    approval_id: str,
    tool: str,
    command: str,
    risk: str,
    context: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    msg = _envelope(
        "approval_request",
        {
            "session_id": session_id,
            "tool": tool,
            "command": command,
            "risk": risk,
            "context": context,
            "timeout_seconds": timeout_seconds,
        },
    )
    # Override the auto-generated id so permission manager can use it as ref
    msg["id"] = approval_id
    return msg


def make_tool_output(
    session_id: str,
    ref_id: str,
    tool: str,
    command: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return _envelope(
        "tool_output",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "tool": tool,
            "command": command,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", 0),
            "truncated": result.get("truncated", False),
            "duration_ms": result.get("duration_ms", 0),
        },
    )


def make_queue_position(
    session_id: str,
    ref_id: str,
    position: int,
    active_session_id: str | None,
) -> dict[str, Any]:
    return _envelope(
        "queue_position",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "position": position,
            "active_session_id": active_session_id,
        },
    )


def make_model_waiting(
    session_id: str,
    ref_id: str,
    waited_seconds: int,
    max_wait_seconds: int,
) -> dict[str, Any]:
    """Gateway→client only: the prompt is the active request but the gateway is
    waiting for the model to come online (distinct from queue_position, which
    means waiting behind another request)."""
    return _envelope(
        "model_waiting",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "waited_seconds": waited_seconds,
            "max_wait_seconds": max_wait_seconds,
        },
    )


def make_project_registry(
    ref_id: str,
    projects: dict[str, Any],
) -> dict[str, Any]:
    return _envelope(
        "project_registry",
        {
            "ref_id": ref_id,
            "projects": projects,
        },
    )


def make_file_index(ref_id: str, project_id: str, files: list[str]) -> dict[str, Any]:
    return _envelope("file_index", {"ref_id": ref_id, "project_id": project_id, "files": files})


def make_project_context(
    ref_id: str,
    project_id: str,
    description: str,
    root_path: str,
    context_file: str,
    agents: dict[str, Any],
    last_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _envelope(
        "project_context",
        {
            "ref_id": ref_id,
            "project_id": project_id,
            "description": description,
            "root_path": root_path,
            "context_file": context_file,
            "agents": agents,
            "last_session": last_session,
        },
    )


def make_session_resumed(
    session_id: str,
    incomplete_turn: bool,
    recovered_from_disk: bool,
) -> dict[str, Any]:
    return _envelope(
        "session_resumed",
        {
            "session_id": session_id,
            "incomplete_turn": incomplete_turn,
            "recovered_from_disk": recovered_from_disk,
        },
    )


def make_session_history(
    session_id: str,
    turns: list[dict[str, Any]],
    has_compacted_prefix: bool,
) -> dict[str, Any]:
    """Send conversation history after session_resumed.

    Each turn: {"role": "user"|"assistant", "content": str, "timestamp": float,
                 "is_summary": bool (only on the compaction summary turn)}
    """
    return _envelope(
        "session_history",
        {
            "session_id": session_id,
            "turns": turns,
            "has_compacted_prefix": has_compacted_prefix,
        },
    )


def make_agent_registry(
    ref_id: str,
    project_id: str,
    agents: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    return _envelope(
        "agent_registry",
        {
            "ref_id": ref_id,
            "project_id": project_id,
            "agents": agents,
            "errors": errors,
        },
    )


def make_config_reloaded(
    scope: str,
    project_id: str | None,
    changed: list[str],
    message: str,
) -> dict[str, Any]:
    return _envelope(
        "config_reloaded",
        {
            "scope": scope,
            "project_id": project_id,
            "changed": changed,
            "message": message,
        },
    )


def make_status(
    session_id: str,
    state: str,
    token_usage: dict[str, Any],
    queue_position: int = 0,
    duration_ms: int | None = None,
    tokens_used: int | None = None,
    tool_call_count: int | None = None,
) -> dict[str, Any]:
    """Build a status message.

    token_usage may be the simple {"used": N, "limit": N} form or the full
    breakdown form {"used": N, "limit": N, "breakdown": {...}} produced by
    ContextBudget.to_dict().
    """
    result: dict[str, Any] = {
        "session_id": session_id,
        "state": state,
        "queue_position": queue_position,
        "token_usage": token_usage,
    }
    if duration_ms is not None:
        result["duration_ms"] = duration_ms
    if tokens_used is not None:
        result["tokens_used"] = tokens_used
    if tool_call_count is not None:
        result["tool_call_count"] = tool_call_count
    return _envelope("status", result)


def make_command_result(
    ref_id: str,
    name: str,
    success: bool,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _envelope(
        "command_result",
        {
            "ref_id": ref_id,
            "name": name,
            "success": success,
            "message": message,
            "data": data or {},
        },
    )


def make_diff_preview(
    session_id: str,
    ref_id: str,
    path: str,
    diff: str,
    additions: int,
    deletions: int,
) -> dict[str, Any]:
    return _envelope(
        "diff_preview",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "path": path,
            "diff": diff,
            "stats": {"additions": additions, "deletions": deletions},
        },
    )


def make_plan_status(
    session_id: str,
    ref_id: str,
    step: int,
    total_steps: int,
    status: str,
    description: str,
    summary: str = "",
) -> dict[str, Any]:
    return _envelope(
        "plan_status",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "step": step,
            "total_steps": total_steps,
            "status": status,
            "description": description,
            "summary": summary,
        },
    )


def make_plan_revision(
    session_id: str,
    ref_id: str,
    revision_number: int,
    action: str,
    reason: str,
    description: str,
    revised_plan: list[dict[str, str | object]],
) -> dict[str, Any]:
    """Create a plan_revision message following the protocol envelope."""
    return _envelope(
        "plan_revision",
        {
            "session_id": session_id,
            "ref_id": ref_id,
            "revision_number": revision_number,
            "action": action,
            "reason": reason,
            "description": description,
            "revised_plan": revised_plan,
        },
    )
