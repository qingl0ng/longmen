# Gateway Protocol Specification

**Version:** 1.0.0
**Transport:** WebSocket (primary), HTTP (admin endpoints)
**Serialization:** JSON (all messages)

---

## Overview

The Gateway is a standalone daemon that mediates between any number of clients (terminal CLI, desktop GUI, web frontend) and a vLLM model server. Clients connect over WebSocket. The Gateway is the **single source of truth** — it owns projects, agent definitions, conversation history, tool execution, permission enforcement, and request scheduling. Clients are thin — they render streaming output, collect user input, and forward approval decisions.

The Gateway MUST be client-agnostic. It knows nothing about terminal escape codes, Qt widgets, or browser DOM. All rendering decisions belong to the client. The Gateway sends structured data; the client decides how to display it.

### Repository Structure

This spec lives under `docs/` in a monorepo shared by the gateway and all clients:

```
longmen/
├── docs/
│   └── GATEWAY_PROTOCOL.md      # this file — shared contract
├── gateway/                     # Python async daemon (source of truth)
│   ├── pyproject.toml
│   ├── gateway.toml
│   └── src/assistant/gateway/
└── clients/
    ├── terminal/                # Python CLI
    └── desktop/                 # C++/Qt GUI
```

All components implement against this single spec. Protocol changes are committed alongside the code changes they require.

---

## Ownership Model

| Concern | Owner | Rationale |
|---------|-------|-----------|
| Projects (root_path, settings) | Gateway | Projects are directories on the gateway's host. Tools execute there. |
| Agent definitions | Gateway | Enables headless scheduling, cron, heartbeat without a client. |
| Agent prompt files | Gateway | Stored alongside agent config in the gateway's data dir. |
| Conversation history | Gateway | Agents within a project can see shared context. |
| Permissions (stored approvals) | Gateway | Persisted per-project across sessions. |
| Tool execution | Gateway | Commands run on the gateway's host filesystem. |
| Request queue | Gateway | Serializes model access across all clients. |
| Model backends | Gateway | API keys and endpoints are server-side secrets. |
| Display state, UI preferences | Client | How to render is the client's business. |
| Project/agent selection | Client | Client picks what to work on; gateway enforces scope. |

Within a project, agents are **not isolated** from each other. They share the project's `root_path` filesystem. Agent A writes a file; Agent B can read it. This is by design — agents collaborate on the same codebase.

Between projects, isolation is strict. An agent in project X cannot access project Y's files. The gateway enforces this by scoping tool execution to the active project's `root_path`.

---

## Connection Lifecycle

```
Client                          Gateway
  |                                |
  |--- WS connect to /ws -------->|
  |                                |-- check auth.mode
  |<-- session_start --------------|   (open → accept immediately)
  |                                |   (paired → validate token)
  |--- project_list -------------->|   (what projects exist?)
  |<-- project_registry ----------|   (list of projects + agents)
  |                                |
  |--- project_select ------------>|   (pick a project to work in)
  |<-- project_context ------------|   (project details + agents + last_session)
  |<-- file_index -----------------|   (gitignore-aware project file list)
  |                                |
  |  If last_session is non-null:  |
  |--- session_resume ------------>|   (reattach to previous session)
  |<-- session_resumed ------------|   (session_id, incomplete_turn, recovered_from_disk)
  |<-- session_history ------------|   (user/assistant turns for display)
  |                                |
  |--- prompt -------------------->|
  |                                |-- if model busy:
  |<-- queue_position { pos: 2 } --|     enqueue, notify position
  |<-- queue_position { pos: 1 } --|     position update as queue drains
  |                                |-- model free, dequeue:
  |<-- plan_status (discovery) ----|   (complex requests only)
  |<-- tool_output (tree/grep) ----|   (discovery phase tool calls)
  |<-- plan_status (planned) ------|   (plan created)
  |<-- plan_status (running) ------|   (step 1 starts)
  |<-- stream_chunk (thinking) ----|
  |<-- stream_chunk (text) --------|
  |<-- approval_request -----------|   (tool call needs approval)
  |--- approval_response -------->|
  |<-- tool_output ----------------|
  |<-- stream_chunk (text) --------|   (model reasons about result)
  |<-- plan_status (completed) ----|   (step 1 done)
  |<-- stream_end -----------------|
  |                                |
  |--- prompt -------------------->|   (history preserved in session)
  |    ...                         |
  |                                |
  |--- project_select ------------>|   (switch to different project)
  |<-- project_context ------------|
  |<-- file_index -----------------|
  |                                |
  |--- WS close ------------------>|   (session preserved for reconnect)
```

### Transport Details

- **WebSocket URL:** `ws://{host}:{port}/ws` (or `wss://` if TLS is configured)
- **Subprotocol:** none (plain WebSocket, JSON text frames)
- **Auth token (paired mode):** sent as query parameter `ws://{host}:{port}/ws?token=eyJ...` on connect
- **Message framing:** each WebSocket text frame contains exactly one JSON envelope
- **Ping/pong:** the gateway sends WebSocket pings every 30 seconds. Clients must respond with pong (most libraries handle this automatically). If no pong is received within 10 seconds, the gateway closes the connection.

### HTTP Endpoints

The gateway listens on the same host/port for a small number of plain-HTTP `GET` requests (handled before the WebSocket upgrade):

- **`GET /health`** → `200 OK` with body `ok\n`. A liveness probe for container healthchecks (e.g. Docker `HEALTHCHECK`, compose). Requires no auth and does not open a session. Any other non-`/ws` path is not a health endpoint and the connection is rejected.

This is an operational endpoint, not part of the client message contract — it does not affect the protocol version.

### Reconnect Behavior

Sessions are persisted to disk as messages are added. On reconnect (network drop, app crash, gateway restart):

1. Client connects and receives `session_start`
2. Client sends `project_select` for the same project
3. Gateway responds with `project_context` including `last_session: { session_id, last_active }` (or `null` if none)
4. If `last_session` is non-null, client sends `session_resume` with the session ID
5. Gateway responds with `session_resumed` (confirms session, flags `incomplete_turn` if the last message was a user prompt with no response)
6. Gateway sends `session_history` with all displayable turns so the client can render the conversation
7. Client continues with a `prompt` using the resumed `session_id`

If the session is no longer on disk (pruned or too old), the gateway responds with `error` code `session_not_found`. The client should start a fresh session in that case.

### Client Implementation Notes

**Accumulating tool call deltas:** When `stream_chunk` arrives with `role: "tool_call"`, the `delta` is a fragment of JSON (e.g., `{"name":"shell","arguments":"{ \"command\":"`). The client should concatenate all `tool_call` deltas until either `approval_request` or `stream_end` arrives, then parse the accumulated JSON. The client may display the raw fragments as they arrive (for transparency) or wait for the complete tool call.

**Client-side state to cache** (for display continuity across reconnects):
- Current project selection
- Current session_id
- Agent list for the selected project (refresh on `config_reloaded`)

Conversation history is owned by the gateway — the client receives it via `session_history` on resume and does not need to persist it locally.

The client should NOT cache agent definitions, permissions, or project config as authoritative — those are owned by the gateway and may change between connections.

---

## Message Envelope

Every message in both directions uses this envelope:

```json
{
  "type": "string",
  "id": "uuid-v4",
  "timestamp": 1711900000000,
  "payload": { }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Message discriminator, used for routing |
| `id` | string | UUID v4. Responses reference the originating request's id via `ref_id` in payload where applicable |
| `timestamp` | int | Unix epoch milliseconds |
| `payload` | object | Type-specific content, defined below |

---

## Client → Gateway Messages

### `prompt`

Send a user message. Scoped to the currently selected project.

```json
{
  "type": "prompt",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string | null",
    "content": [
      {
        "type": "text",
        "text": "Refactor the auth module to use RAII"
      }
    ],
    "agent_name": "string | null"
  }
}
```

`agent_name` is optional. When set to a named agent in the current project, the
gateway resolves that agent's system prompt, tool allow-list, and backend for this
prompt. When null or omitted, the prompt runs against the project default
(all tools, default `[model]` backend).

Content is a list of content blocks, following a multimodal format:

| Content type | Fields | Description |
|-------------|--------|-------------|
| `text` | `text: string` | Plain text prompt |
| `image` | `media_type: string, data: string` | Base64-encoded image (png, jpg, webp) |
| `file` | `filename: string, media_type: string, data: string` | Base64-encoded file attachment |
| `file_ref` | `path: string` | Project-relative path; gateway reads file from `root_path` |

If `session_id` is null, the Gateway creates a new session within the current project. If provided, the Gateway appends to the existing conversation history. A prompt sent without first selecting a project returns an error.

### `approval_response`

User responds to a tool approval request.

```json
{
  "type": "approval_response",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-approval_request",
    "decision": "yes | no | yes_session | yes_always | edit",
    "edited_command": "optional, only when decision is edit"
  }
}
```

### `abort`

Sent by the client to cancel the active agent loop for the current session.

```json
{
  "type": "abort",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string"
  }
}
```

The gateway cancels the running agent loop, saves any partial assistant response to
session history, appends a system message marking the interruption, and sends
`stream_end` with `aborted: true`. The session remains open; the client may send a
new `prompt` immediately after.

If no agent loop is active, the message is silently ignored.

### `command`

Slash commands that don't go through the model.

```json
{
  "type": "command",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "name": "compact | refresh | new | permissions | config | prompt",
    "args": {}
  }
}
```

### `pair_request`

Only accepted when `auth.mode = "paired"`.

```json
{
  "type": "pair_request",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "pairing_code": "482910",
    "device_info": {
      "hostname": "archbox",
      "os": "Linux",
      "client_type": "terminal | desktop | web",
      "client_version": "1.0.0"
    }
  }
}
```

### `project_list`

Request the list of all projects registered on the gateway.

```json
{
  "type": "project_list",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {}
}
```

### `project_select`

Select a project to work in. All subsequent prompts, agent invocations, and commands are scoped to this project until another `project_select` is sent.

```json
{
  "type": "project_select",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "project_id": "my-cpp-app"
  }
}
```

### `project_upsert`

Create or update a project. The `root_path` must be a directory on the gateway's host filesystem.

```json
{
  "type": "project_upsert",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "project_id": "my-cpp-app",
    "project": {
      "description": "C++ application with CMake build",
      "root_path": "/home/user/projects/my-cpp-app",
      "context_file": "PROJECT.md"
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `root_path` | Absolute path on the gateway host. All tool execution is scoped to this directory. |
| `context_file` | Optional. Filename relative to `root_path` that the gateway auto-loads into the system prompt. Defaults to `PROJECT.md`. |

### `project_delete`

Remove a project from the gateway. Does NOT delete files from `root_path` — only removes the project config, agents, and session history from the gateway's data store.

```json
{
  "type": "project_delete",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "project_id": "my-cpp-app"
  }
}
```

### `agent_upsert`

Create or update an agent within the currently selected project. The `system_prompt` is the full prompt text.

```json
{
  "type": "agent_upsert",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "name": "reviewer",
    "agent": {
      "description": "Senior code reviewer",
      "backend": "hf-large",
      "system_prompt": "You are a senior code reviewer...",
      "tools": ["read_file", "grep", "list_dir"]
    }
  }
}
```

### `agent_delete`

Remove an agent from the current project.

```json
{
  "type": "agent_delete",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "name": "reviewer"
  }
}
```

### `agent_list`

Request the agent registry for the current project. Gateway responds with `agent_registry`.

```json
{
  "type": "agent_list",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {}
}
```

### `session_resume`

Reattach to an existing session. Must be sent after `project_select`. The gateway responds with `session_resumed` followed immediately by `session_history`.

```json
{
  "type": "session_resume",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

If the session is not found, the gateway responds with `error` code `session_not_found`. If no project has been selected, it responds with `error` code `no_project_selected`.

---

## Gateway → Client Messages

### `session_start`

Sent immediately after WebSocket connection is accepted.

```json
{
  "type": "session_start",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "auth": "none | paired",
    "device_id": "string | null",
    "gateway_version": "1.0.0",
    "default_model": {
      "backend": "local",
      "model_name": "qwen3-32b"
    },
    "available_backends": ["local", "hf-large"],
    "capabilities": ["streaming", "tools", "images", "file_attachments"]
  }
}
```

### `project_registry`

Response to `project_list`. Contains all projects with their agents summarized.

```json
{
  "type": "project_registry",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-project_list",
    "projects": {
      "my-cpp-app": {
        "description": "C++ application with CMake build",
        "root_path": "/home/user/projects/my-cpp-app",
        "agents": ["reviewer", "tester"],
        "active_sessions": 1
      },
      "data-pipeline": {
        "description": "ETL pipeline for analytics",
        "root_path": "/home/user/projects/data-pipeline",
        "agents": ["processor"],
        "active_sessions": 0
      }
    }
  }
}
```

### `project_context`

Response to `project_select`. Full project details including all agent definitions. Also sent after `project_upsert`.

```json
{
  "type": "project_context",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-project_select-or-project_upsert",
    "project_id": "my-cpp-app",
    "description": "C++ application with CMake build",
    "root_path": "/home/user/projects/my-cpp-app",
    "context_file": "PROJECT.md",
    "agents": {
      "reviewer": {
        "description": "Senior code reviewer",
        "backend": "hf-large",
        "backend_model": "Qwen/Qwen3-235B-A22B",
        "tools": ["read_file", "grep", "list_dir"]
      },
      "tester": {
        "description": "Test suite writer",
        "backend": null,
        "backend_model": "qwen3-32b",
        "tools": ["read_file", "grep", "shell", "write_file"]
      }
    },
    "last_session": {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "last_active": 1711903600000
    }
  }
}
```

### `file_index`

Sent immediately after `project_context` in response to `project_select` or `project_upsert`. Contains a sorted, gitignore-aware list of all files under the project's `root_path`. **Not sent on `/refresh`.**

```json
{
  "type": "file_index",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-project_select-or-project_upsert",
    "project_id": "my-cpp-app",
    "files": ["README.md", "src/foo.py", "tests/test_auth.py"]
  }
}
```

| Field | Description |
|-------|-------------|
| `ref_id` | ID of the triggering `project_select` or `project_upsert` message. |
| `project_id` | The project whose files are indexed. |
| `files` | Sorted list of POSIX-style relative paths, gitignored files excluded. Empty list if the root is inaccessible. |

### `session_resumed`

Confirms a `session_resume` request. Sent immediately before `session_history`.

```json
{
  "type": "session_resumed",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "incomplete_turn": false,
    "recovered_from_disk": true
  }
}
```

| Field | Description |
|-------|-------------|
| `incomplete_turn` | `true` if the last message in the session is a user prompt with no assistant response (crash mid-generation). The client should inform the user their previous prompt was not answered. |
| `recovered_from_disk` | `true` if the session was loaded from the JSONL store; `false` if it was still in memory. |

### `session_history`

Sent immediately after `session_resumed`. Contains all displayable conversation turns so the client can render the history without storing it locally.

```json
{
  "type": "session_history",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "turns": [
      { "role": "user",      "content": "Refactor auth module",       "timestamp": 1711900010.0 },
      { "role": "assistant", "content": "I'll read it first.",         "timestamp": 1711900015.0 },
      { "role": "assistant", "content": "[Compacted history]:\n...",   "timestamp": 1711902000.0, "is_summary": true }
    ],
    "has_compacted_prefix": false
  }
}
```

| Field | Description |
|-------|-------------|
| `turns` | Ordered list of user/assistant turns. Tool calls, tool results, pruned messages, and compacted messages are excluded. |
| `has_compacted_prefix` | `true` if a compaction summary exists; the summary appears first in `turns` with `"is_summary": true`. |

Each turn has `role` (`"user"` or `"assistant"`), `content` (always a plain string — multimodal content blocks are reduced to their text parts), and `timestamp` (Unix seconds float). The compaction summary turn additionally carries `"is_summary": true`.

### `agent_registry`

Response to `agent_upsert`, `agent_delete`, or `agent_list`. Contains the full agent list for the current project with validation results.

```json
{
  "type": "agent_registry",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-originating-message",
    "project_id": "my-cpp-app",
    "agents": {
      "reviewer": {
        "description": "Senior code reviewer",
        "backend": "hf-large",
        "backend_model": "Qwen/Qwen3-235B-A22B",
        "tools": ["read_file", "grep", "list_dir"],
        "valid": true
      }
    },
    "errors": []
  }
}
```

Error codes: `backend_not_found`, `invalid_tool`, `invalid_name`, `prompt_empty`.

### `stream_chunk`

A single token or small batch of tokens from the model.

```json
{
  "type": "stream_chunk",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-originating-prompt",
    "delta": "string",
    "role": "text | thinking | tool_call"
  }
}
```

When `role` is `tool_call`, the `delta` contains a fragment of the structured tool call JSON. The client accumulates these fragments until `stream_end` or `approval_request` arrives.

### `stream_end`

Model finished generating for this turn.

```json
{
  "type": "stream_end",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-originating-prompt",
    "aborted": false,
    "finish_reason": "stop | length | abort | null",
    "tool_calls_made": 7,
    "duration_ms": 4820,
    "tokens_used": 1800,
    "tool_call_count": 7,
    "usage": {
      "prompt_tokens": 1420,
      "completion_tokens": 380,
      "context_budget": {
        "used": 1800,
        "limit": 32000
      }
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `aborted` | `true` when the loop was interrupted (e.g. via `abort`). |
| `finish_reason` | The last vLLM `finish_reason` for this agent loop: `"stop"` (model finished), `"length"` (max_tokens hit), `"abort"` (loop cancelled), or `null` if unavailable. Note: some models (e.g. Qwen3) return `"stop"` even when tool calls are present — the gateway handles this transparently. |
| `tool_calls_made` | Total number of tool calls executed across all loop iterations for this prompt. Useful for observability and debugging. |
| `duration_ms` | Wall-clock time for this agent loop in milliseconds. Omitted when not measured (e.g. some planner-emitted `stream_end` messages). |
| `tokens_used` | Tokens consumed by this loop (delta of session usage plus any tokens freed mid-loop). Omitted when not available. |
| `tool_call_count` | Alias of `tool_calls_made`, carried alongside `duration_ms`/`tokens_used` for the client's per-turn metrics line. Omitted when not available. |

### `approval_request`

Gateway needs user approval for a tool call.

```json
{
  "type": "approval_request",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "tool": "shell",
    "command": "pytest tests/auth/",
    "risk": "safe | moderate | destructive",
    "context": "Running test suite for auth module after refactor",
    "timeout_seconds": 300
  }
}
```

### `plan_status`

Sent during a discover → plan → execute loop to report progress. Only emitted for complex requests that trigger the planning phase; simple requests produce only `stream_chunk` and `stream_end`.

```json
{
  "type": "plan_status",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-originating-prompt",
    "step": 3,
    "total_steps": 6,
    "status": "discovery | planned | running | completed | failed | skipped",
    "description": "Add input validation for email and name fields",
    "summary": "Added email regex validation and required name check to POST /api/users"
  }
}
```

| Status | Meaning |
|--------|---------|
| `discovery` | Sent once when discovery phase starts (`step: 0, total_steps: 0`). Client can show "Exploring codebase...". |
| `planned` | Sent once when `create_plan` is called and the plan is accepted (`step: 0, total_steps: N`). |
| `running` | Sent at the start of each step execution. `summary` is empty. |
| `completed` | Sent when a step finishes successfully. `summary` is a one-line description of what was accomplished. |
| `failed` | Sent when a step exhausts its self-correction retries. `summary` describes the failure. |
| `skipped` | Reserved for future use. |

The full sequence for a two-step plan looks like:
```
plan_status { status: "discovery", step: 0, total_steps: 0 }
plan_status { status: "planned",   step: 0, total_steps: 2 }
plan_status { status: "running",   step: 1, total_steps: 2 }
  ... stream_chunk / tool_output messages for step 1 ...
plan_status { status: "completed", step: 1, total_steps: 2, summary: "..." }
plan_status { status: "running",   step: 2, total_steps: 2 }
  ... stream_chunk / tool_output messages for step 2 ...
plan_status { status: "completed", step: 2, total_steps: 2, summary: "..." }
stream_end
```

---

### `plan_revision`

Sent when the model calls the `revise_plan` tool during plan execution to modify remaining steps. This allows the agent to adapt the plan based on discoveries made during step execution.

```json
{
  "type": "plan_revision",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-originating-prompt",
    "revision_number": 1,
    "action": "add | remove | replace",
    "reason": "Why this revision is needed",
    "description": "Human-readable summary of what changed",
    "revised_plan": [
      { "step": 1, "status": "completed", "description": "✓ Fix email verification module" },
      { "step": 2, "status": "pending", "description": "  Add tests for email verification" },
      { "step": 3, "status": "pending", "description": "  Update API documentation" }
    ]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `revision_number` | int | Sequential revision counter, starting at 1. Incremented for each `revise_plan` call. |
| `action` | string | The revision action taken: `"add"` (inserted new steps), `"remove"` (deleted steps), or `"replace"` (replaced all future steps). |
| `reason` | string | Why the revision was needed. The model should reference what it discovered during execution. |
| `description` | string | Human-readable summary of what changed (e.g., "Added 1 step after step 1", "Removed step 3", "Replaced 3 future steps with 2 new steps"). |
| `revised_plan` | array | The complete plan after revision. Each item has `step` (1-indexed), `status` ("completed" | "in_progress" | "pending" | "failed" | "skipped"), and `description` (with status marker prefix). |

**Revision lifecycle:**

1. Model calls `revise_plan` tool with `action`, `reason`, and action-specific parameters (`insert_after` + `new_steps` for `add`, `steps_to_remove` for `remove`, `new_steps` for `replace`)
2. Gateway validates the revision (checks revision limit, ensures only future steps are modified)
3. Gateway updates `PlanExecution.steps` and records a `PlanRevision` object
4. Gateway sends `plan_revision` message to client
5. Gateway sends updated `plan_status` messages for all steps (with renumbered step numbers)

**Revision limits:**

By default, a maximum of 5 revisions are allowed per plan (configurable via `max_revisions` in `[planning]` section of `gateway.toml`). When the limit is reached, the `revise_plan` tool returns an error.

**Client behavior:**

Upon receiving `plan_revision`:
- Display the revision to the user (e.g., "Plan revised: Added 1 step after step 1")
- Show the reason the model provided
- Update the plan display with the revised steps
- Continue streaming as normal

The full sequence for a plan with one revision looks like:
```
plan_status { status: "planned", step: 0, total_steps: 3 }
plan_status { status: "running", step: 1, total_steps: 3 }
  ... step 1 execution ...
plan_revision { revision_number: 1, action: "add", reason: "Discovered existing module", revised_plan: [...] }
plan_status { status: "running", step: 1, total_steps: 4 }
  ... step 1 continues with revised plan ...
plan_status { status: "completed", step: 1, total_steps: 4, summary: "..." }
plan_status { status: "running", step: 2, total_steps: 4 }
  ... step 2 execution ...
plan_status { status: "completed", step: 2, total_steps: 4, summary: "..." }
stream_end
```

### `tool_output`

Result of executing a tool.

```json
{
  "type": "tool_output",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-approval_request-or-auto-approved-tool",
    "tool": "shell",
    "command": "pytest tests/auth/",
    "stdout": "...",
    "stderr": "...",
    "exit_code": 0,
    "truncated": false,
    "duration_ms": 2340
  }
}
```

### `pair_challenge`

Gateway asks admin to confirm a pairing request (sent to admin interface, not the requesting client).

```json
{
  "type": "pair_challenge",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "device_info": { "hostname": "archbox", "os": "Linux", "client_type": "terminal" }
  }
}
```

### `pair_result`

Sent to the requesting client after admin approves/rejects.

```json
{
  "type": "pair_result",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-pair_request",
    "success": true,
    "token": "eyJ...",
    "error": null
  }
}
```

### `error`

```json
{
  "type": "error",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-originating-message | null",
    "code": "auth_required | token_invalid | session_not_found | no_project_selected | project_not_found | vllm_unavailable | vllm_timeout | vllm_error | timeout | context_overflow | context_warning | internal",
    "message": "Human-readable error description",
    "recoverable": true
  }
}
```

`code` is a free-form string. Notable model-related codes:

- `vllm_unavailable` — the model server was unreachable (connect failed, or it did not come online within `vllm_startup_wait`). `recoverable: true`.
- `vllm_timeout` — the connection was accepted but no first token arrived within `vllm_first_token` (model still loading, or prompt too large). `recoverable: true`.
- `vllm_error` — the model server returned an HTTP error status. `recoverable` varies by status class: `5xx`/`408`/`429` (loading, overloaded, rate-limited) are `recoverable: true`; other `4xx` (esp. `400`, where the request itself was rejected — prompt exceeds the model's limit, malformed body) are `recoverable: false`.

### `model_waiting`

Gateway→client only. Sent when the prompt is the **active** request (not behind another in the queue) but the gateway is waiting for the model server to come online. Distinct from `queue_position` ("waiting behind another request"). One message is sent per failed readiness re-probe, with a rising `waited_seconds`, so the client can show a live "waiting Ns / max" indicator. A healthy model emits **zero** `model_waiting` messages.

```json
{
  "type": "model_waiting",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-waiting-prompt",
    "waited_seconds": 3,
    "max_wait_seconds": 120
  }
}
```

### `queue_position`

Sent when a prompt is enqueued because the model is busy.

```json
{
  "type": "queue_position",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-queued-prompt",
    "position": 2,
    "active_session_id": "string | null"
  }
}
```

### `status`

Periodic or on-demand status update.

```json
{
  "type": "status",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "state": "idle | queued | streaming | awaiting_approval | executing_tool",
    "queue_position": 0,
    "token_usage": {
      "used": 14000,
      "limit": 32000,
      "breakdown": {
        "system_prompt": 1200,
        "pinned_first": 1800,
        "compacted_summary": 900,
        "active_context": 6100,
        "pinned_recent": 2800,
        "response_reserve": 4096,
        "available": 15104
      }
    },
    "duration_ms": 4820,
    "tokens_used": 1800,
    "tool_call_count": 7
  }
}
```

The `breakdown` field is included when the context manager is active (i.e., after any prompt in a session). It shows where tokens are allocated across the context budget. `available` is the remaining space after all allocations and reserves.

`duration_ms`, `tokens_used`, and `tool_call_count` are optional, included when the gateway has per-turn metrics to report (e.g. after a completed agent loop). Clients use them to render a per-turn metrics line; each is omitted when unavailable.

### `command_result`

Response to a slash command. Sent after the gateway processes `/refresh`, `/new`, `/permissions`, `/config`, or `/compact`.

```json
{
  "type": "command_result",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "ref_id": "id-of-the-command",
    "name": "refresh",
    "success": true,
    "message": "Context file reloaded (PROJECT.md, 1240 tokens)",
    "data": {}
  }
}
```

The `data` field is command-specific:
- `/new` — `{ "session_id": "..." }` — the new session's ID
- `/permissions` — `{ "persistent": {...}, "session": {...}, "defaults": {...} }`
- `/config` — the current session config dict (see Session Config below)
- `/compact` — `{ "tokens_before": 24000, "tokens_after": 8000, "tokens_freed": 16000, "breakdown": { ... } }` — token counts before and after compaction, with the post-compact budget breakdown
- `/prompt` — `{ "system_prompt": "..." }` — the current effective system prompt that will be sent to the model

For the `/prompt` command, the gateway may also return an `error` message in these cases:
- `no_session` — No active session exists to retrieve the system prompt from
- `internal` — System prompt not yet available for this session (no prompt has been sent yet)

#### Session Config

The `/config` command reads and writes the session-scoped configuration. All fields are optional when setting — unset fields inherit their defaults.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_plan` | bool | `true` | Whether complex requests trigger the discover → plan → execute loop. Set to `false` to always go directly to the agent loop. |
| `discovery_budget_pct` | float | `0.30` | Maximum fraction of the context limit to spend on the discovery phase (e.g. `0.30` = 30%). The gateway nudges the model toward planning at 80% of this budget and blocks exploration tools at 100%. |
| `max_retries` | int | `3` | Maximum self-correction attempts per build/test failure before marking the step as failed. |
| `triage_max_reads` | int | `10` | Maximum read-only tool calls during a single triage phase. After this limit the gateway nudges the model to apply a fix. |
| `verbosity` | string | `"ask_ambiguous"` | Clarification behavior in interactive mode: `"ask_always"` (ask for any ambiguity), `"ask_ambiguous"` (only genuinely ambiguous requests), `"best_guess"` (make best inference, still ask for destructive actions). Ignored in workflow mode. |
| `diff_mode` | bool | `false` | When `true`, file-write tools send a `diff_preview` for approval before applying. |
| `auto_prune` | bool | `null` | Override the gateway-level `prune_threshold` for this session. `null` inherits from `gateway.toml`. |
| `auto_compact` | bool | `null` | Override the gateway-level `auto_compact` setting for this session. `null` inherits from `gateway.toml`. |
| `pin_first_tokens` | int | `null` | Override the token budget pinned at the start of conversation. `null` inherits from `gateway.toml`. |
| `pin_recent_tokens` | int | `null` | Override the token budget pinned at the end of conversation. `null` inherits from `gateway.toml`. |
| `compact_target_tokens` | int | `null` | Override the target summary size after compaction. `null` inherits from `gateway.toml`. |

### `diff_preview`

Sent when `diff_mode=true` is set and a file write tool is called. The client displays the diff and prompts for approval before applying.

```json
{
  "type": "diff_preview",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "session_id": "string",
    "ref_id": "id-of-the-tool-call",
    "path": "src/assistant/gateway/auth.py",
    "diff": "--- a/src/assistant/gateway/auth.py\n+++ b/src/assistant/gateway/auth.py\n@@ -45,8 +45,10 @@\n...",
    "stats": { "additions": 10, "deletions": 8 }
  }
}
```

The client responds with `approval_response` using `ref_id` pointing to the `diff_preview` message id.

### `config_reloaded`

Broadcast to all connected clients when a config file changes on disk. Clients that have the affected project selected should refresh their local state (e.g., re-display the agent list).

```json
{
  "type": "config_reloaded",
  "id": "...",
  "timestamp": 1711900000000,
  "payload": {
    "scope": "gateway | project",
    "project_id": "my-cpp-app | null",
    "changed": ["agents", "permissions", "project_settings"],
    "message": "Agent 'reviewer' prompt updated"
  }
}
```

| Field | Description |
|-------|-------------|
| `scope` | `"gateway"` if `gateway.toml` changed (backends, timeouts, etc). `"project"` if a project's config changed. |
| `project_id` | Which project was affected. Null when scope is `"gateway"`. |
| `changed` | List of what changed. Clients can decide whether to refresh based on this. |

After receiving `config_reloaded`, a client can send `project_select` (for the same project) to get a fresh `project_context`, or `agent_list` to refresh just the agents.

---

## Configuration

### `gateway.toml` (server config — lives alongside the gateway binary)

```toml
[server]
host = "0.0.0.0"                 # requires restart to change
port = 8420                      # requires restart to change
log_level = "info"               # hot-reloadable
data_dir = "/var/lib/longmen/gateway"  # requires restart to change

[auth]
mode = "open"                    # hot-reloadable (existing connections unaffected)
pairing_code_ttl = 300           # hot-reloadable
token_lifetime_days = 90         # hot-reloadable

[model]
# Default backend — used for interactive sessions and agents without a backend override.
vllm_base_url = "http://localhost:8000"   # hot-reloadable (new connections use new URL)
model_name = "qwen3-32b"                  # hot-reloadable
api_key = ""                              # hot-reloadable
temperature = 0.7                         # hot-reloadable
top_p = 0.95                              # hot-reloadable
max_tokens = 32000                        # hot-reloadable; set high enough for thinking-mode models (Qwen3)
context_limit = 32000                     # hot-reloadable
tokenizer_path = "/path/to/model/tokenizer.json"  # REQUIRED, hot-reloadable; local tokenizer.json (offline — the gateway never downloads tokenizers)

# Named backends — agents reference these by key.
# Hot-reloadable: adding/removing/modifying backends takes effect on next agent invocation.
# [model.backends.hf-large]
# vllm_base_url = "https://api-inference.huggingface.co/v1"
# model_name = "Qwen/Qwen3-235B-A22B"
# api_key = "hf_..."
# context_limit = 131000

[permissions]
workflow_mode = "allow_all"      # hot-reloadable
default_safe = ["read_file", "list_dir", "grep", "tree", "symbols", "git_status", "git_diff", "git_log", "web_search", "web_fetch", "rag_search"]
default_destructive = ["rm", "git_push_force", "drop_table", "truncate"]

[timeouts]
tool_execution = 120             # hot-reloadable
approval_wait = 300              # hot-reloadable
vllm_request = 60                # hot-reloadable
vllm_startup_wait = 120          # hot-reloadable; max seconds to wait for the model to come online before sending vllm_unavailable
vllm_first_token = 300           # hot-reloadable; max seconds to wait for the first streamed token before sending vllm_timeout

[web]
# Web tools configuration — all hot-reloadable.
search_enabled = true            # enable/disable web_search tool
fetch_enabled = true             # enable/disable web_fetch tool

# API key for Brave Search. Prefer using the BRAVE_API_KEY environment variable
# instead of storing the key in this file to avoid accidental commits.
brave_api_key = ""

# Number of search results to return (passed to Brave API as 'count')
search_count = 5

# Timeout for fetching web pages (seconds)
fetch_timeout = 15

# Maximum number of HTTP redirects to follow
fetch_max_redirects = 5

# Domains blocked from web_fetch (web_search results are not filtered)
# Supports exact domain match and wildcard prefix (*.example.com)
fetch_blocked_domains = []

# User-Agent string for HTTP requests
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

[context]
# Context window management — all hot-reloadable.
pin_first_tokens = 2000          # token budget pinned at the start of conversation (task definition)
pin_recent_tokens = 3000         # token budget pinned at the end of conversation (current work)
reserved_response_tokens = 4096  # reserved for the model's response
prune_threshold = 0.75           # utilization at which stale tool outputs are pruned (no model call)
compact_threshold = 0.85         # utilization at which the middle is compacted (model summarises)
warn_threshold = 0.95            # utilization at which a context_warning error is sent to the client
compact_target_tokens = 1000     # target size for the compaction summary

[rag]
# RAG service integration — all hot-reloadable.
enabled = false                        # enable/disable rag_search tool
base_url = "http://127.0.0.1:8421"    # RAG service base URL
timeout = 30.0                         # request timeout in seconds
top_k = 10                             # results requested per search call
context_budget_threshold = 60000       # budget.used above which RAG is refused

[sessions]
max_sessions_per_project = 50   # keep only N most recent sessions per project
max_session_age_days = 30       # delete sessions older than this
```

### Gateway Data Directory

The gateway stores all project config, agents, and persistent state in `data_dir`. All files in this directory are watched for changes and hot-reloaded automatically.

```
{data_dir}/
├── projects/
│   ├── my-cpp-app/
│   │   ├── project.toml         # project metadata
│   │   ├── agents.toml          # agent definitions (prompts stored separately)
│   │   ├── prompts/
│   │   │   ├── reviewer.md      # system prompt for "reviewer" agent
│   │   │   └── tester.md        # system prompt for "tester" agent
│   │   ├── permissions.toml     # stored "yes, always" approvals
│   │   └── sessions/            # persisted conversation history
│   │       ├── {session_id}.jsonl       # append-only message log
│   │       └── {session_id}.meta.json   # session metadata (atomic rewrite)
│   └── data-pipeline/
│       ├── project.toml
│       ├── agents.toml
│       └── prompts/
│           └── processor.md
```

**`project.toml`** — project metadata:
```toml
description = "C++ application with CMake build"
root_path = "/home/user/projects/my-cpp-app"
context_file = "PROJECT.md"

# Optional: bind RAG collections for this project (requires [rag] enabled in gateway.toml)
[rag]
collections = ["godot-docs", "project-memories"]
```

**`agents.toml`** — agent definitions (system prompts in separate .md files):
```toml
[reviewer]
description = "Senior code reviewer"
backend = "hf-large"
prompt_file = "prompts/reviewer.md"    # relative to project data dir
tools = ["read_file", "grep", "list_dir"]

[tester]
description = "Test suite writer"
# no backend → uses default [model]
prompt_file = "prompts/tester.md"
tools = ["read_file", "grep", "shell", "write_file"]
```

**`prompts/reviewer.md`** — system prompt as a standalone markdown file:
```markdown
You are a senior code reviewer working on a C++ codebase.

Analyze code for:
- Memory safety issues (use-after-free, leaks, dangling pointers)
- Performance problems (unnecessary copies, cache misses)
- Style violations against the project conventions

Be specific. Reference line numbers. Suggest fixes.
```

**`permissions.toml`** — stored "yes, always" approvals:
```toml
[rules]
"pytest tests/**" = "allow"
"cargo build *" = "allow"
```

**Design rationale:**
- TOML for all config: the gateway is Python, TOML is stdlib (`tomllib`). Human-readable, comment-friendly, clean key-value structure.
- Markdown for prompts: comfortable to write and edit, clean git diffs, no escaping issues with multi-line text.
- The gateway reads and writes these files. Clients CRUD through WebSocket messages, never touch the filesystem directly.
- Agent `backend` references a key in `[model.backends.*]` from `gateway.toml`. If omitted, the default `[model]` is used.

---

## Hot Reload

The gateway watches `gateway.toml` and the entire `data_dir` using inotify (via `watchfiles` or `watchdog`). When a file changes on disk, the gateway reloads the affected config without restarting.

### Reload behavior by file type

| File changed | What happens | In-flight sessions |
|-------------|-------------|-------------------|
| `gateway.toml` — model, auth, timeouts, permissions | Config reloaded. New vLLM client instances created for changed backends. | Active agent loops finish with old config. Next invocation uses new config. |
| `gateway.toml` — host, port, data_dir | Warning logged: "requires restart". No reload. | Unaffected. |
| `project.toml` | Project metadata updated in memory. | Unaffected (root_path change takes effect on next tool call). |
| `agents.toml` | Agent registry reloaded for that project. | Active agent loops finish with old agent config. Next invocation uses new config. |
| `prompts/*.md` | Prompt text reloaded for the affected agent. | Active agent loops finish with old prompt. Next invocation uses new prompt. |
| `permissions.toml` | Permission rules reloaded. | Next tool call uses new rules. |

### Reload flow

1. File watcher detects a change (create, modify, delete)
2. Gateway determines the scope: `gateway.toml` → gateway-level, `projects/{id}/*` → project-level
3. Gateway re-reads and validates the changed file(s)
4. If validation fails, the old config is kept and an error is logged (no crash, no partial state)
5. If validation succeeds, the in-memory config is swapped atomically
6. Gateway broadcasts `config_reloaded` to all connected clients with the affected `scope` and `project_id`
7. For backend changes: existing `vllm_client` instances are replaced; the old ones are kept alive until their active requests complete

### Debouncing

File watchers fire rapidly during a save (temp file → rename, or multiple writes). The gateway debounces reloads with a 500ms window — after the first change event, it waits 500ms of quiet before reloading. This prevents partial reads during multi-file saves.

---

## Design Constraints

1. **Client-agnostic**: The Gateway must never assume what kind of client is connected. No terminal escape codes, no HTML, no widget-specific data. Structured JSON only.

2. **Gateway is the source of truth**: Projects, agents, sessions, permissions, and all persistent state are owned by the Gateway. Clients are stateless displays — they select, they input, they render. They don't store anything the gateway couldn't reconstruct.

3. **Project scoping**: All prompts, tool executions, and agent invocations are scoped to the currently selected project. A client must send `project_select` before sending prompts. Tool execution is sandboxed to the project's `root_path`.

4. **Agent non-isolation within projects**: Agents within the same project share the project's filesystem. Agent A's writes are visible to Agent B. This enables collaborative workflows (developer writes code, tester tests it, reviewer reviews it — all seeing the same files). Cross-project access is forbidden.

5. **Stateful sessions**: The Gateway maintains conversation history per session, scoped to a project, and persists it to disk as messages are added. On reconnect, clients send `session_resume`; the gateway responds with `session_resumed` and `session_history` so the client can render the conversation without caching it locally. Sessions are pruned by age and count at gateway startup.

6. **Multimodal input**: Images and files are sent as base64 in content blocks. The Gateway is responsible for formatting these into the vLLM request.

7. **Streaming**: All model output is streamed token-by-token. The Gateway never buffers a full response before sending.

8. **Tool execution happens on the Gateway's host**, not the client's machine. The client is purely a display and input device.

9. **Workflow mode**: When `execution_mode` is `"workflow"`, all tool calls auto-execute without approval. The execution mode is set by the Gateway internally and cannot be spoofed by clients.

10. **Request queue**: The Gateway maintains a FIFO request queue. Only one agent loop runs at a time. When the model is busy, incoming prompts are enqueued and clients receive `queue_position` updates. The queue is shared across all backends.

11. **Backend resolution is per-agent.** The default `[model]` config is used for interactive sessions. Each agent can specify a `backend` key referencing a `[model.backends.*]` entry. Named backends inherit unset fields from `[model]`. The `vllm_client.py` module is instantiated per-backend so HTTP connection pools are reused. When `api_key` is non-empty, send it as `Authorization: Bearer`.

12. **Future-ready for headless operation**: Because the gateway owns agents and project config, it can run agent tasks without a client connected — scheduled reviews, periodic builds, heartbeat checks. The protocol supports this: the gateway runs the agent loop, stores results in the session, and the client reads them on next connect.

13. **Hot reload via file watcher**: The gateway watches `gateway.toml` and `data_dir` for changes using inotify. Config changes are reloaded without restart (except host, port, data_dir which require restart). Active agent loops are unaffected — new config takes effect on the next invocation. Connected clients receive `config_reloaded` so they can refresh their state. Reloads are debounced (500ms) to handle multi-file saves cleanly.
