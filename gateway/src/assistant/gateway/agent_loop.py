"""Core loop: prompt → model → tool_call → execute → model."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import mimetypes
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from .context_manager import ContextManager, ContextOverflowError
from .protocol import make_stream_chunk, make_stream_end, make_tool_output
from .session import TrackedMessage, count_tokens
from .tools import execute_tool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from .compactor import Compactor
    from .config import GatewayConfig
    from .permissions import PermissionManager
    from .session import Session


class ExecutionMode(Enum):
    """Execution context for the agent loop."""

    INTERACTIVE = "interactive"  # user is in the loop, can ask questions
    AGENT = "agent"  # agent invocation, semi-autonomous
    WORKFLOW = "workflow"  # pipeline step, fully autonomous


log = structlog.get_logger(__name__)

# Codebase navigation strategy injected into every system prompt.
_CODEBASE_NAV_PROMPT = """
## Autonomous operation

Work through the task completely without stopping to check in. Do not generate
intermediate progress reports, summaries, or questions mid-task. Only produce
a final response when the task is fully done or you have hit an unresolvable
blocker that requires user input. If you need to make multiple tool calls,
make them — do not stop to narrate what you are about to do.

## Codebase navigation strategy

When exploring a codebase, follow this workflow:
1. Start with `tree` to understand the project structure
2. Use `grep` to find specific patterns, function calls, or error messages
3. Use `symbols` to see a file's structure before reading it
4. Use `read_file` with line ranges to read specific functions or sections
5. Avoid reading entire large files — use symbols + line ranges instead

The context budget is limited. Check the token usage in status messages.
Prefer targeted reads over full file reads. A 3000-line file can consume
half the context window — always use `symbols` first to find the right section.

## File and directory deletion

Always use `delete_tool` to delete files and directories — never use shell
commands (`rm`, `rmdir`, `find -delete`, etc.). Do not attempt to delete
any file or folder outside the project root under any circumstances, even
if the user explicitly asks you to.
""".strip()

# Fraction of context budget at which the nav hint becomes more urgent
_BUDGET_WARNING_THRESHOLD = 0.80

_SEARCH_DEDUP_TOOLS = frozenset({"rag_search", "web_search"})


def _format_content_for_vllm(
    content_blocks: list[Any],
    root_path: Path | None = None,
) -> Any:
    """Convert protocol content blocks to vLLM message content format.

    - TextContent → plain text (or text part in multimodal list)
    - ImageContent → image_url with data URI
    - FileContent text/* → fenced code block inlined into text
    - FileContent image/* → image_url with data URI

    If all blocks are text-only (no images), returns a plain string.
    Otherwise returns a list of vLLM content parts.
    """
    has_non_text = False

    # Pre-process: convert file text/* to text blocks
    processed: list[dict[str, Any]] = []
    for block in content_blocks:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

        if btype == "file":
            if isinstance(block, dict):
                media_type = block.get("media_type", "")
                data = block.get("data", "")
                filename = block.get("filename", "file")
            else:
                media_type = block.media_type
                data = block.data
                filename = block.filename

            if media_type.startswith("text/"):
                # Decode and inline as fenced code block
                decoded = base64.b64decode(data).decode("utf-8", errors="replace")
                ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                fenced = f"File: {filename}\n```{ext}\n{decoded}\n```"
                processed.append({"type": "text", "text": fenced})
            elif media_type.startswith("image/"):
                # Treat as image
                has_non_text = True
                processed.append(
                    {
                        "type": "image",
                        "media_type": media_type,
                        "data": data,
                    }
                )
            else:
                # Unsupported binary — include as note
                processed.append(
                    {
                        "type": "text",
                        "text": f"[Unsupported attachment: {filename} ({media_type})]",
                    }
                )
        elif btype == "file_ref":
            path_str: str = str(block.get("path", "") if isinstance(block, dict) else block.path)
            if root_path is None:
                processed.append({"type": "text", "text": f"[file_ref: no root_path — {path_str}]"})
                continue
            abs_path = root_path / path_str
            # Sandbox check — reject path traversal outside root_path
            try:
                abs_path.resolve().relative_to(root_path.resolve())
            except ValueError:
                processed.append(
                    {"type": "text", "text": f"[file_ref: path escapes project root — {path_str}]"}
                )
                continue
            try:
                raw = abs_path.read_bytes()
            except OSError as e:
                processed.append({"type": "text", "text": f"[Could not read {path_str}: {e}]"})
                continue
            media_type, _ = mimetypes.guess_type(str(abs_path))
            media_type = media_type or "application/octet-stream"
            filename = abs_path.name
            if media_type.startswith("text/"):
                decoded = raw.decode("utf-8", errors="replace")
                ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                fenced = f"File: {path_str}\n```{ext}\n{decoded}\n```"
                processed.append({"type": "text", "text": fenced})
            elif media_type.startswith("image/"):
                has_non_text = True
                data = base64.b64encode(raw).decode()
                processed.append({"type": "image", "media_type": media_type, "data": data})
            else:
                processed.append(
                    {"type": "text", "text": f"[Unsupported attachment: {path_str} ({media_type})]"}
                )
        else:
            if btype == "image":
                has_non_text = True
            if isinstance(block, dict):
                processed.append(block)
            else:
                processed.append(block.model_dump())

    if not has_non_text:
        # All text — join into single string
        texts = []
        for block in processed:
            if block.get("type") == "text":
                texts.append(block["text"])
        return "\n\n".join(texts) if len(texts) > 1 else (texts[0] if texts else "")

    # Multimodal — build parts list
    vllm_parts: list[dict[str, Any]] = []
    for block in processed:
        btype = block.get("type")
        if btype == "text":
            vllm_parts.append({"type": "text", "text": block["text"]})
        elif btype == "image":
            media_type = block.get("media_type", "image/png")
            data = block.get("data", "")
            vllm_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }
            )
    return vllm_parts


def _tool_result_content(result: dict[str, Any]) -> str:
    """Format tool result as a string for the tool message (model context)."""
    if "error" in result:
        return f"error: {result['error']}"
    if "stdout" in result or "stderr" in result:
        parts = []
        if result.get("stdout"):
            parts.append(f"stdout: {result['stdout']}")
        if result.get("stderr"):
            parts.append(f"stderr: {result['stderr']}")
        parts.append(f"\nexit_code: {result.get('exit_code', 0)}")
        return "\n".join(parts)
    # Rich tool results: pick the primary output field in priority order
    for key in ("content", "tree", "symbols", "matches"):
        if key in result:
            return str(result[key])
    return json.dumps(result)


def _result_stdout(result: dict[str, Any]) -> str:
    """Flatten a tool result to a single string for display in tool_output.stdout.

    Shell tools already have stdout/stderr. File/codebase tools return rich dicts
    with keys like 'tree', 'symbols', 'content', 'matches'. We pick the primary
    field so the terminal client always has something to display.
    """
    if "error" in result:
        return f"error: {result['error']}"
    if "stdout" in result:
        return str(result["stdout"])
    for key in ("content", "tree", "symbols", "matches"):
        if key in result:
            return str(result[key])
    return json.dumps(result, indent=2)


def _build_tool_metadata(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from tool arguments for staleness detection."""
    metadata: dict[str, Any] = {}
    # File path — used to detect stale reads and writes
    for key in ("path", "file_path", "filename"):
        if key in arguments:
            metadata["path"] = arguments[key]
            break
    # Command — used to classify build/test output
    if "command" in arguments:
        metadata["command"] = arguments["command"]
    if tool_name == "rag_search" and "query" in arguments:
        metadata["query"] = arguments["query"]
    return metadata


async def run_agent_loop(
    session: Session,
    user_content: list[Any],
    ws: Any,
    vllm_client: Any,
    config: GatewayConfig,
    ref_id: str,
    root_path: str,
    permission_manager: PermissionManager,
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    context_manager: ContextManager | None = None,
    compactor: Compactor | None = None,
    execution_mode: ExecutionMode = ExecutionMode.INTERACTIVE,
    blocked_tools: frozenset[str] | None = None,
    blocked_tool_message: str = "",
    tool_interceptors: (
        dict[str, Callable[..., Coroutine[Any, Any, dict[str, Any]]]] | None
    ) = None,
    on_tool_result: (Callable[[str, dict[str, Any], dict[str, Any]], None] | None) = None,
    get_blocked_tools: (Callable[[], tuple[frozenset[str], str] | None] | None) = None,
    get_system_prompt: (Callable[[], str | None] | None) = None,
    emit_stream_end: bool = True,
) -> tuple[int, int, int]:
    """Main agent loop: send prompt → stream response → handle tool calls → repeat.

    Returns (duration_ms, tokens_used, total_tool_calls) for this agent loop call.

    Parameters
    ----------
    blocked_tools:
        Tool names that are not allowed in this phase. Calling one returns an
        error result to the model without executing anything.
    blocked_tool_message:
        The message returned when a blocked tool is called.
    tool_interceptors:
        Map of tool_name → async callable(arguments) → dict.  When a tool
        has an interceptor the interceptor is called instead of execute_tool.
        If the returned dict contains ``{"__stop__": True}`` the loop exits
        immediately (without sending stream_end — the caller handles that).
    execution_mode:
        Context for clarification / approval behaviour (see ExecutionMode).
    on_tool_result:
        Optional callback invoked after each tool call with
        (tool_name, arguments, result). Use this for side-effect tracking
        (e.g. correction state) without modifying the loop itself.
    get_blocked_tools:
        Optional callable that returns the *current* (blocked_tools_frozenset,
        message) pair or None.  Called before each tool execution, allowing the
        caller to change which tools are blocked mid-loop (e.g. for dynamic
        self-correction state).  When set, takes precedence over blocked_tools.
    get_system_prompt:
        Optional callable returning a system prompt string (or None) that is
        called at the start of each loop iteration.  When set, takes precedence
        over the static ``system_prompt`` parameter, allowing the caller to
        inject dynamic content (e.g. budget nudges) into every model call.
    emit_stream_end:
        Whether to send a stream_end message when the loop exits normally
        (no tool calls).  When False, the loop still adds the assistant
        message to the session and exits cleanly, but skips sending stream_end.
        This is used by the planner to prevent multiple stream_end messages
        during multi-step plan execution.
    """

    # Record baseline for per-loop duration and token delta
    tokens_before = session.tokens_used
    loop_start = time.time()

    # Format user content for vLLM
    vllm_content = _format_content_for_vllm(
        user_content, Path(root_path) if root_path else None
    )
    await session.add_user_message(vllm_content)

    loop_count = 0
    last_usage: dict[str, Any] | None = None
    total_tool_calls = 0
    tokens_freed_mid_loop = 0
    accumulated_text = ""
    _assistant_added = False
    last_search_query: dict[str, str] = {}

    try:
        while True:
            loop_count += 1
            log.debug("agent_loop.iteration", loop=loop_count, session_id=session.session_id)

            # Build system prompt: nav strategy + optional budget warning + caller prompt
            nav_parts: list[str] = [_CODEBASE_NAV_PROMPT]
            if session.context_pressure >= _BUDGET_WARNING_THRESHOLD:
                pct = int(session.context_pressure * 100)
                nav_parts.append(
                    f"Context budget is at {pct}%. "
                    "Prefer targeted reads (symbols + line ranges) over full file reads."
                )
            caller_prompt = get_system_prompt() if get_system_prompt is not None else system_prompt
            if caller_prompt:
                nav_parts.append(caller_prompt)
            effective_system_prompt = "\n\n".join(nav_parts)

            # Store in session for retrieval by /prompt command
            if session is not None:
                session.last_system_prompt = effective_system_prompt

            # Build messages list for vLLM
            system_message: dict[str, Any] = {"role": "system", "content": effective_system_prompt}

            if context_manager is not None:
                try:
                    messages = context_manager.assemble_context(session, system_message)
                except ContextOverflowError as exc:
                    log.error(
                        "agent_loop.context_overflow",
                        used=exc.used,
                        limit=exc.limit,
                        session_id=session.session_id,
                    )
                    # Send error to client and abort
                    from .protocol import make_error

                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=ref_id,
                                code="context_overflow",
                                message=(
                                    f"Context is full ({exc.used} tokens, limit {exc.limit}). "
                                    "Use /compact to compact the conversation"
                                    " or /new to start fresh."
                                ),
                                recoverable=True,
                            )
                        )
                    )
                    return int((time.time() - loop_start) * 1000), 0, total_tool_calls
            else:
                # Fallback: no context manager — send all messages directly
                messages = [system_message] + [m.to_openai_format() for m in session.messages]

            # Stream from vLLM
            accumulated_text = ""
            _assistant_added = False
            tool_call_accumulators: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            last_usage = None

            try:
                async for chunk in vllm_client.stream(messages, tools=tools):
                    if chunk.usage:
                        last_usage = chunk.usage

                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason

                    # Text delta
                    if chunk.delta_text:
                        accumulated_text += chunk.delta_text
                        if chunk.delta_text:
                            await ws.send(
                                json.dumps(
                                    make_stream_chunk(
                                        session_id=session.session_id,
                                        ref_id=ref_id,
                                        delta=chunk.delta_text,
                                        role="text",
                                    )
                                )
                            )

                    # Tool call deltas
                    for tc_delta in chunk.tool_call_deltas:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_call_accumulators[idx]
                        if "id" in tc_delta and tc_delta["id"]:
                            acc["id"] = tc_delta["id"]
                        if "type" in tc_delta:
                            acc["type"] = tc_delta["type"]
                        fn = tc_delta.get("function", {})
                        if "name" in fn and fn["name"]:
                            acc["function"]["name"] += fn["name"]
                        if "arguments" in fn and fn["arguments"] is not None:
                            acc["function"]["arguments"] += fn["arguments"]

                        # Stream tool_call delta to client
                        delta_str = json.dumps(tc_delta)
                        await ws.send(
                            json.dumps(
                                make_stream_chunk(
                                    session_id=session.session_id,
                                    ref_id=ref_id,
                                    delta=delta_str,
                                    role="tool_call",
                                )
                            )
                        )
            except httpx.ReadTimeout:
                # Either a server-side prompt-processing timeout (llama.cpp's ~60s
                # limit) or a bounded first-token stall (vllm_first_token, raised as
                # ReadTimeout in vllm_client.stream()). Only safe before the first
                # token: if any chunk was already emitted, surface and stop (see below).
                from .protocol import make_error

                if accumulated_text or tool_call_accumulators:
                    raise
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=ref_id,
                            code="vllm_timeout",
                            message=(
                                "The model server accepted the connection but sent no "
                                "response in time — it may still be loading, or the prompt "
                                "may be too large. Try again shortly."
                            ),
                            recoverable=True,
                        )
                    )
                )
                log.error("vllm_timeout", messages_count=len(messages))
                return int((time.time() - loop_start) * 1000), 0, total_tool_calls
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Process fully down (port closed) or connect budget exceeded. Only
                # safe before the first token (no chunk can have been emitted here,
                # since connect happens on the first stream advance — but guard anyway).
                from .protocol import make_error

                if accumulated_text or tool_call_accumulators:
                    raise
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=ref_id,
                            code="vllm_unavailable",
                            message=(
                                "The model server is unreachable. "
                                "It may be offline or still starting up."
                            ),
                            recoverable=True,
                        )
                    )
                )
                log.error("vllm_unavailable", messages_count=len(messages))
                return int((time.time() - loop_start) * 1000), 0, total_tool_calls
            except httpx.HTTPStatusError as exc:
                # raise_for_status() failed. recoverable is split by status class:
                # 5xx/408/429 (loading, overloaded, rate-limited) → a later retry of
                # the same prompt can succeed; other 4xx (esp. 400) → the request
                # itself was rejected, so resending the identical prompt fails the same.
                from .protocol import make_error

                if accumulated_text or tool_call_accumulators:
                    raise
                status = exc.response.status_code
                if status >= 500 or status in (408, 429):
                    message = (
                        f"The model server returned an error (HTTP {status}). "
                        "It may be loading or overloaded — try again shortly."
                    )
                    recoverable = True
                else:
                    message = (
                        f"The model server rejected the request (HTTP {status}). "
                        "The prompt may exceed the model's limit or be malformed."
                    )
                    recoverable = False
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=ref_id,
                            code="vllm_error",
                            message=message,
                            recoverable=recoverable,
                        )
                    )
                )
                log.error("vllm_error", status=status, messages_count=len(messages))
                return int((time.time() - loop_start) * 1000), 0, total_tool_calls

            # After streaming completes for this turn.
            # Some models (e.g. Qwen3) return finish_reason="stop" or "length" even when
            # tool calls are present. Trust the accumulator, not finish_reason.
            log.info(
                "agent_loop.turn_complete",
                loop=loop_count,
                finish_reason=finish_reason,
                text_chars=len(accumulated_text),
                pending_tool_calls=len(tool_call_accumulators),
                session_id=session.session_id,
            )
            if tool_call_accumulators:
                # Build tool call list
                tool_calls: list[dict[str, Any]] = []
                for idx in sorted(tool_call_accumulators.keys()):
                    acc = tool_call_accumulators[idx]
                    # Ensure arguments is valid JSON
                    try:
                        json.loads(acc["function"]["arguments"])
                    except json.JSONDecodeError:
                        acc["function"]["arguments"] = "{}"
                    tool_calls.append(acc)

                # Add assistant message with tool calls to session
                await session.add_assistant_message(accumulated_text, tool_calls=tool_calls)
                _assistant_added = True

                # Process each tool call
                _stop_loop = False
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    # Get representative command string for display/approval
                    command_str = arguments.get(
                        "command",
                        arguments.get("path", json.dumps(arguments)),
                    )

                    # ----------------------------------------------------------
                    # 0. Deduplicate consecutive identical search queries
                    # ----------------------------------------------------------
                    if tool_name in _SEARCH_DEDUP_TOOLS:
                        _query = arguments.get("query", "")
                        if _query:
                            if _query == last_search_query.get(tool_name):
                                _dedup_result = {
                                    "stdout": (
                                        f'Duplicate search detected: you already queried'
                                        f' "{_query}" using {tool_name}.'
                                        " The results are already in your context."
                                        " You must use a different query."
                                    ),
                                    "stderr": "",
                                    "exit_code": 0,
                                    "duration_ms": 0,
                                }
                                await ws.send(
                                    json.dumps(
                                        make_tool_output(
                                            session_id=session.session_id,
                                            ref_id=ref_id,
                                            tool=tool_name,
                                            command=_query,
                                            result=_dedup_result,
                                        )
                                    )
                                )
                                await session.add_tool_result(
                                    tool_call_id=tc["id"],
                                    content=_tool_result_content(_dedup_result),
                                    tool_name=tool_name,
                                    metadata={},
                                )
                                log.info(
                                    "agent_loop.duplicate_search_blocked",
                                    tool=tool_name,
                                    query=_query,
                                    session_id=session.session_id,
                                )
                                continue
                            else:
                                last_search_query[tool_name] = _query

                    # ----------------------------------------------------------
                    # 1. Check interceptors (e.g. create_plan in discovery phase)
                    # ----------------------------------------------------------
                    if tool_interceptors and tool_name in tool_interceptors:
                        int_result = await tool_interceptors[tool_name](arguments)
                        if int_result.get("__stop__"):
                            log.info(
                                "agent_loop.interceptor_stop",
                                tool=tool_name,
                                session_id=session.session_id,
                            )
                            # Add a stub tool result so session history stays consistent
                            await session.add_tool_result(
                                tool_call_id=tc["id"],
                                content=json.dumps(
                                    {k: v for k, v in int_result.items() if k != "__stop__"}
                                ),
                                tool_name=tool_name,
                                metadata={},
                            )
                            _stop_loop = True
                            break
                        # Interceptor returned a plain result — use it as the tool result
                        result: dict[str, Any] = int_result
                        # Skip permission check and execute_tool; go straight to recording
                        log.info(
                            "agent_loop.interceptor_result",
                            tool=tool_name,
                            session_id=session.session_id,
                        )
                        await session.add_tool_result(
                            tool_call_id=tc["id"],
                            content=_tool_result_content(result),
                            tool_name=tool_name,
                            metadata=_build_tool_metadata(tool_name, arguments),
                        )
                        continue

                    # ----------------------------------------------------------
                    # 2. Check blocked tools (e.g. write tools during discovery)
                    #    get_blocked_tools() overrides static blocked_tools set.
                    # ----------------------------------------------------------
                    _dynamic = get_blocked_tools() if get_blocked_tools else None
                    _active_blocked = (_dynamic[0] if _dynamic else blocked_tools) or frozenset()
                    _active_blocked_msg = (_dynamic[1] if _dynamic else blocked_tool_message) or ""
                    if tool_name in _active_blocked:
                        msg = _active_blocked_msg or f"Tool '{tool_name}' is not available."
                        log.info(
                            "agent_loop.tool_blocked",
                            tool=tool_name,
                            session_id=session.session_id,
                        )
                        result = {
                            "stdout": "",
                            "stderr": msg,
                            "exit_code": 1,
                            "duration_ms": 0,
                        }
                        # Send tool_output so client sees the block
                        await ws.send(
                            json.dumps(
                                make_tool_output(
                                    session_id=session.session_id,
                                    ref_id=ref_id,
                                    tool=tool_name,
                                    command=str(command_str),
                                    result=result,
                                )
                            )
                        )
                        await session.add_tool_result(
                            tool_call_id=tc["id"],
                            content=_tool_result_content(result),
                            tool_name=tool_name,
                            metadata={},
                        )
                        continue

                    # ----------------------------------------------------------
                    # 3. Normal flow: permission check → execute_tool
                    # ----------------------------------------------------------

                    # Permission check
                    approved = await permission_manager.check(
                        session_id=session.session_id,
                        tool_name=tool_name,
                        command=str(command_str),
                        ws=ws,
                    )

                    log.info(
                        "agent_loop.tool_call",
                        tool=tool_name,
                        command=str(command_str),
                        approved=approved,
                        session_id=session.session_id,
                    )

                    if approved:
                        try:
                            result = await execute_tool(tool_name, root_path, arguments)
                        except Exception as e:
                            log.error(
                                "agent_loop.tool_error",
                                tool=tool_name,
                                error=str(e),
                                session_id=session.session_id,
                            )
                            result = {
                                "stdout": "",
                                "stderr": f"Tool execution error: {e}",
                                "exit_code": 1,
                                "duration_ms": 0,
                            }
                    else:
                        result = {
                            "stdout": "",
                            "stderr": "User denied this action",
                            "exit_code": 1,
                            "duration_ms": 0,
                        }

                    _output_str = (
                        result.get("stdout")
                        or result.get("content")
                        or result.get("tree")
                        or result.get("symbols")
                        or result.get("matches")
                        or result.get("entries")
                        or ""
                    )
                    log.info(
                        "agent_loop.tool_done",
                        tool=tool_name,
                        exit_code=result.get("exit_code", 0),
                        duration_ms=result.get("duration_ms", 0),
                        output_bytes=len(str(_output_str)),
                        session_id=session.session_id,
                    )

                    # Send tool_output to client.
                    # make_tool_output only reads stdout/stderr/exit_code; for rich
                    # tool results (tree, symbols, content, matches) we flatten to stdout.
                    display_result = {
                        **result,
                        "stdout": _result_stdout(result),
                        "stderr": result.get("stderr", ""),
                        "exit_code": result.get("exit_code", 0),
                    }
                    await ws.send(
                        json.dumps(
                            make_tool_output(
                                session_id=session.session_id,
                                ref_id=ref_id,
                                tool=tool_name,
                                command=str(command_str),
                                result=display_result,
                            )
                        )
                    )

                    # Build metadata for staleness detection
                    tool_metadata = _build_tool_metadata(tool_name, arguments)

                    # Add tool result to session history with tool_name and metadata
                    await session.add_tool_result(
                        tool_call_id=tc["id"],
                        content=_tool_result_content(result),
                        tool_name=tool_name,
                        metadata=tool_metadata,
                    )

                    # Notify caller of tool result (used for correction state tracking)
                    if on_tool_result is not None:
                        on_tool_result(tool_name, arguments, result)

                total_tool_calls += len(tool_calls)
                # If an interceptor signalled stop, exit the loop entirely
                if _stop_loop:
                    log.info(
                        "agent_loop.interceptor_stopped",
                        loops=loop_count,
                        session_id=session.session_id,
                    )
                    duration_ms = int((time.time() - loop_start) * 1000)
                    tokens_this_loop = session.tokens_used - tokens_before
                    return duration_ms, tokens_this_loop, total_tool_calls
                # Mid-loop: prune stale reads/volatile output, then compact if needed
                if context_manager is not None:
                    tokens_freed_mid_loop += context_manager.prune_stale(session)
                    if compactor is not None and context_manager.should_compact(session):
                        try:
                            tokens_freed_mid_loop += await compactor.run(
                                session, vllm_client, skip_initial_prune=True
                            )
                        except Exception as exc:
                            log.warning(
                                "agent_loop.mid_loop_compact_failed",
                                error=str(exc),
                                session_id=session.session_id,
                            )
                # Loop continues — model sees tool results
                continue

            else:
                # No tool calls — check for null response (empty text + no tool calls).
                # This happens with Qwen3 thinking mode: the model spends the full turn
                # generating reasoning tokens, emits empty visible content, then stops.
                # Injecting a nudge keeps the loop alive instead of going idle mid-task.
                if not accumulated_text.strip() and total_tool_calls > 0:
                    log.warning(
                        "agent_loop.null_response",
                        loop=loop_count,
                        finish_reason=finish_reason,
                        session_id=session.session_id,
                    )
                    await session.add_assistant_message("")
                    await session.add_user_message("Continue.")
                    continue

                log.info(
                    "agent_loop.finished",
                    loops=loop_count,
                    total_tool_calls=total_tool_calls,
                    finish_reason=finish_reason,
                    session_id=session.session_id,
                )
                await session.add_assistant_message(accumulated_text)
                duration_ms = int((time.time() - loop_start) * 1000)
                tokens_this_loop = (session.tokens_used - tokens_before) + tokens_freed_mid_loop
                if emit_stream_end:
                    await ws.send(
                        json.dumps(
                            make_stream_end(
                                session_id=session.session_id,
                                ref_id=ref_id,
                                aborted=False,
                                usage=last_usage,
                                context_limit=config.model.context_limit,
                                session_budget=session.context_budget,
                                finish_reason=finish_reason,
                                tool_calls_made=total_tool_calls,
                                duration_ms=duration_ms,
                                tokens_used=tokens_this_loop,
                                tool_call_count=total_tool_calls,
                            )
                        )
                    )
                return duration_ms, tokens_this_loop, total_tool_calls

    except asyncio.CancelledError:
        # 1. Save partial assistant message if any text was streamed this turn
        if accumulated_text and not _assistant_added:
            _tm_tokens = count_tokens(accumulated_text)
            session.messages.append(
                TrackedMessage(
                    role="assistant",
                    content=accumulated_text,
                    tokens=_tm_tokens,
                    timestamp=time.time(),
                    message_type="response",
                )
            )
            session.tokens_used += _tm_tokens
        # 2. Inject a user-role message marking the interruption.
        # Must be "user" not "system" — chat templates reject system messages
        # that appear after non-system messages in the conversation.
        _interrupt = (
            "[Assistant response was interrupted by the user redirecting the conversation.]"
        )
        _sys_tokens = count_tokens(_interrupt)
        session.messages.append(
            TrackedMessage(
                role="user",
                content=_interrupt,
                tokens=_sys_tokens,
                timestamp=time.time(),
                message_type="system",
            )
        )
        session.tokens_used += _sys_tokens
        # 3. Notify client — shield so the send is not itself cancelled
        _dur = int((time.time() - loop_start) * 1000)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(
                ws.send(
                    json.dumps(
                        make_stream_end(
                            session_id=session.session_id,
                            ref_id=ref_id,
                            aborted=True,
                            finish_reason="abort",
                            usage=None,
                            context_limit=config.model.context_limit,
                            session_budget=session.context_budget,
                            tool_calls_made=total_tool_calls,
                            duration_ms=_dur,
                        )
                    )
                )
            )
        raise
