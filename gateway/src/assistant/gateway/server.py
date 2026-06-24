"""WebSocket server — main entry point."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import structlog
import websockets.exceptions
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve

if TYPE_CHECKING:
    from websockets.http11 import Request, Response

from .agent_registry import AgentRegistry
from .auth import AuthManager
from .compactor import Compactor
from .config import GatewayConfig, RAGConfig
from .config_watcher import ConfigWatcher
from .context_manager import ContextManager
from .file_indexer import build_file_index
from .permissions import PermissionManager
from .planner import ExecutionMode, Planner
from .project_store import ProjectStore
from .protocol import (
    AgentDeletePayload,
    AgentUpsertPayload,
    ApprovalResponsePayload,
    CommandPayload,
    ProjectDeletePayload,
    ProjectSelectPayload,
    ProjectUpsertPayload,
    PromptPayload,
    SessionResumePayload,
    make_agent_registry,
    make_command_result,
    make_config_reloaded,
    make_error,
    make_file_index,
    make_model_waiting,
    make_project_context,
    make_project_registry,
    make_session_history,
    make_session_resumed,
    make_session_start,
    make_status,
    make_stream_end,
    make_tool_output,
    parse_client_message,
)
from .rag_client import RAGChunk, RAGClient, RAGCollectionInfo
from .request_queue import RequestQueue
from .session import Session, SessionManager, count_tokens, init_tokenizer
from .session_store import SessionNotFoundError, SessionStore
from .tools import get_known_tools, get_schemas, update_tool_registry
from .tools.project_detect import ProjectType, build_system_prompt_section, detect_project_type
from .tools.web_client import close_web_client
from .vllm_client import VLLMClient

log = structlog.get_logger(__name__)


def _build_agents_for_context(
    agents_raw: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    """Transform raw agents.toml data into protocol-conformant agent entries.

    Adds backend_model (resolved), includes tools, strips internal prompt_file.
    """
    out: dict[str, Any] = {}
    for name, data in agents_raw.items():
        backend_key = data.get("backend")
        resolved = config.resolve_backend(backend_key)
        out[name] = {
            "description": data.get("description", ""),
            "backend": backend_key,
            "backend_model": resolved.model_name,
            "tools": data.get("tools", []),
        }
    return out


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _build_history_turns(session: Session) -> list[dict[str, Any]]:
    """Extract displayable turns from a session for the session_history message.

    Returns the compaction summary (if any) followed by user/assistant pairs,
    skipping tool calls, tool results, pruned messages, and already-compacted messages.
    """
    turns: list[dict[str, Any]] = []
    if session.compacted_summary:
        turns.append({
            "role": "assistant",
            "content": session.compacted_summary.content,
            "timestamp": session.compacted_summary.timestamp,
            "is_summary": True,
        })
    for m in session.messages:
        if m.message_type not in ("prompt", "response"):
            continue
        if m.pruned or m.compacted:
            continue
        if isinstance(m.content, list):
            parts = [
                b.get("text", "")
                for b in m.content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content: str = "\n".join(parts)
        else:
            content = m.content or ""
        turns.append({"role": m.role, "content": content, "timestamp": m.timestamp})
    return turns


_DEFAULT_SESSION_CONFIG: dict[str, Any] = {
    "diff_mode": False,
    "verbosity": "ask_ambiguous",
    "max_retries": 3,
    # Planner / discovery settings
    "auto_plan": True,
    "discovery_budget_pct": 0.38,
    "triage_max_reads": 10,
    # Context management overrides (None = inherit from gateway.toml)
    "auto_prune": None,
    "auto_compact": None,
    "pin_first_tokens": None,
    "pin_recent_tokens": None,
    "compact_target_tokens": None,
}


def _build_web_search_prompt_section() -> str:
    """Return the web search strategy system prompt section.

    This section is only included when web_search tool is available.
    """
    return """## Web search strategy

When you need information beyond your training data — library recommendations,
API documentation, current best practices, or researching a topic:

1. Use `web_search` first to find relevant pages
2. Read the snippets — they often answer the question without fetching
3. Use `web_fetch` to read specific pages when snippets aren't enough
4. Prefer official sources: documentation sites, GitHub repos, PyPI pages
5. When researching a topic, search multiple angles before synthesizing

**Important:** Always use the `web_search` tool for web searches. Do not use
curl or external HTTP tools to search search engines directly, as they have
captcha and bot protection that will block your requests. The `web_search`
tool is the safe, preferred method for searching the web.

Do not use web search for questions you can confidently answer from training.
Do not fetch multiple pages at once — read one, decide if you need more."""


def _build_rag_prompt_section(collections: list[RAGCollectionInfo]) -> str:
    lines = [
        "## Reference materials",
        "",
        "You have access to indexed reference materials via the `rag_search` tool.",
        "Available references for this project:",
    ]
    for col in collections:
        desc = f": {col.description}" if col.description else ""
        lines.append(f"- {col.name}{desc}")
    lines.extend([
        "",
        "Use `rag_search` when the answer is likely in these references",
        "rather than in the project's source code or your own knowledge.",
        "Formulate specific, targeted queries.",
    ])
    return "\n".join(lines)


def _format_rag_results(query: str, results: list[RAGChunk], total: int) -> str:
    shown = len(results)
    lines = [f'RAG Search: "{query}" ({shown} of {total} results)', ""]
    for i, chunk in enumerate(results, 1):
        source_parts = [chunk.collection, chunk.document]
        if chunk.page is not None:
            source_parts.append(f"p.{chunk.page}")
        if chunk.section:
            source_parts.append(chunk.section)
        lines.append(f"[{i}] {' / '.join(source_parts)}")
        lines.append(f"Path: {chunk.path} ({chunk.size})")
        lines.append(f"Score: {chunk.score:.2f} | {chunk.token_count} tokens")
        lines.append("---")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def _make_rag_interceptor(
    session: Session,
    context_manager: ContextManager,
    rag_client: RAGClient,
    collections: list[str],
    ws: Any,
    ref_id: str,
    config_rag: RAGConfig,
) -> Any:

    async def _interceptor(arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query", "")

        budget = context_manager.compute_budget(session)
        if budget.used > config_rag.context_budget_threshold:
            result: dict[str, Any] = {
                "stdout": (
                    "The context budget for this tool has been exceeded. "
                    "Results cannot be displayed."
                ),
                "stderr": "",
                "exit_code": 1,
            }
            await ws.send(
                json.dumps(
                    make_tool_output(
                        session_id=session.session_id,
                        ref_id=ref_id,
                        tool="rag_search",
                        command=query,
                        result=result,
                    )
                )
            )
            return result

        search_result = await rag_client.search(
            query=query,
            collections=collections,
            top_k=config_rag.top_k,
        )

        if search_result is None:
            result = {
                "stdout": (
                    "RAG search failed: service unavailable. "
                    "Proceeding without reference materials."
                ),
                "stderr": "",
                "exit_code": 1,
            }
        elif not search_result.results:
            result = {
                "stdout": f'No relevant results found for: "{query}"',
                "stderr": "",
                "exit_code": 0,
            }
        else:
            result = {
                "stdout": _format_rag_results(
                    query, search_result.results, search_result.total_results
                ),
                "stderr": "",
                "exit_code": 0,
            }

        await ws.send(
            json.dumps(
                make_tool_output(
                    session_id=session.session_id,
                    ref_id=ref_id,
                    tool="rag_search",
                    command=query,
                    result=result,
                )
            )
        )
        return result

    return _interceptor


def _effective_context_manager(
    base: ContextManager | None,
    session_config: dict[str, Any],
) -> ContextManager | None:
    """Return a ContextManager with any session_config overrides applied.

    Returns the base instance unchanged when no context keys are overridden.
    """
    if base is None:
        return None
    cm_keys = ("pin_first_tokens", "pin_recent_tokens")
    overrides = {k: session_config[k] for k in cm_keys if session_config.get(k) is not None}
    if not overrides:
        return base
    return ContextManager(
        pin_first_tokens=overrides.get("pin_first_tokens", base.pin_first_tokens),
        pin_recent_tokens=overrides.get("pin_recent_tokens", base.pin_recent_tokens),
        reserved_response_tokens=base.reserved_response_tokens,
        prune_threshold=base.prune_threshold,
        compact_threshold=base.compact_threshold,
        warn_threshold=base.warn_threshold,
    )


class ConnectionState:
    """Per-connection mutable state."""

    def __init__(self) -> None:
        self.active_project_id: str | None = None
        self.active_session: Session | None = None
        # Context file loaded on project_select / refresh
        self.context_file_content: str | None = None
        self.context_file_name: str | None = None
        self.context_file_tokens: int = 0
        # Project type detected on project_select
        self.project_type: ProjectType | None = None
        # Session-scoped config overrides
        self.session_config: dict[str, Any] = dict(_DEFAULT_SESSION_CONFIG)
        # RAG state set on project_select
        self.rag_collections: list[str] = []
        self.rag_system_prompt: str | None = None
        # Background task running the active agent loop (None when idle)
        self.active_loop_task: asyncio.Task[None] | None = None


async def _run_prompt_task(
    planner: Planner,
    content: list[dict[str, Any]],
    request_queue: RequestQueue,
    session: Session,
    effective_cm: ContextManager | None,
    ws: ServerConnection,
    config: GatewayConfig,
    state: ConnectionState,
    msg_id: str,
    vllm_client: VLLMClient,
) -> None:
    """Run planner.run_with_planning() as a background task.

    Owns the full lifecycle: run, complete queue, post-processing.
    Cancellation is handled by agent_loop (partial message + stream_end(aborted=True)).
    """
    loop_duration_ms: int | None = None
    loop_tokens_used: int | None = None
    loop_tool_call_count: int | None = None
    # Tracks whether the planner has taken over. Until then, this task owns the
    # terminal contract (an abort during the wait must self-emit stream_end).
    planner_started = False
    try:
        # Wait-for-ready gate: the enqueued request already holds the active slot,
        # so other prompts correctly queue behind it (desired FIFO). Probe first;
        # only announce model_waiting if unhealthy — a healthy model emits zero
        # model_waiting messages and goes straight to the planner.
        wait_start = time.time()
        backoff = 1.0
        while True:
            if await vllm_client.health():
                break
            waited = time.time() - wait_start  # wall-clock budget (probe time + sleeps)
            if waited >= config.timeouts.vllm_startup_wait:
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=msg_id,
                            code="vllm_unavailable",
                            message=(
                                "The model server did not come online in time. "
                                "It may be offline — try again later."
                            ),
                            recoverable=True,
                        )
                    )
                )
                log.error("server.vllm_startup_timeout", session_id=session.session_id)
                return  # no stream_end (error is its own terminal); finally runs complete()
            await ws.send(
                json.dumps(
                    make_model_waiting(
                        session_id=session.session_id,
                        ref_id=msg_id,
                        waited_seconds=int(waited),
                        max_wait_seconds=config.timeouts.vllm_startup_wait,
                    )
                )
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)  # 1 → 2 → 4 → 5 (cap)

        planner_started = True
        loop_duration_ms, loop_tokens_used, loop_tool_call_count = (
            await planner.run_with_planning(content)
        )
    except asyncio.CancelledError:
        if not planner_started:
            # Aborted/disconnected during the wait: the planner never ran, so no
            # stream_end was sent. The client is in a "waiting" UI state and needs
            # a terminal event — self-emit it, then re-raise so finally runs complete().
            with contextlib.suppress(Exception):
                await ws.send(
                    json.dumps(
                        make_stream_end(
                            session_id=session.session_id,
                            ref_id=msg_id,
                            aborted=True,
                            usage=None,
                        )
                    )
                )
            raise
        # planner took over: agent_loop already saved partial message + sent
        # stream_end(aborted=True).
        return
    finally:
        await asyncio.shield(request_queue.complete())
        state.active_loop_task = None

    # Auto-prune / compact after each agent loop
    if effective_cm is not None:
        try:
            auto_compact = state.session_config.get("auto_compact")
            if auto_compact is None:
                auto_compact = config.context.auto_compact
            auto_prune = state.session_config.get("auto_prune")
            if auto_prune is None:
                auto_prune = config.context.auto_prune
            compact_target = (
                state.session_config.get("compact_target_tokens")
                or config.context.compact_target_tokens
            )

            if auto_compact and effective_cm.should_compact(session):
                compactor = Compactor(
                    context_manager=effective_cm,
                    compact_target_tokens=compact_target,
                )
                freed = await compactor.run(session, vllm_client)
                if freed > 0:
                    log.info(
                        "server.auto_compacted",
                        session_id=session.session_id,
                        tokens_freed=freed,
                    )
            elif auto_prune and effective_cm.should_prune(session):
                freed = effective_cm.prune(session)
                if freed > 0:
                    log.info(
                        "server.auto_pruned",
                        session_id=session.session_id,
                        tokens_freed=freed,
                    )

            # Send updated token usage with breakdown, plus loop metrics
            budget = effective_cm.compute_budget(session)
            await ws.send(
                json.dumps(
                    make_status(
                        session_id=session.session_id,
                        state="idle",
                        token_usage=budget.to_dict(),
                        duration_ms=loop_duration_ms,
                        tokens_used=loop_tokens_used,
                        tool_call_count=loop_tool_call_count,
                    )
                )
            )

            if effective_cm.should_warn(session):
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=msg_id,
                            code="context_warning",
                            message=(
                                f"Context is at {int(session.context_pressure * 100)}%."
                                " Consider using /compact or /new."
                            ),
                            recoverable=True,
                        )
                    )
                )
        except Exception as exc:
            log.error("server.auto_compact_error", error=str(exc))


async def handle_connection(
    ws: ServerConnection,
    config: GatewayConfig,
    session_manager: SessionManager,
    project_store: ProjectStore,
    agent_registry: AgentRegistry,
    request_queue: RequestQueue,
    auth_manager: AuthManager,
    permission_manager_factory: Any,  # callable(project_id) -> PermissionManager
    context_manager: ContextManager | None = None,
    rag_client: RAGClient | None = None,
    session_store: SessionStore | None = None,
) -> None:
    """Handle a single WebSocket connection."""

    # 1. Auth check
    request_path = ws.request.path if hasattr(ws, "request") and ws.request else ""
    query = parse_qs(urlparse(request_path).query) if request_path else {}
    token_list = query.get("token", [])
    token = token_list[0] if token_list else None

    if not auth_manager.check_connection(token):
        await ws.send(
            json.dumps(
                make_error(
                    ref_id=None,
                    code="auth_required",
                    message="Authentication required",
                    recoverable=False,
                )
            )
        )
        await ws.close()
        return

    # 2. Send session_start
    available_backends = ["local"] + list(config.model.backends.keys())
    await ws.send(
        json.dumps(
            make_session_start(
                auth_mode=config.auth.mode,
                default_model_name=config.model.model_name,
                available_backends=available_backends,
            )
        )
    )

    state = ConnectionState()

    # 3. Message loop
    try:
        async for raw_msg in ws:
            try:
                raw_str = raw_msg.decode() if isinstance(raw_msg, bytes) else raw_msg
                msg = parse_client_message(raw_str)
            except Exception as e:
                await ws.send(
                    json.dumps(
                        make_error(
                            ref_id=None,
                            code="internal",
                            message=f"Failed to parse message: {e}",
                        )
                    )
                )
                continue

            msg_type = msg.type
            msg_id = msg.id
            effective_cm = _effective_context_manager(context_manager, state.session_config)

            if msg_type == "project_list":
                projects_raw = project_store.list_projects()
                projects_out: dict[str, Any] = {}
                for pid, pdata in projects_raw.items():
                    agents = agent_registry.list_agents(pid)
                    sessions = session_manager.list_by_project(pid)
                    projects_out[pid] = {
                        "description": pdata.get("description", ""),
                        "root_path": pdata.get("root_path", ""),
                        "agents": list(agents.keys()),
                        "active_sessions": len(sessions),
                    }
                await ws.send(
                    json.dumps(make_project_registry(ref_id=msg_id, projects=projects_out))
                )

            elif msg_type == "project_select":
                p = msg.payload
                if not isinstance(p, ProjectSelectPayload):
                    continue
                project_id = p.project_id
                project_data = project_store.get(project_id)
                if project_data is None:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="project_not_found",
                                message=f"Project '{project_id}' not found",
                            )
                        )
                    )
                    continue
                state.active_project_id = project_id

                # Detect project type (scan build files; caches result for build/run_tests tools)
                root_path_for_detect = project_data.get("root_path", "")
                if root_path_for_detect:
                    try:
                        state.project_type = detect_project_type(root_path_for_detect)
                    except Exception as e:
                        log.warning("project_select.detect_error", error=str(e))
                        state.project_type = None
                else:
                    state.project_type = None

                # Load context file (auto-detect or use configured name)
                context_result = project_store.load_context_file(
                    root_path=project_data.get("root_path", ""),
                    context_file=project_data.get("context_file") or None,
                )
                if context_result:
                    state.context_file_content, state.context_file_name = context_result
                    state.context_file_tokens = count_tokens(state.context_file_content)
                else:
                    state.context_file_content = None
                    state.context_file_name = None
                    state.context_file_tokens = 0

                # RAG setup — validate collections and build system prompt section
                state.rag_collections = []
                state.rag_system_prompt = None
                if config.rag.enabled and rag_client is not None:
                    project_collections = project_data.get("rag", {}).get("collections", [])
                    if project_collections:
                        all_collections = await rag_client.list_collections()
                        if all_collections is None:
                            log.warning(
                                "rag_unavailable_at_project_select", project=project_id
                            )
                        else:
                            collection_map = {c.name: c for c in all_collections}
                            available: list[RAGCollectionInfo] = []
                            for name in project_collections:
                                if name not in collection_map:
                                    log.warning("rag_collection_not_found", collection=name)
                                elif not collection_map[name].compatible:
                                    log.warning(
                                        "rag_collection_incompatible", collection=name
                                    )
                                else:
                                    available.append(collection_map[name])
                            if available:
                                state.rag_collections = [c.name for c in available]
                                state.rag_system_prompt = _build_rag_prompt_section(available)

                agents_raw = agent_registry.list_agents(project_id)
                agents = _build_agents_for_context(agents_raw, config)
                last_session: dict[str, Any] | None = None
                if session_store is not None:
                    last_session_meta = await session_store.get_last_session(project_id)
                    if last_session_meta is not None:
                        last_session = {
                            "session_id": last_session_meta.session_id,
                            "last_active": int(last_session_meta.last_active * 1000),
                        }
                await ws.send(
                    json.dumps(
                        make_project_context(
                            ref_id=msg_id,
                            project_id=project_id,
                            description=project_data.get("description", ""),
                            root_path=project_data.get("root_path", ""),
                            context_file=(
                                state.context_file_name or project_data.get("context_file", "")
                            ),
                            agents=agents,
                            last_session=last_session,
                        )
                    )
                )
                _root_path_str = project_data.get("root_path", "")
                try:
                    _files = await asyncio.get_running_loop().run_in_executor(
                        None, build_file_index, Path(_root_path_str)
                    )
                except Exception:
                    log.exception("file_index.build_failed", project_id=project_id)
                    _files = []
                await ws.send(
                    json.dumps(make_file_index(ref_id=msg_id, project_id=project_id, files=_files))
                )

            elif msg_type == "session_resume":
                p = msg.payload
                if not isinstance(p, SessionResumePayload):
                    continue

                if not state.active_project_id:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="no_project_selected",
                                message="Select a project before resuming a session",
                                recoverable=True,
                            )
                        )
                    )
                    continue

                if session_store is None:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="internal",
                                message="Session persistence not available",
                                recoverable=True,
                            )
                        )
                    )
                    continue

                session_id = p.session_id
                session = session_manager.get(session_id)
                recovered_from_disk = False

                if session is None:
                    try:
                        meta, messages, compacted_summary = await session_store.load_session(
                            state.active_project_id, session_id
                        )
                    except SessionNotFoundError:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="session_not_found",
                                    message=f"Session {session_id} not found",
                                    recoverable=True,
                                )
                            )
                        )
                        continue

                    session = Session.from_persisted(
                        meta=meta,
                        messages=messages,
                        compacted_summary=compacted_summary,
                        store=session_store,
                        context_limit=config.model.context_limit,
                    )
                    session_manager._sessions[session_id] = session
                    recovered_from_disk = True

                incomplete_turn = (
                    len(session.messages) > 0
                    and session.messages[-1].role == "user"
                )
                state.active_session = session

                await ws.send(
                    json.dumps(
                        make_session_resumed(
                            session_id=session_id,
                            incomplete_turn=incomplete_turn,
                            recovered_from_disk=recovered_from_disk,
                        )
                    )
                )
                history_turns = _build_history_turns(session)
                await ws.send(
                    json.dumps(
                        make_session_history(
                            session_id=session_id,
                            turns=history_turns,
                            has_compacted_prefix=session.compacted_summary is not None,
                        )
                    )
                )

            elif msg_type == "project_upsert":
                p = msg.payload
                if not isinstance(p, ProjectUpsertPayload):
                    continue
                project_id = p.project_id
                project = p.project
                try:
                    project_store.upsert(
                        project_id=project_id,
                        description=project.get("description", ""),
                        root_path=project.get("root_path", ""),
                        context_file=project.get("context_file", "PROJECT.md"),
                    )
                    project_data = project_store.get(project_id)
                    agents_raw = agent_registry.list_agents(project_id)
                    agents = _build_agents_for_context(agents_raw, config)
                    await ws.send(
                        json.dumps(
                            make_project_context(
                                ref_id=msg_id,
                                project_id=project_id,
                                description=(
                                    project_data.get("description", "") if project_data else ""
                                ),
                                root_path=(
                                    project_data.get("root_path", "") if project_data else ""
                                ),
                                context_file=(
                                    project_data.get("context_file", "PROJECT.md")
                                    if project_data
                                    else "PROJECT.md"
                                ),
                                agents=agents,
                            )
                        )
                    )
                    _root_path_str = project_data.get("root_path", "") if project_data else ""
                    try:
                        _files = await asyncio.get_running_loop().run_in_executor(
                            None, build_file_index, Path(_root_path_str)
                        )
                    except Exception:
                        log.exception("file_index.build_failed", project_id=project_id)
                        _files = []
                    await ws.send(
                        json.dumps(
                            make_file_index(ref_id=msg_id, project_id=project_id, files=_files)
                        )
                    )
                except ValueError as e:
                    await ws.send(
                        json.dumps(make_error(ref_id=msg_id, code="internal", message=str(e)))
                    )

            elif msg_type == "project_delete":
                p = msg.payload
                if not isinstance(p, ProjectDeletePayload):
                    continue
                try:
                    project_store.delete(p.project_id)
                except Exception as e:
                    await ws.send(
                        json.dumps(
                            make_error(ref_id=msg_id, code="project_not_found", message=str(e))
                        )
                    )

            elif msg_type == "agent_upsert":
                if not state.active_project_id:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="no_project_selected",
                                message="Select a project first",
                            )
                        )
                    )
                    continue
                p = msg.payload
                if not isinstance(p, AgentUpsertPayload):
                    continue
                errors = agent_registry.upsert(
                    project_id=state.active_project_id,
                    name=p.name,
                    agent=p.agent,
                )
                agents = agent_registry.list_agents(state.active_project_id)
                await ws.send(
                    json.dumps(
                        make_agent_registry(
                            ref_id=msg_id,
                            project_id=state.active_project_id,
                            agents=agents,
                            errors=errors,
                        )
                    )
                )

            elif msg_type == "agent_delete":
                if not state.active_project_id:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="no_project_selected",
                                message="Select a project first",
                            )
                        )
                    )
                    continue
                p = msg.payload
                if not isinstance(p, AgentDeletePayload):
                    continue
                try:
                    agent_registry.delete(state.active_project_id, p.name)
                except KeyError as e:
                    await ws.send(
                        json.dumps(make_error(ref_id=msg_id, code="internal", message=str(e)))
                    )
                    continue
                agents = agent_registry.list_agents(state.active_project_id)
                await ws.send(
                    json.dumps(
                        make_agent_registry(
                            ref_id=msg_id,
                            project_id=state.active_project_id,
                            agents=agents,
                            errors=[],
                        )
                    )
                )

            elif msg_type == "agent_list":
                if not state.active_project_id:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="no_project_selected",
                                message="Select a project first",
                            )
                        )
                    )
                    continue
                agents = agent_registry.list_agents(state.active_project_id)
                await ws.send(
                    json.dumps(
                        make_agent_registry(
                            ref_id=msg_id,
                            project_id=state.active_project_id,
                            agents=agents,
                            errors=[],
                        )
                    )
                )

            elif msg_type == "prompt":
                if not state.active_project_id:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="no_project_selected",
                                message="Select a project before sending prompts",
                            )
                        )
                    )
                    continue

                project_data = project_store.get(state.active_project_id)
                if project_data is None:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="project_not_found",
                                message="Selected project no longer exists",
                            )
                        )
                    )
                    continue

                root_path = project_data.get("root_path", "")
                p = msg.payload
                if not isinstance(p, PromptPayload):
                    continue

                # Get or create session
                if p.session_id:
                    session = session_manager.get(p.session_id)
                    if session is None:
                        session = session_manager.create(
                            state.active_project_id,
                            context_limit=config.model.context_limit,
                            store=session_store,
                        )
                        # Pre-seed token budget with context file overhead
                        if state.context_file_tokens:
                            session.tokens_used += state.context_file_tokens
                else:
                    session = session_manager.create(
                        state.active_project_id,
                        context_limit=config.model.context_limit,
                        store=session_store,
                    )
                    # Pre-seed token budget with context file overhead
                    if state.context_file_tokens:
                        session.tokens_used += state.context_file_tokens

                state.active_session = session

                # Agent lookup — resolve system prompt, tools, and backend
                agent_system_prompt: str | None = None
                tool_schemas: list[Any] = get_schemas(list(get_known_tools()))  # default: all tools
                agent_backend_key: str | None = None

                if p.agent_name:
                    agents = agent_registry.list_agents(state.active_project_id)
                    agent_data = agents.get(p.agent_name)
                    if agent_data:
                        agent_system_prompt = agent_registry.get_prompt(
                            state.active_project_id, p.agent_name
                        )
                        tool_names = agent_data.get("tools", [])
                        tool_schemas = get_schemas(tool_names)
                        agent_backend_key = agent_data.get("backend")

                # Combine context file + build/test section + agent system prompt
                prompt_parts: list[str] = []
                if state.context_file_content:
                    prompt_parts.append(f"## Project Context\n\n{state.context_file_content}")
                if state.project_type and state.project_type.types:
                    prompt_parts.append(build_system_prompt_section(state.project_type))

                # Add web search strategy section if web_search tool is available
                if "web_search" in get_known_tools():
                    prompt_parts.append(_build_web_search_prompt_section())

                # Add RAG section if configured for this project
                if state.rag_system_prompt:
                    prompt_parts.append(state.rag_system_prompt)

                if agent_system_prompt:
                    prompt_parts.append(agent_system_prompt)
                system_prompt: str | None = "\n\n".join(prompt_parts) if prompt_parts else None

                # Build vLLM client using agent's backend (or default)
                backend = config.resolve_backend(agent_backend_key)
                vllm_client = VLLMClient(
                    base_url=backend.vllm_base_url,
                    model_name=backend.model_name,
                    api_key=backend.api_key,
                    temperature=backend.temperature or config.model.temperature,
                    top_p=backend.top_p or config.model.top_p,
                    max_tokens=backend.max_tokens or config.model.max_tokens,
                    timeout=config.timeouts.vllm_request,
                    first_token_timeout=config.timeouts.vllm_first_token,
                )

                pm = permission_manager_factory(state.active_project_id)

                # Build RAG interceptor if RAG is configured for this project
                rag_interceptor: dict[str, Any] | None = None
                if (
                    config.rag.enabled
                    and rag_client is not None
                    and state.rag_collections
                    and effective_cm is not None
                ):
                    rag_interceptor = {
                        "rag_search": _make_rag_interceptor(
                            session=session,
                            context_manager=effective_cm,
                            rag_client=rag_client,
                            collections=state.rag_collections,
                            ws=ws,
                            ref_id=msg_id,
                            config_rag=config.rag,
                        )
                    }

                # Enqueue (blocks until the queue accepts this session)
                await request_queue.enqueue(
                    session_id=session.session_id,
                    ref_id=msg_id,
                    ws=ws,
                )
                planner = Planner(
                    session=session,
                    ws=ws,
                    vllm_client=vllm_client,
                    config=config,
                    ref_id=msg_id,
                    root_path=root_path,
                    permission_manager=pm,
                    context_manager=effective_cm,
                    execution_mode=ExecutionMode.INTERACTIVE,
                    session_config=state.session_config,
                    base_system_prompt=system_prompt,
                    tool_schemas=tool_schemas or None,
                    extra_tool_interceptors=rag_interceptor,
                )
                loop_task = asyncio.create_task(
                    _run_prompt_task(
                        planner=planner,
                        content=[b.model_dump() for b in p.content],
                        request_queue=request_queue,
                        session=session,
                        effective_cm=effective_cm,
                        ws=ws,
                        config=config,
                        state=state,
                        msg_id=msg_id,
                        vllm_client=vllm_client,
                    )
                )
                state.active_loop_task = loop_task
                # Do NOT await — continue the receive loop immediately so
                # approval_response and abort messages can be dispatched.

            elif msg_type == "approval_response":
                # Route to the active permission manager
                # The PM looks up pending futures by approval_id
                if state.active_project_id:
                    p = msg.payload
                    if not isinstance(p, ApprovalResponsePayload):
                        continue
                    pm = permission_manager_factory(state.active_project_id)
                    pm.resolve_approval(
                        p.ref_id,
                        p.decision,
                    )

            elif msg_type == "abort":
                task = state.active_loop_task
                if task and not task.done():
                    task.cancel()
                # stream_end(aborted=True) is sent by agent_loop's CancelledError handler

            elif msg_type == "command":
                p = msg.payload
                if not isinstance(p, CommandPayload):
                    continue
                cmd_name = p.name
                cmd_args = p.args
                log.info("command.received", name=cmd_name)

                if cmd_name == "refresh":
                    if not state.active_project_id:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="no_project_selected",
                                    message="Select a project before using /refresh",
                                )
                            )
                        )
                        continue
                    proj = project_store.get(state.active_project_id)
                    if proj is None:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="project_not_found",
                                    message="Selected project no longer exists",
                                )
                            )
                        )
                        continue
                    context_result = project_store.load_context_file(
                        root_path=proj.get("root_path", ""),
                        context_file=proj.get("context_file") or None,
                    )
                    if context_result:
                        state.context_file_content, state.context_file_name = context_result
                        state.context_file_tokens = count_tokens(state.context_file_content)
                        refresh_msg = (
                            f"Context file reloaded ({state.context_file_name}, "
                            f"{state.context_file_tokens} tokens)"
                        )
                    else:
                        state.context_file_content = None
                        state.context_file_name = None
                        state.context_file_tokens = 0
                        refresh_msg = "No context file found"

                    # If there's an active session, send updated token_usage
                    if state.active_session:
                        await ws.send(
                            json.dumps(
                                make_status(
                                    session_id=state.active_session.session_id,
                                    state="idle",
                                    token_usage=state.active_session.context_budget,
                                )
                            )
                        )
                    await ws.send(
                        json.dumps(
                            make_command_result(
                                ref_id=msg_id,
                                name="refresh",
                                success=True,
                                message=refresh_msg,
                            )
                        )
                    )

                elif cmd_name == "new":
                    if not state.active_project_id:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="no_project_selected",
                                    message="Select a project before using /new",
                                )
                            )
                        )
                        continue
                    new_session = session_manager.create(
                        state.active_project_id,
                        context_limit=config.model.context_limit,
                        store=session_store,
                    )
                    if state.context_file_tokens:
                        new_session.tokens_used += state.context_file_tokens
                    state.active_session = new_session
                    await ws.send(
                        json.dumps(
                            make_command_result(
                                ref_id=msg_id,
                                name="new",
                                success=True,
                                message="New session created",
                                data={"session_id": new_session.session_id},
                            )
                        )
                    )

                elif cmd_name == "permissions":
                    if not state.active_project_id:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="no_project_selected",
                                    message="Select a project before using /permissions",
                                )
                            )
                        )
                        continue
                    pm = permission_manager_factory(state.active_project_id)
                    perms_data: dict[str, Any] = {
                        "persistent": pm.stored_rules(),
                        "session": {},
                        "defaults": {
                            "safe": config.permissions.default_safe,
                            "destructive": config.permissions.default_destructive,
                        },
                    }
                    await ws.send(
                        json.dumps(
                            make_command_result(
                                ref_id=msg_id,
                                name="permissions",
                                success=True,
                                message="Current permissions",
                                data=perms_data,
                            )
                        )
                    )

                elif cmd_name == "compact":
                    if not state.active_session:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="no_session",
                                    message="No active session to compact",
                                )
                            )
                        )
                        continue
                    if effective_cm is None:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="internal",
                                    message="Context manager not available",
                                )
                            )
                        )
                        continue

                    before_tokens = state.active_session.tokens_used

                    # Build a vllm_client for the compaction call
                    backend = config.resolve_backend(None)
                    compact_vllm = VLLMClient(
                        base_url=backend.vllm_base_url,
                        model_name=backend.model_name,
                        api_key=backend.api_key,
                        temperature=backend.temperature or config.model.temperature,
                        top_p=backend.top_p or config.model.top_p,
                        max_tokens=backend.max_tokens or config.model.max_tokens,
                        timeout=config.timeouts.vllm_request,
                    )
                    compact_target = (
                        state.session_config.get("compact_target_tokens")
                        or config.context.compact_target_tokens
                    )
                    compactor = Compactor(
                        context_manager=effective_cm,
                        compact_target_tokens=compact_target,
                    )
                    try:
                        freed = await compactor.manual_compact(
                            state.active_session,
                            compact_vllm,
                        )
                        after_tokens = state.active_session.tokens_used
                        budget = effective_cm.compute_budget(state.active_session)
                        await ws.send(
                            json.dumps(
                                make_status(
                                    session_id=state.active_session.session_id,
                                    state="idle",
                                    token_usage=budget.to_dict(),
                                )
                            )
                        )
                        await ws.send(
                            json.dumps(
                                make_command_result(
                                    ref_id=msg_id,
                                    name="compact",
                                    success=True,
                                    message=(
                                        f"Compacted {before_tokens - after_tokens} tokens freed"
                                        f" ({before_tokens} → {after_tokens})"
                                    ),
                                    data={
                                        "tokens_before": before_tokens,
                                        "tokens_after": after_tokens,
                                        "tokens_freed": freed,
                                        "breakdown": budget.to_dict()["breakdown"],
                                    },
                                )
                            )
                        )
                    except Exception as exc:
                        log.error("server.compact_error", error=str(exc))
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="internal",
                                    message=f"Compaction failed: {exc}",
                                )
                            )
                        )

                elif cmd_name == "config":
                    _valid_config_keys = {
                        "diff_mode",
                        "verbosity",
                        "max_retries",
                        "auto_plan",
                        "discovery_budget_pct",
                        "triage_max_reads",
                        "auto_prune",
                        "auto_compact",
                        "pin_first_tokens",
                        "pin_recent_tokens",
                        "compact_target_tokens",
                    }
                    if cmd_args:
                        invalid_keys = set(cmd_args.keys()) - _valid_config_keys
                        if invalid_keys:
                            await ws.send(
                                json.dumps(
                                    make_error(
                                        ref_id=msg_id,
                                        code="internal",
                                        message=f"Unknown config keys: {sorted(invalid_keys)}",
                                    )
                                )
                            )
                            continue
                        state.session_config.update(cmd_args)
                        await ws.send(
                            json.dumps(
                                make_command_result(
                                    ref_id=msg_id,
                                    name="config",
                                    success=True,
                                    message="Session config updated",
                                    data=dict(state.session_config),
                                )
                            )
                        )
                    else:
                        await ws.send(
                            json.dumps(
                                make_command_result(
                                    ref_id=msg_id,
                                    name="config",
                                    success=True,
                                    message="Current session config",
                                    data=dict(state.session_config),
                                )
                            )
                        )

                elif cmd_name == "prompt":
                    if not state.active_session:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="no_session",
                                    message="No active session to get system prompt from",
                                )
                            )
                        )
                        continue

                    prompt_text = state.active_session.last_system_prompt
                    if prompt_text is None:
                        await ws.send(
                            json.dumps(
                                make_error(
                                    ref_id=msg_id,
                                    code="internal",
                                    message="System prompt not available for this session",
                                )
                            )
                        )
                        continue

                    await ws.send(
                        json.dumps(
                            make_command_result(
                                ref_id=msg_id,
                                name="prompt",
                                success=True,
                                message="Current system prompt",
                                data={"system_prompt": prompt_text},
                            )
                        )
                    )

                else:
                    await ws.send(
                        json.dumps(
                            make_error(
                                ref_id=msg_id,
                                code="internal",
                                message=f"Unknown command: {cmd_name!r}",
                            )
                        )
                    )

    except websockets.exceptions.ConnectionClosedOK:
        log.info("connection.closed_clean", project_id=state.active_project_id)
    except websockets.exceptions.ConnectionClosedError:
        log.info("connection.closed", project_id=state.active_project_id)
    except Exception as e:
        context_size = None
        context_limit = None
        if state.active_session:
            context_size = state.active_session.tokens_used
            context_limit = state.active_session.context_limit
        log.error(
            "connection.error",
            error=str(e),
            exc_type=type(e).__name__,
            context_size=context_size,
            context_limit=context_limit,
            exc_info=True,
        )
    finally:
        if state.active_loop_task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await state.active_loop_task
        if state.active_session:
            await request_queue.remove(state.active_session.session_id)


def _process_health_request(connection: ServerConnection, request: Request) -> Response | None:
    """Answer ``GET /health`` with 200 for container healthchecks.

    Returning a ``Response`` short-circuits the WebSocket handshake; returning
    ``None`` lets normal WS upgrade requests (including ``/ws``) proceed.
    """
    if urlparse(request.path).path == "/health":
        return connection.respond(HTTPStatus.OK, "ok\n")
    return None


async def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    # Find gateway.toml
    config_path = os.environ.get("GATEWAY_CONFIG", "gateway.toml")
    if not Path(config_path).exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent.parent.parent.parent
        candidate = script_dir / "gateway.toml"
        if candidate.exists():
            config_path = str(candidate)

    config = GatewayConfig.from_toml(config_path)
    _configure_logging(config.server.log_level)

    log.info("gateway.starting", host=config.server.host, port=config.server.port)

    # Prime the tokenizer from the default backend's local tokenizer.json in a
    # background thread (non-blocking). tokenizer_path is guaranteed non-empty by
    # the config validator; count_tokens() serves a char/4 estimate until it loads.
    tokenizer_path = config.resolve_backend(None).tokenizer_path
    assert tokenizer_path, "[model].tokenizer_path must be set (enforced by config validator)"
    init_tokenizer(tokenizer_path)

    # Initialize RAG client if enabled
    rag_client: RAGClient | None = None
    if config.rag.enabled:
        rag_client = RAGClient(base_url=config.rag.base_url, timeout=config.rag.timeout)

    # Initialize tool registry - update_tool_registry handles priority (env var > config file)
    update_tool_registry(
        web_config_brave_key=config.web.brave_api_key,
        search_enabled=config.web.search_enabled,
        search_count=config.web.search_count,
        fetch_enabled=config.web.fetch_enabled,
        fetch_timeout=config.web.fetch_timeout,
        fetch_max_redirects=config.web.fetch_max_redirects,
        fetch_blocked_domains=config.web.fetch_blocked_domains,
        user_agent=config.web.user_agent,
        rag_client=rag_client,
    )

    # Initialize components
    data_dir_path = config.data_dir_path
    data_dir = str(data_dir_path)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    session_store = SessionStore(data_dir_path)
    pruned = await session_store.prune_all(
        max_age_days=config.sessions.max_session_age_days,
        max_per_project=config.sessions.max_sessions_per_project,
    )
    if pruned:
        log.info("sessions.pruned_at_startup", count=pruned)

    session_manager = SessionManager()
    project_store = ProjectStore(data_dir)
    known_backends = {"local"} | set(config.model.backends.keys())
    agent_registry = AgentRegistry(data_dir, known_backends, get_known_tools())
    request_queue = RequestQueue()
    auth_manager = AuthManager(config.auth)

    # Context manager — stateless, shared across all connections
    ctx_cfg = config.context
    context_manager = ContextManager(
        pin_first_tokens=ctx_cfg.pin_first_tokens,
        pin_recent_tokens=ctx_cfg.pin_recent_tokens,
        reserved_response_tokens=ctx_cfg.reserved_response_tokens,
        prune_threshold=ctx_cfg.prune_threshold,
        compact_threshold=ctx_cfg.compact_threshold,
        warn_threshold=ctx_cfg.warn_threshold,
    )

    # Permission manager per project (cached)
    pm_cache: dict[str, PermissionManager] = {}

    def permission_manager_factory(project_id: str) -> PermissionManager:
        if project_id not in pm_cache:
            project_data_dir = data_dir_path / "projects" / project_id
            project_data_dir.mkdir(parents=True, exist_ok=True)
            pm_cache[project_id] = PermissionManager(
                config=config.permissions,
                project_data_dir=project_data_dir,
                approval_timeout=config.timeouts.approval_wait,
            )
        return pm_cache[project_id]

    # Connected clients for broadcast
    connected_clients: set[ServerConnection] = set()

    async def on_config_reload(scope: str, project_id: str | None, changed: list[str]) -> None:
        nonlocal config, rag_client
        try:
            new_config = GatewayConfig.from_toml(config_path)
            config = new_config
            auth_manager.reload(new_config.auth)
            for pm in pm_cache.values():
                pm.reload(new_config.permissions)
            # Update tool registry if web config changed
            if "web" in changed:
                # Close existing WebClient so the singleton is recreated with new settings
                await close_web_client()
                update_tool_registry(
                    web_config_brave_key=new_config.web.brave_api_key,
                    search_enabled=new_config.web.search_enabled,
                    search_count=new_config.web.search_count,
                    fetch_enabled=new_config.web.fetch_enabled,
                    fetch_timeout=new_config.web.fetch_timeout,
                    fetch_max_redirects=new_config.web.fetch_max_redirects,
                    fetch_blocked_domains=new_config.web.fetch_blocked_domains,
                    user_agent=new_config.web.user_agent,
                    rag_client=rag_client,
                )
                log.info(
                    "config.web_tool_registry_updated",
                    web_search_enabled="web_search" in get_known_tools(),
                    web_fetch_enabled="web_fetch" in get_known_tools(),
                )
            # Update RAG client if rag config changed
            if "rag" in changed:
                if rag_client:
                    await rag_client.close()
                rag_client = (
                    RAGClient(
                        base_url=new_config.rag.base_url, timeout=new_config.rag.timeout
                    )
                    if new_config.rag.enabled
                    else None
                )
                update_tool_registry(
                    web_config_brave_key=new_config.web.brave_api_key,
                    search_enabled=new_config.web.search_enabled,
                    search_count=new_config.web.search_count,
                    fetch_enabled=new_config.web.fetch_enabled,
                    fetch_timeout=new_config.web.fetch_timeout,
                    fetch_max_redirects=new_config.web.fetch_max_redirects,
                    fetch_blocked_domains=new_config.web.fetch_blocked_domains,
                    user_agent=new_config.web.user_agent,
                    rag_client=rag_client,
                )
                log.info(
                    "config.rag_tool_registry_updated",
                    rag_enabled=rag_client is not None,
                )
            log.info("config.reloaded", scope=scope, project_id=project_id)
        except Exception as e:
            log.error("config.reload_error", error=str(e))
            return

        msg = json.dumps(
            make_config_reloaded(
                scope=scope,
                project_id=project_id,
                changed=changed,
                message=f"{scope} config reloaded",
            )
        )
        for client_ws in list(connected_clients):
            with contextlib.suppress(Exception):
                await client_ws.send(msg)

    async def handle_ws(ws: ServerConnection) -> None:
        connected_clients.add(ws)
        try:
            await handle_connection(
                ws=ws,
                config=config,
                session_manager=session_manager,
                project_store=project_store,
                agent_registry=agent_registry,
                request_queue=request_queue,
                auth_manager=auth_manager,
                permission_manager_factory=permission_manager_factory,
                context_manager=context_manager,
                rag_client=rag_client,
                session_store=session_store,
            )
        finally:
            connected_clients.discard(ws)

    # Start config watcher
    watcher = ConfigWatcher(config_path, data_dir, on_config_reload)

    async with ws_serve(
        handle_ws,
        config.server.host,
        config.server.port,
        process_request=_process_health_request,
    ):
        log.info(
            "gateway.ready",
            host=config.server.host,
            port=config.server.port,
        )
        try:
            await asyncio.gather(
                asyncio.get_event_loop().create_future(),  # run forever
                watcher.watch(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            if rag_client:
                await rag_client.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
