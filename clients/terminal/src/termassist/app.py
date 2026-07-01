"""Application orchestrator — owns the main loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from . import approval as approval_mod
from .connection import Connection
from .file_attach import parse_attachments
from .history import History
from .input_handler import InputHandler
from .protocol import (
    make_abort,
    make_approval_response,
    make_command,
    make_project_list,
    make_project_select,
    make_project_upsert,
    make_prompt,
    make_session_resume,
)
from .renderer import Renderer
from .slash_commands import (
    UnknownCommandError,
    list_commands_with_descriptions,
    parse_slash_command,
)
from .state import StateMachine

if TYPE_CHECKING:
    from .config import ClientConfig

logger = logging.getLogger(__name__)

_CRASH_LOG = Path.home() / ".longmen" / "terminal" / "crash.log"


def _format_duration_ms(duration_ms: int) -> str:
    """Format a duration in milliseconds as a human-readable string.

    Examples: 0 → "0s", 5000 → "0m 5s", 305000 → "5m 5s"
    """
    if duration_ms == 0:
        return "0s"
    total_seconds = (duration_ms + 999) // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}m {seconds}s"


class App:
    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._state = StateMachine()
        self._history = History()
        self._renderer: Renderer | None = None
        self._conn: Connection | None = None
        self._abort_requested = False
        self._redirect_active: bool = False
        self._input: InputHandler | None = None
        self._file_index: set[str] = set()
        self._msg_log_file: IO[str] | None = None  # open file handle for message log
        # Set by `/project add` to the id being created. project_upsert only
        # creates the project — it does NOT activate it on the gateway session.
        # When the listener sees the matching creation-ack project_context, it
        # follows up with a real project_select so context/root actually switch.
        self._pending_created_project: str | None = None

    async def run(self) -> None:
        cfg = self._config
        self._renderer = Renderer(cfg.display)
        renderer = self._renderer

        try:
            conn = await Connection.connect(
                cfg.gateway.url,
                token=cfg.gateway.auth.token or None,
            )
        except Exception as e:
            renderer.render_error(
                {"code": "connection_failed", "message": str(e), "recoverable": False}
            )
            return

        self._conn = conn
        state = self._state

        # Open message log if configured
        if cfg.logging.message_log:
            log_path = Path(cfg.logging.message_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._msg_log_file = log_path.open("a")

        # Show welcome banner
        if conn.session_start_payload:
            renderer.render_session_start(conn.session_start_payload)

        conn.on_disconnect = renderer.render_reconnecting

        # Ensure a project is selected before entering the main loop
        await self._ensure_project_selected(conn, renderer)

        self._input = InputHandler(cfg.keybindings)
        # Transfer the file index captured during project selection
        self._input.set_file_index(self._file_index)

        # Capture terminal settings now — before TaskGroup starts and before
        # prompt_toolkit enters raw_mode.  This is the baseline we restore on
        # exit.  The atexit handler is a last-resort safety net: if the escape
        # watcher leaves the terminal in cbreak/raw mode and the finally block
        # somehow doesn't run (e.g. process killed mid-cleanup), this fires.
        import atexit
        import sys
        import termios

        _clean_terminal_settings = None
        with contextlib.suppress(Exception):
            _clean_terminal_settings = termios.tcgetattr(sys.stdin.fileno())

        if _clean_terminal_settings is not None:
            def _restore_terminal_atexit(
                fd: int = sys.stdin.fileno(),
                settings: Any = _clean_terminal_settings,
            ) -> None:
                with contextlib.suppress(Exception):
                    termios.tcsetattr(fd, termios.TCSANOW, settings)

            atexit.register(_restore_terminal_atexit)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._gateway_listener(conn, state, renderer))
                tg.create_task(self._input_loop(conn, state, renderer, self._input))
                tg.create_task(self._escape_watcher(conn, state, renderer))
        except* (EOFError, KeyboardInterrupt):
            pass
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception("Unhandled exception", exc_info=exc)
                self._log_crash(exc)
                renderer.render_error({
                    "code": "internal",
                    "message": f"An unexpected error occurred. Details logged to {_CRASH_LOG}",
                    "recoverable": False,
                })
        finally:
            await conn.close()
            if self._msg_log_file:
                self._msg_log_file.close()
                self._msg_log_file = None

    async def _ensure_project_selected(self, conn: Connection, renderer: Renderer) -> None:
        """Select a project before the main loop starts.

        Priority:
          1. Zero projects on the gateway → first-run bootstrap (create one)
          2. Exactly one project → auto-select it
          3. Multiple projects → always show the numbered picker (the saved
             config.project.id is not auto-loaded when several projects exist)
        """
        # Fetch the project list
        await conn.send(make_project_list())
        try:
            msg = await asyncio.wait_for(conn.recv(), timeout=10)
        except TimeoutError:
            renderer.render_warning(
                "Could not fetch project list (timeout). Continuing without a project."
            )
            return

        if msg["type"] != "project_registry":
            renderer.render_warning(f"Unexpected response while fetching projects: {msg['type']}")
            return

        projects: dict[str, Any] = msg["payload"].get("projects", {})

        if not projects:
            await self._bootstrap_first_project(conn, renderer)
            return

        # If only one project, auto-select it — there is nothing to choose.
        if len(projects) == 1:
            project_id = next(iter(projects))
            renderer.render_info(f"Auto-selected project: {project_id}")
            await self._do_select_project(conn, renderer, project_id)
            return

        # Multiple projects — always show the picker so the user chooses which to
        # load. We deliberately do NOT auto-select the saved config.project.id
        # here: with several projects the saved id is ambiguous (it only ever
        # tracks the bootstrap project, since runtime switches aren't persisted),
        # so silently loading it surprised users. Let them pick every time.
        renderer.print("\nAvailable projects:")
        project_list = list(projects.items())
        for i, (pid, info) in enumerate(project_list, 1):
            desc = info.get("description", "")
            desc_part = f"  — {desc}" if desc else ""
            renderer.print(f"  [{i}] {pid}{desc_part}")
        renderer.print()

        # Block for user input via executor so asyncio ping/pong can proceed
        loop = asyncio.get_event_loop()
        while True:
            try:
                choice = await loop.run_in_executor(
                    None, lambda: input("Select project (number or name): ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                renderer.render_info("No project selected.")
                return

            if not choice:
                continue

            # Accept a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(project_list):
                    project_id = project_list[idx][0]
                    break
                renderer.render_warning(f"Invalid selection. Enter 1–{len(project_list)}.")
                continue

            # Accept a project name directly
            if choice in projects:
                project_id = choice
                break

            renderer.render_warning(f"Unknown project {choice!r}. Try again.")

        await self._do_select_project(conn, renderer, project_id)

    async def _bootstrap_first_project(self, conn: Connection, renderer: Renderer) -> None:
        """First-run create-and-retry loop when the gateway has zero projects.

        Runs before the listener TaskGroup starts, so it sends project_upsert and
        recvs project_context / file_index inline. This is the one deliberate
        persistence point: on success it calls config.save() so the new project
        becomes the saved default. Empty id / EOF / KeyboardInterrupt aborts and
        returns "continue without a project".
        """
        renderer.render_info("No projects found on the gateway. Let's create one.")
        loop = asyncio.get_event_loop()

        def _ask(prompt: str) -> str:
            return input(prompt).strip()

        while True:
            try:
                project_id = await loop.run_in_executor(None, _ask, "Project id: ")
                if not project_id:
                    renderer.render_info("No project created. Continuing without a project.")
                    return
                root_path = await loop.run_in_executor(None, _ask, "Root path: ")
                description = await loop.run_in_executor(None, _ask, "Description (optional): ")
            except (EOFError, KeyboardInterrupt):
                renderer.render_info("No project created. Continuing without a project.")
                return

            project: dict[str, Any] = {"root_path": root_path}
            if description:
                project["description"] = description
            await conn.send(make_project_upsert(project_id, project))

            try:
                msg = await self._recv_skipping_broadcasts(conn, 10)
            except TimeoutError:
                renderer.render_warning(
                    "Timeout waiting for project creation. Continuing without a project."
                )
                return

            if msg["type"] == "project_context":
                pid = msg["payload"].get("project_id", project_id)
                # The project_upsert reply is only a creation ack — it does NOT
                # make the project active on the gateway session. Drain the
                # trailing file_index it sends, then issue a real project_select
                # so the session's active_project_id is set; otherwise the first
                # prompt fails with no_project_selected. project_select also runs
                # project-type detection + RAG setup gateway-side.
                with contextlib.suppress(TimeoutError):
                    await self._recv_skipping_broadcasts(conn, 5)
                await self._do_select_project(conn, renderer, pid)
                # This is the one deliberate persistence point: make the freshly
                # created project the saved default for next launch.
                self._config.project.id = pid
                self._config.save()
                return
            elif msg["type"] == "error":
                renderer.render_error(msg["payload"])
                # Loop back and re-prompt all three fields.
                continue
            else:
                renderer.render_warning(
                    f"Unexpected response to project creation: {msg['type']}."
                    " Continuing without a project."
                )
                return

    async def _recv_skipping_broadcasts(
        self, conn: Connection, timeout: float
    ) -> dict[str, Any]:
        """Recv the next message, skipping out-of-band broadcasts.

        The pre-listener inline flows (project select / first-run bootstrap)
        expect a specific ordered reply sequence. The config watcher can
        broadcast `config_reloaded` at any time — notably, `project_upsert`
        writes `project.toml`, which the watcher picks up and broadcasts, and
        that broadcast can land *between* the `project_context` and `file_index`
        of the reply (the file_index is built in an executor, so it can lag past
        the watcher's debounce). Drop such broadcasts here so the inline
        sequence stays aligned. The runtime listener ignores `config_reloaded`
        too, so nothing is lost. Raises `TimeoutError` if no relevant message
        arrives within `timeout`.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            msg = await asyncio.wait_for(conn.recv(), timeout=remaining)
            if msg["type"] == "config_reloaded":
                continue
            return msg

    async def _do_select_project(
        self, conn: Connection, renderer: Renderer, project_id: str
    ) -> None:
        """Send project_select and wait for project_context."""
        await conn.send(make_project_select(project_id))
        try:
            msg = await self._recv_skipping_broadcasts(conn, 10)
        except TimeoutError:
            renderer.render_warning(f"Timeout waiting for project context for {project_id!r}.")
            return

        if msg["type"] == "project_context":
            self._config.project.id = project_id
            payload = msg["payload"]
            agents = payload.get("agents", {})
            agent_names = list(agents.keys())
            if agent_names:
                renderer.render_info(f"Project: {project_id}  |  Agents: {', '.join(agent_names)}")
            else:
                renderer.render_info(f"Project: {project_id}")
            last_session = payload.get("last_session")
            # Drain file_index — gateway sends it immediately after project_context.
            # If the next message is not file_index (e.g. the gateway doesn't send one),
            # stash it so subsequent recv calls still see it.
            stashed: dict[str, Any] | None = None
            try:
                next_msg = await self._recv_skipping_broadcasts(conn, 5)
                if next_msg["type"] == "file_index":
                    self._file_index = set(next_msg["payload"]["files"])
                    if self._input is not None:
                        self._input.set_file_index(self._file_index)
                else:
                    stashed = next_msg
            except TimeoutError:
                pass  # gateway did not send file_index — ignore

            async def _recv_next(timeout: float) -> dict[str, Any]:
                nonlocal stashed
                if stashed is not None:
                    msg = stashed
                    stashed = None
                    return msg
                return await self._recv_skipping_broadcasts(conn, timeout)

            if last_session:
                self._state.last_session_id = last_session["session_id"]
                await conn.send(make_session_resume(last_session["session_id"]))
                try:
                    resume_msg = await _recv_next(timeout=10)
                    if resume_msg["type"] == "session_resumed":
                        self._state.session_id = resume_msg["payload"]["session_id"]
                        if resume_msg["payload"].get("incomplete_turn"):
                            renderer.render_info(
                                "Last request was interrupted."
                                " Your previous prompt was not answered."
                            )
                        # Gateway sends session_history immediately after session_resumed
                        try:
                            history_msg = await _recv_next(timeout=10)
                            if history_msg["type"] == "session_history":
                                self._render_session_history(history_msg["payload"], renderer)
                        except TimeoutError:
                            pass
                    elif resume_msg["type"] == "error":
                        renderer.render_error(resume_msg["payload"])
                except TimeoutError:
                    renderer.render_warning("Timeout waiting for session resume confirmation.")
        elif msg["type"] == "error":
            renderer.render_error(msg["payload"])
        else:
            renderer.render_warning(f"Unexpected response to project_select: {msg['type']}")

    def _render_session_history(self, payload: dict[str, Any], renderer: Renderer) -> None:
        turns = payload.get("turns", [])
        if not turns:
            return
        renderer.render_info(f"── Resuming session ({len(turns)} turns) ──")
        for turn in turns:
            role = turn.get("role")
            content = turn.get("content") or ""
            if role == "user":
                renderer.render_user_prompt(content)
            elif role == "assistant":
                if turn.get("is_summary"):
                    renderer.render_info("[Compacted history summary]")
                renderer.render_history_assistant(content)
        renderer.render_info("── Continue below ──")

    def _log_message(self, msg: dict[str, Any], direction: str = "recv") -> None:
        """Append a raw message to the message log (jsonl format)."""
        if self._msg_log_file:
            entry = {"t": time.time(), "dir": direction, "msg": msg}
            self._msg_log_file.write(json.dumps(entry) + "\n")
            self._msg_log_file.flush()

    async def _gateway_listener(
        self,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
    ) -> None:
        async for msg in conn.recv_iter():
            self._log_message(msg)
            msg_type = msg["type"]
            payload = msg["payload"]

            match msg_type:
                case "stream_chunk":
                    renderer.stop_spinner()
                    if state.current != "streaming":
                        state.go("streaming")
                    if not state.session_id and payload.get("session_id"):
                        state.session_id = payload["session_id"]
                    renderer.render_stream_chunk(
                        payload.get("delta", ""), payload.get("role", "text")
                    )

                case "stream_end":
                    renderer.stop_spinner()
                    renderer.render_stream_end(payload)
                    if payload.get("aborted") and not self._redirect_active:
                        renderer.render_info("(Generation aborted)")
                    sid = payload.get("session_id")
                    if sid:
                        state.session_id = sid
                    state.go("idle")

                case "approval_request":
                    renderer.stop_spinner()
                    state.go("awaiting_approval")

                    renderer.render_approval_request(payload)
                    cfg = self._config
                    decision, edited_command = await approval_mod.prompt_approval(
                        payload,
                        renderer,
                        auto_approve_safe=cfg.display.auto_approve_safe,
                    )

                    resp = make_approval_response(
                        ref_id=msg["id"],
                        decision=decision,
                        edited_command=edited_command,
                    )
                    await conn.send(resp)

                    state.go("executing_tool")

                case "tool_output":
                    if not state.session_id and payload.get("session_id"):
                        state.session_id = payload["session_id"]
                    renderer.render_tool_output(payload)

                case "error":
                    renderer.stop_spinner()
                    # A failed `/project add` (e.g. bad root_path) replies with an
                    # error instead of project_context — drop the pending select.
                    self._pending_created_project = None
                    renderer.render_error(payload)
                    if payload.get("recoverable", True):
                        with contextlib.suppress(Exception):
                            state.force_transition("idle")
                    else:
                        state.force_transition("disconnected")

                case "model_waiting":
                    # Entry guard: only act from a pre-stream state. Ignore a stray
                    # model_waiting that arrives once streaming/tool/approval is
                    # underway — go()'s force-fallback would otherwise tear down a
                    # live stream and pop the spinner back up.
                    if state.current in {"idle", "queued", "waiting_for_model"}:
                        if not state.session_id and payload.get("session_id"):
                            state.session_id = payload["session_id"]
                        if state.current != "waiting_for_model":
                            # One-time entry: clear the running "Thinking" spinner
                            # line first so the persistent note prints cleanly, then
                            # restart the spinner via the countdown below.
                            renderer.stop_spinner()
                            state.go("waiting_for_model")
                            renderer.render_warning(
                                "Model is not responding yet — "
                                "waiting for it to come online..."
                            )
                        waited = payload.get("waited_seconds", 0)
                        max_wait = payload.get("max_wait_seconds", 0)
                        countdown = (
                            f"Waiting for model — {waited}s / {max_wait}s"
                            if max_wait
                            else f"Waiting for model — {waited}s"
                        )
                        renderer.set_spinner_message(countdown)

                case "status":
                    renderer.render_status(payload)

                case "queue_position":
                    try:
                        if state.current == "idle":
                            state.transition("queued")
                    except Exception:
                        pass
                    renderer.render_queue_position(payload)

                case "file_index":
                    self._file_index = set(payload["files"])
                    if self._input is not None:
                        self._input.set_file_index(self._file_index)

                case "project_context":
                    pid_ctx = payload.get("project_id", "")
                    if pid_ctx and pid_ctx == self._pending_created_project:
                        # Creation ack from `/project add`. project_upsert created
                        # the project but did NOT activate it on the gateway
                        # session, so context/root would stay on the previous
                        # project. Follow up with a real project_select; its
                        # project_context (below) does the actual switch render.
                        self._pending_created_project = None
                        await conn.send(make_project_select(pid_ctx))
                        continue
                    # Response to /project <id> sent at runtime
                    self._config.project.id = payload.get("project_id", self._config.project.id)
                    agents = payload.get("agents", {})
                    agent_names = list(agents.keys())
                    pid = payload.get("project_id", "")
                    if agent_names:
                        renderer.render_info(
                            f"Switched to project: {pid}  |  Agents: {', '.join(agent_names)}"
                        )
                    else:
                        renderer.render_info(f"Switched to project: {pid}")
                    last_session = payload.get("last_session")
                    if last_session:
                        state.last_session_id = last_session["session_id"]
                        await conn.send(make_session_resume(last_session["session_id"]))

                case "session_resumed":
                    state.session_id = payload["session_id"]
                    if payload.get("incomplete_turn"):
                        renderer.render_info(
                            "Last request was interrupted. Your previous prompt was not answered."
                        )

                case "session_history":
                    self._render_session_history(payload, renderer)

                case "reconnected":
                    if self._config.project.id:
                        await conn.send(make_project_select(self._config.project.id))

                case "project_registry":
                    # Response to /project (no args) sent at runtime
                    projects = payload.get("projects", {})
                    if not projects:
                        renderer.render_info("No projects found.")
                    else:
                        renderer.print("Projects:")
                        for pid, info in projects.items():
                            desc = info.get("description", "")
                            active = " [current]" if pid == self._config.project.id else ""
                            renderer.print(f"  {pid}{active}" + (f"  — {desc}" if desc else ""))

                case "session_start":
                    renderer.render_session_start(payload)
                    renderer.render_info("Reconnected.")
                    renderer.reset_reconnect_counter()
                    with contextlib.suppress(Exception):
                        state.force_transition("idle")

                case "command_result":
                    name = payload.get("name", "")
                    if name == "prompt":
                        prompt_text = payload.get("data", {}).get("system_prompt", "")
                        renderer.print(prompt_text)

                case _:
                    logger.debug("Unhandled message type: %s", msg_type)

    async def _input_loop(
        self,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
        input_handler: InputHandler,
    ) -> None:
        _ctrl_c_time = 0.0

        while True:
            # Wait for idle before accepting input
            if state.current != "idle":
                try:
                    await asyncio.wait_for(state.wait_for("idle"), timeout=300)
                except TimeoutError:
                    continue

            if self._redirect_active:
                await asyncio.sleep(0)
                continue

            try:
                text = await input_handler.get_input()
            except EOFError:
                await self._graceful_shutdown(conn, state, renderer)
                return
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - _ctrl_c_time < 2.0:
                    await self._graceful_shutdown(conn, state, renderer)
                    raise KeyboardInterrupt from None
                else:
                    _ctrl_c_time = now
                    renderer.render_info("Press Ctrl+C again to quit")
                continue

            if not text:
                continue

            if text.startswith("/"):
                await self._handle_slash(text, conn, state, renderer)
                continue

            try:
                cleaned_text, attachments = parse_attachments(text, file_index=self._file_index)
            except ValueError as e:
                renderer.render_error(
                    {"code": "attachment_too_large", "message": str(e), "recoverable": True}
                )
                continue

            content = attachments + [{"type": "text", "text": cleaned_text}]
            msg = make_prompt(
                session_id=state.session_id,
                project_id=self._config.project.id,
                content=content,
            )
            self._log_message(msg, direction="send")
            await conn.send(msg)
            state.go("queued")  # prevent re-entering input before gateway responds
            renderer.start_spinner()

    async def _handle_slash(
        self,
        text: str,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
    ) -> None:
        try:
            cmd = parse_slash_command(text)
        except UnknownCommandError as e:
            renderer.render_error({
                "code": "unknown_command",
                "message": str(e) + (
                    f" — did you mean: {', '.join(e.suggestions)}?" if e.suggestions else ""
                ),
                "recoverable": True,
            })
            return

        if cmd.name == "quit":
            await self._graceful_shutdown(conn, state, renderer)
            raise EOFError

        elif cmd.name == "help":
            lines = [f"  /{name:<12} {desc}" for name, desc in list_commands_with_descriptions()]
            renderer.print("\n".join(lines))

        elif cmd.name == "config":
            if not cmd.args:
                cfg = self._config
                renderer.print(f"gateway.url = {cfg.gateway.url!r}")
                renderer.print(f"gateway.auth.mode = {cfg.gateway.auth.mode!r}")
                renderer.print(f"project.id = {cfg.project.id!r}")
                renderer.print(f"display.theme = {cfg.display.theme!r}")
                renderer.print(f"display.show_thinking = {cfg.display.show_thinking}")
                renderer.print(f"display.show_token_bar = {cfg.display.show_token_bar}")
                renderer.print(f"display.max_output_lines = {cfg.display.max_output_lines}")
                renderer.print(f"display.auto_approve_safe = {cfg.display.auto_approve_safe}")
            else:
                for k, v in cmd.args.items():
                    try:
                        self._config.set_value(k, str(v))
                        renderer.render_info(f"Config updated: {k} = {v}")
                    except KeyError as e:
                        renderer.render_error(
                            {"code": "config_error", "message": str(e), "recoverable": True}
                        )
                self._config.save()

        elif cmd.name == "new":
            state.session_id = None
            renderer.render_info("Started new conversation.")
            await conn.send(make_command("new"))

        elif cmd.name == "project":
            if cmd.args.get("verb") == "add":
                # Create/update a project. Fire-and-forget: the listener's
                # project_context handler auto-switches in-memory (no config.save()).
                # A bad root_path surfaces via the listener's error handler.
                if not cmd.args.get("id") or not cmd.args.get("root_path"):
                    renderer.render_error({
                        "code": "usage",
                        "message": "Usage: /project add <id> <root_path> [description]",
                        "recoverable": True,
                    })
                    return
                project: dict[str, Any] = {"root_path": cmd.args["root_path"]}
                if cmd.args.get("description"):
                    project["description"] = cmd.args["description"]
                # Remember which project we're creating so the listener can issue
                # a real project_select once the creation-ack project_context
                # arrives (upsert alone does not switch the gateway session).
                self._pending_created_project = cmd.args["id"]
                await conn.send(make_project_upsert(cmd.args["id"], project))
            elif "id" in cmd.args:
                # Switch to a specific project — send project_select.
                # _gateway_listener handles the project_context response.
                await conn.send(make_project_select(cmd.args["id"]))
            else:
                # List all projects — _gateway_listener handles project_registry response.
                await conn.send(make_project_list())

        elif cmd.is_local:
            renderer.render_info(f"(/{cmd.name} is a local command)")

        else:
            msg = make_command(cmd.name, cmd.args or {})
            await conn.send(msg)

    async def _handle_redirect(
        self,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
    ) -> None:
        self._redirect_active = True
        try:
            if state.session_id:
                await conn.send(make_abort(state.session_id))
            renderer.start_spinner(message="Stopping")
            try:
                await asyncio.wait_for(state.wait_for("idle"), timeout=10.0)
            except TimeoutError:
                state.force_transition("idle")
            finally:
                renderer.stop_spinner()

            # _redirect_active stays True here so _input_loop doesn't enter get_input()
            renderer.render_redirect_hint()
            assert self._input is not None
            try:
                redirect_text = await self._input.get_input()
            except (KeyboardInterrupt, EOFError):
                return
            if not redirect_text:
                return

            content = [{"type": "text", "text": redirect_text}]
            await conn.send(make_prompt(state.session_id, self._config.project.id, content))
            renderer.render_user_prompt(redirect_text)
            state.go("queued")  # prevent _input_loop from re-entering before gateway responds
            renderer.start_spinner()
        finally:
            self._redirect_active = False

    async def _escape_watcher(
        self,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
    ) -> None:
        """Raw-stdin escape watcher. Active only while streaming/tool/approval."""
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        loop = asyncio.get_event_loop()

        while True:
            await state.wait_for_any(
                {"streaming", "executing_tool", "awaiting_approval", "waiting_for_model"}
            )

            try:
                old_settings = termios.tcgetattr(fd)
            except termios.error:
                # stdin is not a TTY (e.g. piped input in tests) — wait and retry
                await asyncio.sleep(0.1)
                continue

            try:
                tty.setcbreak(fd)
                byte = await loop.run_in_executor(None, lambda: sys.stdin.buffer.read(1))
            finally:
                # Use TCSANOW (apply immediately) rather than TCSADRAIN (wait for
                # output to drain) so the restore is not blocked by the executor
                # thread that may still be holding the fd open.
                termios.tcsetattr(fd, termios.TCSANOW, old_settings)

            if byte != b"\x1b":
                continue

            # Disambiguate lone Escape from escape sequences (arrow keys, etc.)
            # 50ms window: if more bytes follow, it's an escape sequence — skip.
            try:
                next_byte = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: sys.stdin.buffer.read(1)),
                    timeout=0.05,
                )
                _ = next_byte  # discard — this was an escape sequence, not lone Escape
                continue
            except TimeoutError:
                pass  # lone Escape confirmed

            if state.current not in {
                "streaming",
                "executing_tool",
                "awaiting_approval",
                "waiting_for_model",
            }:
                continue

            await self._handle_redirect(conn, state, renderer)

    async def _graceful_shutdown(
        self,
        conn: Connection,
        state: StateMachine,
        renderer: Renderer,
    ) -> None:
        if state.current == "streaming" and state.session_id:
            await conn.send(make_abort(state.session_id))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(state.wait_for("idle"), timeout=5)
        renderer.render_info("Disconnecting...")
        await conn.close()

    def _log_crash(self, exc: Exception) -> None:
        import traceback
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _CRASH_LOG.open("a") as f:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
