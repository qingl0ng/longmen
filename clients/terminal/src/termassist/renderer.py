"""Rich-based terminal output rendering."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .themes import get_theme
from .token_display import render_token_bar

if TYPE_CHECKING:
    from .config import DisplayConfig

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _truncate(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    extra = len(lines) - limit
    return lines[:limit] + [f"... ({extra} more lines)"]


class Renderer:
    def __init__(self, display_config: DisplayConfig, theme_name: str | None = None) -> None:
        self._cfg = display_config
        self._theme = get_theme(theme_name or display_config.theme)
        self._console = Console()
        self._stream_buffer: list[str] = []
        self._live: Live | None = None
        self._spinner_task: asyncio.Task[None] | None = None
        self._spinner_active = False
        self._spinner_message = "Thinking"
        self._reconnect_attempt = 0

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_spinner(self, message: str = "Thinking") -> None:
        """Show an animated spinner while waiting for the first response token."""
        if self._spinner_active:
            return
        self._spinner_message = message
        self._spinner_active = True
        # RuntimeError: no event loop running — skip animation
        with contextlib.suppress(RuntimeError):
            self._spinner_task = asyncio.get_running_loop().create_task(self._run_spinner())

    def set_spinner_message(self, message: str) -> None:
        """Update the live spinner text in place. Starts the spinner if not running."""
        self._spinner_message = message
        if not self._spinner_active:
            self.start_spinner(message)

    def stop_spinner(self) -> None:
        """Stop and erase the spinner. Safe to call even if not running."""
        if not self._spinner_active:
            return
        self._spinner_active = False
        if self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None
        # Clear the spinner line synchronously before any subsequent print.
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    async def _run_spinner(self) -> None:
        i = 0
        try:
            while True:
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                sys.stdout.write(f"\r{frame} {self._spinner_message}...")
                sys.stdout.flush()
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass  # line is cleared synchronously in stop_spinner()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def render_stream_chunk(self, delta: str, role: str) -> None:
        if role == "thinking":
            if self._cfg.show_thinking:
                t = Text(delta, style=self._theme.get("thinking", "dim italic"))
                self._console.print(t, end="")
        elif role == "tool_call":
            self._stop_live()
            # Show tool name on first delta that carries it (function.name field)
            try:
                import json as _json
                parsed = _json.loads(delta)
                name = parsed.get("function", {}).get("name", "")
                if name:
                    t = Text(f"  ⚙ {name}...", style=self._theme.get("tool_call", "dim cyan"))
                    self._console.print(t)
            except Exception:
                pass
        else:  # text
            if not self._stream_buffer:
                self._console.print(
                    Text("Assistant: ", style=self._theme.get("assistant", "bold blue"))
                )
            if self._live is None:
                self._live = Live(console=self._console, auto_refresh=False)
                self._live.start()
            self._stream_buffer.append(delta)
            self._live.update(Markdown("".join(self._stream_buffer)))
            self._live.refresh()

    def flush_stream(self) -> str:
        """Flush accumulated streaming text and return it."""
        text = "".join(self._stream_buffer)
        self._stream_buffer.clear()
        return text

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def render_session_start(self, payload: dict[str, Any]) -> None:
        model_info = payload.get("default_model", {})
        if model_info:
            model_name = model_info.get("model_name", "unknown")
        else:
            model_name = payload.get("model", "unknown")
        version = payload.get("gateway_version", "?")
        caps = ", ".join(payload.get("capabilities", []))
        banner = Text()
        banner.append("\u26a1 Connected to Gateway ", style="bold")
        banner.append(f"v{version}", style="bold cyan")
        banner.append(" | Model: ")
        banner.append(model_name, style="bold green")
        if caps:
            banner.append(f"\nCapabilities: {caps}", style="dim")
        self._console.print(Panel(banner, expand=False))

    def render_stream_end(self, payload: dict[str, Any]) -> None:
        self._stop_live()
        self.flush_stream()

        finish_reason = payload.get("finish_reason")
        tool_calls_made = payload.get("tool_calls_made", 0)

        # Warn on unexpected finish reasons
        if finish_reason and finish_reason not in ("stop", "tool_calls", None):
            t = Text(
                f"  finish_reason={finish_reason}  tool_calls={tool_calls_made}",
                style=self._theme.get("warning", "bold yellow"),
            )
            self._console.print(t)
        elif self._cfg.show_thinking and finish_reason:
            # In debug/thinking mode, always show the finish metadata
            t = Text(
                f"  finish_reason={finish_reason}  tool_calls={tool_calls_made}",
                style="dim",
            )
            self._console.print(t)

        if self._cfg.show_token_bar:
            usage = payload.get("usage", {})
            budget = usage.get("context_budget", {}) if usage else {}
            if budget:
                bar = render_token_bar(
                    budget.get("used", 0),
                    budget.get("limit", 32000),
                    self._theme,
                )
                self._console.print(bar, justify="right")

    def _render_tool_panel(self, payload: dict[str, Any], max_lines: int | None = None) -> None:
        command = payload.get("command", "")
        exit_code = payload.get("exit_code", 0)
        stdout = payload.get("stdout", "")
        stderr = payload.get("stderr", "")
        duration_ms = payload.get("duration_ms", 0)
        truncated = payload.get("truncated", False)

        limit = max_lines if max_lines is not None else self._cfg.max_output_lines
        stdout_lines = _truncate(stdout.splitlines(), limit)
        stderr_lines = _truncate(stderr.splitlines(), limit)

        output = ""
        if stdout_lines:
            output += "\n".join(stdout_lines)
        if stderr_lines:
            if output:
                output += "\n[stderr]\n"
            output += "\n".join(stderr_lines)
        if truncated:
            output += "\n(output truncated — full output sent to model)"

        exit_style = "green" if exit_code == 0 else "red"
        title = Text()
        title.append("$ ")
        title.append(command, style="bold")
        title.append(f"  exit: {exit_code}", style=exit_style)
        if duration_ms:
            title.append(f"  {duration_ms}ms", style="dim")

        self._console.print(
            Panel(
                output or "(no output)",
                title=title,
                border_style=self._theme.get("tool_output_title", "white"),
            )
        )

    def render_tool_output(self, payload: dict[str, Any]) -> None:
        tool = payload.get("tool", "")
        command = payload.get("command", "")
        stdout = payload.get("stdout", "")

        # Full panel tools — unchanged behavior
        if tool in ("tree", "web_search", "rag_search", "shell"):
            self._render_tool_panel(payload)
            return

        # One-liner tools
        if tool == "list_dir":
            self._console.print(Text(f" ⎿  List {command}", style="dim"))
            return
        if tool == "read_file":
            n = len(stdout.splitlines())
            self._console.print(Text(f" ⎿  Read {command} ({n} lines)", style="dim"))
            return
        if tool == "grep":
            self._console.print(Text(f" ⎿  Grep {command}", style="dim"))
            return
        if tool == "symbols":
            self._console.print(Text(f" ⎿  Symbols {command}", style="dim"))
            return
        if tool == "write_file":
            self._console.print(Text(f" ⎿  Write {command}", style="dim"))
            return
        if tool == "search_replace":
            self._console.print(Text(f" ⎿  Edit {command}", style="dim"))
            return
        if tool == "git_status":
            self._console.print(Text(" ⎿  git status", style="dim"))
            return
        if tool == "git_diff":
            self._console.print(Text(f" ⎿  git diff {command}", style="dim"))
            return
        if tool == "git_log":
            self._console.print(Text(f" ⎿  git log {command}", style="dim"))
            return
        if tool == "git_add":
            self._console.print(Text(f" ⎿  git add {command}", style="dim"))
            return
        if tool == "git_commit":
            self._console.print(Text(f" ⎿  git commit {command}", style="dim"))
            return
        if tool == "web_fetch":
            self._console.print(Text(f" ⎿  Fetch {command}", style="dim"))
            return

        # Everything else: panel with stdout truncated to 10 lines
        self._render_tool_panel(payload, max_lines=10)

    def render_approval_request(self, payload: dict[str, Any]) -> None:
        """Render approval request panel (without the interactive menu — that's approval.py)."""
        command = payload.get("command", "")
        risk = payload.get("risk", "moderate")
        context = payload.get("context", "")

        risk_styles = {
            "safe": self._theme.get("approval_safe", "bold green"),
            "moderate": self._theme.get("approval_moderate", "bold yellow"),
            "destructive": self._theme.get("approval_destructive", "bold red"),
        }
        border_style = risk_styles.get(risk, "bold yellow")

        content = Text()
        content.append(command + "\n", style="bold")
        content.append(f"Risk: {risk}", style=border_style)
        if context:
            content.append(f"\nContext: {context}", style="dim")

        self._console.print(
            Panel(content, title="Shell Command", border_style=border_style)
        )

    def render_error(self, payload: dict[str, Any]) -> None:
        self._stop_live()
        code = payload.get("code", "unknown")
        message = payload.get("message", "An error occurred")

        content = Text()
        content.append(f"[{code}] ", style="bold red")
        content.append(message)

        self._console.print(Panel(content, title="Error", border_style="red"))

    def render_status(self, payload: dict[str, Any]) -> None:
        state = payload.get("state", "idle")
        pos = payload.get("queue_position", 0)
        usage = payload.get("token_usage", {})
        duration_ms = payload.get("duration_ms")
        tokens_used = payload.get("tokens_used")
        tool_call_count = payload.get("tool_call_count")

        text = Text()
        text.append(f"Status: {state}", style=self._theme.get("status", "dim"))
        if pos:
            text.append(f" | Queue: {pos}", style=self._theme.get("warning", "yellow"))
        if duration_ms is not None:
            if duration_ms == 0:
                duration_str = "0s"
            else:
                total_seconds = (duration_ms + 999) // 1000
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                duration_str = f"{minutes}m {seconds}s"
            text.append(f" | Duration: {duration_str}", style=self._theme.get("status", "dim"))
        if tokens_used is not None:
            text.append(f" | Tokens: {tokens_used:,}", style=self._theme.get("status", "dim"))
        if tool_call_count is not None:
            text.append(
                f" | Tool Call Count: {tool_call_count}",
                style=self._theme.get("status", "dim"),
            )
        self._console.print(text)

        if usage and self._cfg.show_token_bar:
            bar = render_token_bar(
                usage.get("used", 0),
                usage.get("limit", 32000),
                self._theme,
            )
            self._console.print(bar, justify="right")

    def render_queue_position(self, payload: dict[str, Any]) -> None:
        pos = payload.get("position", 0)
        t = Text(f"Queued: position {pos}", style=self._theme.get("warning", "yellow"))
        self._console.print(t)

    def render_redirect_hint(self) -> None:
        """Show a dim hint above the redirect input prompt."""
        self._console.print(
            "Tell assistant what to do instead",
            style="dim",
        )

    def render_user_prompt(self, text: str) -> None:
        t = Text()
        t.append("You: ", style=self._theme.get("user", "bold green"))
        t.append(text)
        self._console.print(t)

    def render_history_assistant(self, text: str) -> None:
        self._console.print(
            Text("Assistant: ", style=self._theme.get("assistant", "bold blue"))
        )
        self._console.print(Markdown(text))

    def render_info(self, message: str) -> None:
        self._console.print(Text(message, style=self._theme.get("info", "dim")))

    def render_warning(self, message: str) -> None:
        self._console.print(Text(message, style=self._theme.get("warning", "bold yellow")))

    def render_reconnecting(self) -> None:
        self._reconnect_attempt += 1
        msg = f"Connection lost. Reconnecting... (attempt {self._reconnect_attempt})"
        if self._reconnect_attempt > 1:
            self._console.file.write("\033[1A\r\033[2K")
            self._console.file.flush()
        self._console.print(Text(msg, style=self._theme.get("warning", "bold yellow")))

    def reset_reconnect_counter(self) -> None:
        self._reconnect_attempt = 0

    def print(self, *args: Any, **kwargs: Any) -> None:
        self._console.print(*args, **kwargs)
