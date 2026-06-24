"""Interactive approval dialog for tool approval decisions."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import Renderer


_MENU = """\
  [y] Yes — run this command
  [n] No — skip
  [s] Yes, remember for this session
  [a] Always allow (persist across sessions)
  [e] Edit command before running"""

_KEY_TO_DECISION = {
    "y": "yes",
    "n": "no",
    "s": "yes_session",
    "a": "yes_always",
    "e": "edit",
}


async def prompt_approval(
    payload: dict[str, Any],
    renderer: Renderer,
    auto_approve_safe: bool = True,
) -> tuple[str, str | None]:
    """Present the approval dialog and return (decision, edited_command).

    Returns immediately with ('yes', None) for safe risk when auto_approve_safe is True.
    """
    risk = payload.get("risk", "moderate")
    command = payload.get("command", "")

    if auto_approve_safe and risk == "safe":
        renderer.render_info(f"Auto-approved (safe): {command}")
        return "yes", None

    renderer.print(_MENU)

    # Read a single keypress
    key = await _read_key()

    decision = _KEY_TO_DECISION.get(key.lower(), "no")

    if decision == "edit":
        edited = await _prompt_edit(command, renderer)
        return "edit", edited

    return decision, None


async def _read_key() -> str:
    """Read a single keypress without requiring Enter."""
    loop = asyncio.get_running_loop()

    def _read() -> str:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    try:
        return await loop.run_in_executor(None, _read)
    except Exception:
        # Fallback for non-TTY environments (tests)
        line = await loop.run_in_executor(None, input, "Decision [y/n/s/a/e]: ")
        return line.strip()[:1] or "n"


async def _prompt_edit(command: str, renderer: Renderer) -> str:
    """Prompt the user to edit the command."""
    loop = asyncio.get_running_loop()

    try:
        from prompt_toolkit import PromptSession

        session: PromptSession[str] = PromptSession()
        edited = await session.prompt_async("Edit command: ", default=command)
        return edited
    except Exception:
        edited = await loop.run_in_executor(None, input, f"Edit command [{command}]: ")
        return edited.strip() or command
