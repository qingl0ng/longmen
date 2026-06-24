"""User input via prompt-toolkit."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.filters import completion_is_selected, has_completions
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prompt_toolkit.document import Document

    from .config import KeybindingsConfig


def _subsequence_score(query: str, text: str) -> int | None:
    """Return a quality score if query is a case-sensitive subsequence of text, else None.

    Score = sum of consecutive run lengths. Longer consecutive runs score higher
    than the same characters matched in scattered positions.
    """
    qi = 0
    score = 0
    run = 0
    for ch in text:
        if qi < len(query) and ch == query[qi]:
            qi += 1
            run += 1
            score += run  # consecutive bonus: run of 3 scores 1+2+3=6
        else:
            run = 0
    return score if qi == len(query) else None


def _score_matches(query: str, paths: set[str]) -> list[tuple[str, int]]:
    """Return (path, score) pairs for all paths where query is a subsequence of the filename.

    Only the final path component (filename) is matched against — directory
    segments are ignored. The full path is still returned as the completion text.
    Sorted best-first: higher score = tighter (more consecutive) match.
    Ties broken alphabetically.
    """
    results = [
        (p, s) for p in paths
        if (s := _subsequence_score(query, p.split("/")[-1])) is not None
    ]
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


class _CommandAndPathCompleter(Completer):
    """Tab-complete slash commands and @file paths."""

    def __init__(self) -> None:
        self._file_index: set[str] = set()

    def set_file_index(self, index: set[str]) -> None:
        self._file_index = index

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor

        # After /: complete command names
        if text.startswith("/") and " " not in text:
            partial = text[1:].lower()
            from .slash_commands import list_commands_with_descriptions
            for cmd, desc in list_commands_with_descriptions():
                if cmd.startswith(partial):
                    yield Completion("/" + cmd, start_position=-len(text), display_meta=desc)
            return

        # @file: subsequence match against file index (filename only)
        at_pos = text.rfind("@")
        if at_pos != -1 and (at_pos == 0 or not text[at_pos - 1].isalnum()):
            query = text[at_pos + 1:]
            if not query:
                return  # show nothing on bare @
            for path, _ in _score_matches(query, self._file_index):
                yield Completion(path, start_position=-len(query))


def _make_key_bindings() -> KeyBindings:
    """Build key bindings for the input session.

    Multiline: Escape then Enter (Meta+Enter).
    Most terminals can't distinguish Shift+Enter from plain Enter,
    so we use the universally supported Escape+Enter chord instead.
    """
    kb = KeyBindings()

    @kb.add("enter", filter=completion_is_selected)
    def _enter_apply_completion(event: KeyPressEvent) -> None:
        """Accept the highlighted completion item.

        Slash-command completions submit immediately (one keystroke).
        File-path completions insert a trailing space so the user can
        keep typing the rest of the message.
        """
        buf = event.current_buffer
        assert buf.complete_state is not None
        completion = buf.complete_state.current_completion
        assert isinstance(completion, Completion)
        buf.apply_completion(completion)
        if completion.text.startswith("/"):
            buf.validate_and_handle()
        else:
            buf.insert_text(" ")

    @kb.add("enter", filter=has_completions & ~completion_is_selected)
    def _enter_apply_first_completion(event: KeyPressEvent) -> None:
        """Dropdown open but no item navigated to — apply the first option.

        Same submit-vs-space logic as _enter_apply_completion.
        """
        buf = event.current_buffer
        assert buf.complete_state is not None
        completions = buf.complete_state.completions
        if completions:
            completion = completions[0]
            buf.apply_completion(completion)
            if completion.text.startswith("/"):
                buf.validate_and_handle()
            else:
                buf.insert_text(" ")
        else:
            buf.cancel_completion()

    @kb.add("escape", "enter")
    def _meta_enter(event: KeyPressEvent) -> None:
        """Escape then Enter inserts a newline without submitting."""
        event.current_buffer.insert_text("\n")

    return kb


class InputHandler:
    def __init__(self, keybindings_config: KeybindingsConfig | None = None) -> None:
        self._history = InMemoryHistory()
        self._completer = _CommandAndPathCompleter()
        self._kb = _make_key_bindings()
        self._session: PromptSession[str] = PromptSession(
            history=self._history,
            completer=self._completer,
            key_bindings=self._kb,
            multiline=False,
        )

    def set_file_index(self, index: set[str]) -> None:
        self._completer.set_file_index(index)

    async def get_input(
        self, prompt: HTML = HTML('<style fg="#00a86b">门</style> ')
    ) -> str:
        """Read one line of user input. Returns the stripped text."""
        try:
            width = shutil.get_terminal_size().columns
            sep = "─" * (width - 1)
            sys.stdout.write(sep + "\n")
            sys.stdout.flush()
            text = await self._session.prompt_async(prompt)
            sys.stdout.write(sep + "\n")
            sys.stdout.flush()
            return text.strip()
        except EOFError:
            raise  # Ctrl+D — let the caller handle
        except KeyboardInterrupt:
            raise  # Ctrl+C — let the caller handle
