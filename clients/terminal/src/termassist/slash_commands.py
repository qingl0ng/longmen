"""Slash command registry and dispatch."""

from __future__ import annotations

import difflib
import shlex
from dataclasses import dataclass, field
from typing import Any

_PROJECT_VERBS = {"add"}  # future: {"add", "rm"}


class UnknownCommandError(Exception):
    def __init__(self, name: str, suggestions: list[str]) -> None:
        super().__init__(f"Unknown command: {name!r}")
        self.name = name
        self.suggestions = suggestions


@dataclass
class SlashCommand:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    is_local: bool = False

    def to_message(self) -> dict[str, Any]:
        """Return the payload dict for a command message (non-local commands)."""
        return {"name": self.name, "args": self.args}



# Registry: name -> (is_local, arg_parser_hint, description)
# arg_parser_hint: None = no args, "key=val" = key=value pairs, "id" = positional id,
#                  "agent" = name + quoted prompt, "raw" = raw string
_COMMANDS: dict[str, dict[str, Any]] = {
    "new": {
        "local": False, "args": None,
        "description": "clear current context and start a new session",
    },
    "compact": {
        "local": False, "args": None,
        "description": "compact the conversation history",
    },
    "undo":        {"local": False, "args": None,    "description": "undo the last turn"},
    "refresh": {
        "local": False, "args": None,
        "description": "refresh the agent's context",
    },
    "permissions": {"local": False, "args": None,    "description": "show tool permissions"},
    "config": {
        "local": True, "args": "key=val",
        "description": "show or set configuration values",
    },
    "project": {
        "local": False, "args": "project",
        "description": "list, switch, or add (/project add <id> <root_path> [description])",
    },
    "agent":       {"local": False, "args": "agent", "description": "invoke a named agent"},
    "status":      {"local": False, "args": None,    "description": "show session status"},
    "quit":        {"local": True,  "args": None,    "description": "disconnect and exit"},
    "help":        {"local": True,  "args": None,    "description": "show available commands"},
    "prompt": {
        "local": False, "args": None,
        "description": "show the current system prompt",
    },
}


def parse_slash_command(text: str) -> SlashCommand:
    """Parse a slash command string into a SlashCommand.

    Raises UnknownCommandError if the command name is not recognized.
    """
    text = text.strip()
    if not text.startswith("/"):
        raise ValueError("Not a slash command")

    # Split off the command name
    parts = text[1:].split(None, 1)
    name = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if name not in _COMMANDS:
        suggestions = difflib.get_close_matches(name, _COMMANDS.keys(), n=3, cutoff=0.5)
        raise UnknownCommandError(name, suggestions)

    spec = _COMMANDS[name]
    is_local = spec["local"]
    arg_style = spec["args"]

    args: dict[str, Any] = {}
    if rest and arg_style == "key=val":
        # Parse key=value pairs
        for token in rest.split():
            if "=" in token:
                k, v = token.split("=", 1)
                args[k.strip()] = v.strip()
            else:
                args[token.strip()] = True
    elif rest and arg_style == "id":
        args["id"] = rest.strip()
    elif rest and arg_style == "agent":
        # /agent <name> "optional prompt"
        try:
            tokens = shlex.split(rest)
        except ValueError:
            tokens = rest.split(None, 1)
        if tokens:
            args["agent_name"] = tokens[0]
        if len(tokens) > 1:
            args["prompt"] = tokens[1]
    elif rest and arg_style == "project":
        try:
            tokens = shlex.split(rest)
        except ValueError:
            tokens = rest.split()
        if tokens and tokens[0] in _PROJECT_VERBS:
            # Verb path. For `add`, assign positionally — each only if present.
            # Missing fields are NOT an error here; app.py renders a usage hint.
            args["verb"] = tokens[0]
            if len(tokens) > 1:
                args["id"] = tokens[1]
            if len(tokens) > 2:
                args["root_path"] = tokens[2]
            if len(tokens) > 3:
                args["description"] = " ".join(tokens[3:])
        else:
            # Switch — preserve byte-for-byte the whole stripped rest (may contain spaces).
            args["id"] = rest.strip()
    # None or unrecognized: no args

    return SlashCommand(name=name, args=args, is_local=is_local)


def list_commands() -> list[str]:
    return sorted(_COMMANDS.keys())


def list_commands_with_descriptions() -> list[tuple[str, str]]:
    return sorted((name, spec["description"]) for name, spec in _COMMANDS.items())
