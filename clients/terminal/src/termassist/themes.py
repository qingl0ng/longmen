"""Color theme definitions."""

from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "default": {
        "user": "bold green",
        "assistant": "bold blue",
        "thinking": "dim italic",
        "tool_call": "dim cyan",
        "tool_output_title": "bold white",
        "approval_safe": "bold green",
        "approval_moderate": "bold yellow",
        "approval_destructive": "bold red",
        "error": "bold red",
        "warning": "bold yellow",
        "info": "dim",
        "token_bar_low": "green",
        "token_bar_mid": "yellow",
        "token_bar_high": "red",
        "spinner": "cyan",
        "status": "dim",
    },
    "light": {
        "user": "bold dark_green",
        "assistant": "bold dark_blue",
        "thinking": "dim italic",
        "tool_call": "dim dark_cyan",
        "tool_output_title": "bold black",
        "approval_safe": "bold dark_green",
        "approval_moderate": "bold dark_orange",
        "approval_destructive": "bold dark_red",
        "error": "bold dark_red",
        "warning": "bold dark_orange",
        "info": "dim",
        "token_bar_low": "dark_green",
        "token_bar_mid": "dark_orange",
        "token_bar_high": "dark_red",
        "spinner": "dark_cyan",
        "status": "dim",
    },
    "minimal": {
        "user": "",
        "assistant": "",
        "thinking": "dim",
        "tool_call": "",
        "tool_output_title": "",
        "approval_safe": "",
        "approval_moderate": "",
        "approval_destructive": "",
        "error": "",
        "warning": "",
        "info": "",
        "token_bar_low": "",
        "token_bar_mid": "",
        "token_bar_high": "",
        "spinner": "",
        "status": "",
    },
}


def get_theme(name: str) -> dict[str, str]:
    return THEMES.get(name, THEMES["default"])
