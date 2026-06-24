"""Local conversation display cache for scrollback after reconnect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MAX_TURNS = 1000


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    extra: dict[str, Any] = field(default_factory=dict)  # tool outputs, etc.


class History:
    def __init__(self) -> None:
        self._turns: list[Turn] = []

    def add(self, role: str, text: str, extra: dict[str, Any] | None = None) -> None:
        self._turns.append(Turn(role=role, text=text, extra=extra or {}))
        if len(self._turns) > _MAX_TURNS:
            self._turns = self._turns[-_MAX_TURNS:]

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
