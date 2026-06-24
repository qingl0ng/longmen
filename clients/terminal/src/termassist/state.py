"""Client-side state machine tracking what the gateway is doing."""

from __future__ import annotations

import asyncio


class InvalidTransitionError(Exception):
    pass


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: dict[str, set[str]] = {
    "idle": {"streaming", "queued", "waiting_for_model", "disconnected"},
    "streaming": {"idle", "awaiting_approval", "disconnected"},
    "awaiting_approval": {"executing_tool", "idle", "disconnected"},
    "executing_tool": {"streaming", "idle", "disconnected"},
    "queued": {"streaming", "idle", "waiting_for_model", "disconnected"},
    "waiting_for_model": {"streaming", "queued", "idle", "disconnected"},
    "disconnected": {"idle"},  # after reconnect
}


class StateMachine:
    def __init__(self) -> None:
        self.current: str = "idle"
        self.session_id: str | None = None
        self.last_session_id: str | None = None
        self._events: dict[str, asyncio.Event] = {
            state: asyncio.Event() for state in _TRANSITIONS
        }
        self._events["idle"].set()

    def transition(self, new_state: str) -> None:
        allowed = _TRANSITIONS.get(self.current, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self.current!r} to {new_state!r}"
            )
        old_state = self.current
        self.current = new_state

        # Reset old state event, set new state event
        self._events[old_state].clear()
        if new_state in self._events:
            self._events[new_state].set()

    def go(self, new_state: str) -> None:
        """Transition, falling back to force on InvalidTransition (for normal message dispatch)."""
        try:
            self.transition(new_state)
        except InvalidTransitionError:
            self.force_transition(new_state)

    def force_transition(self, new_state: str) -> None:
        """Transition without checking validity (for error recovery)."""
        old_state = self.current
        self.current = new_state
        self._events[old_state].clear()
        if new_state in self._events:
            self._events[new_state].set()

    async def wait_for(self, state: str) -> None:
        """Block until the state machine enters the given state."""
        if self.current == state:
            return
        event = self._events.get(state)
        if event is None:
            raise ValueError(f"Unknown state: {state!r}")
        await event.wait()

    async def wait_for_any(self, states: set[str]) -> None:
        """Block until the machine is in any of the given states."""
        if self.current in states:
            return
        tasks = [
            asyncio.create_task(self._events[s].wait())
            for s in states
            if s in self._events
        ]
        if not tasks:
            return
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
