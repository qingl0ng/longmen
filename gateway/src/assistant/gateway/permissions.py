"""Risk classification + approval flow + stored permissions."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from .protocol import make_approval_request

if TYPE_CHECKING:
    from pathlib import Path

    from .config import PermissionsConfig

log = structlog.get_logger(__name__)


class PermissionManager:
    def __init__(
        self,
        config: PermissionsConfig,
        project_data_dir: Path,
        approval_timeout: int = 300,
    ) -> None:
        self._config = config
        self._project_data_dir = project_data_dir
        self._approval_timeout = approval_timeout
        # Pending approvals: approval_id → asyncio.Future[str]
        self._pending: dict[str, asyncio.Future[str]] = {}
        # Stored "yes_always" rules: command_pattern → "allow"
        self._stored_rules: dict[str, str] = {}
        self._load_permissions()

    def _permissions_path(self) -> Path:
        return self._project_data_dir / "permissions.toml"

    def _load_permissions(self) -> None:
        import tomllib

        path = self._permissions_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            self._stored_rules = data.get("rules", {})
        except Exception as e:
            log.error("permissions.load_error", error=str(e))

    def _save_permissions(self) -> None:
        import tomli_w

        path = self._permissions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump({"rules": self._stored_rules}, f)

    def stored_rules(self) -> dict[str, str]:
        """Return a copy of stored yes_always rules."""
        return dict(self._stored_rules)

    def classify_risk(self, tool_name: str, command: str) -> str:
        """Returns 'safe' | 'moderate' | 'destructive'."""
        if tool_name in self._config.default_safe:
            return "safe"
        if tool_name in self._config.default_destructive:
            return "destructive"
        # Check command text for destructive keywords
        cmd_lower = command.lower()
        for destructive in self._config.default_destructive:
            if destructive in cmd_lower:
                return "destructive"
        return "moderate"

    def _is_stored_allowed(self, command: str) -> bool:
        """Check if command matches any stored 'yes_always' rule."""
        for pattern, decision in self._stored_rules.items():
            if decision == "allow" and self._matches(pattern, command):
                return True
        return False

    def _matches(self, pattern: str, command: str) -> bool:
        """Simple glob-like matching."""
        import fnmatch

        return fnmatch.fnmatch(command, pattern) or pattern in command

    async def check(
        self,
        session_id: str,
        tool_name: str,
        command: str,
        ws: Any,
    ) -> bool:
        """Returns True if approved. Sends approval_request, waits for approval_response."""
        # In workflow_mode=allow_all, auto-approve everything
        if self._config.workflow_mode == "allow_all":
            return True

        # Check stored permissions
        if self._is_stored_allowed(command):
            log.info("permissions.auto_approved", tool=tool_name, command=command)
            return True

        risk = self.classify_risk(tool_name, command)

        # Safe tools in non-allow_all mode: still approve automatically
        if risk == "safe":
            return True

        # Need user approval
        approval_id = str(uuid.uuid4())
        context = f"Running {tool_name}: {command}"

        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending[approval_id] = future

        request_msg = make_approval_request(
            session_id=session_id,
            approval_id=approval_id,
            tool=tool_name,
            command=command,
            risk=risk,
            context=context,
            timeout_seconds=self._approval_timeout,
        )
        await ws.send(json.dumps(request_msg))

        try:
            decision = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._approval_timeout,
            )
        except TimeoutError:
            log.warning("permissions.approval_timeout", approval_id=approval_id)
            self._pending.pop(approval_id, None)
            return False
        finally:
            self._pending.pop(approval_id, None)

        if decision == "yes_always":
            self._stored_rules[command] = "allow"
            self._save_permissions()

        return decision in ("yes", "yes_session", "yes_always", "edit")

    def resolve_approval(self, approval_id: str, decision: str) -> None:
        """Called when an approval_response arrives from the client."""
        future = self._pending.get(approval_id)
        if future and not future.done():
            future.set_result(decision)

    def reload(self, config: PermissionsConfig) -> None:
        self._config = config
        self._load_permissions()
