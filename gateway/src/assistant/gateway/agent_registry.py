"""Agent registry per project — CRUD agents, validate backends."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import structlog
import tomli_w

log = structlog.get_logger(__name__)


class AgentRegistry:
    def __init__(
        self,
        data_dir: str,
        known_backends: set[str],
        known_tools: set[str],
    ) -> None:
        self._data_dir = Path(data_dir)
        self._known_backends = known_backends
        self._known_tools = known_tools

    def _project_dir(self, project_id: str) -> Path:
        return self._data_dir / "projects" / project_id

    def _agents_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "agents.toml"

    def _prompt_path(self, project_id: str, name: str) -> Path:
        return self._project_dir(project_id) / "prompts" / f"{name}.md"

    def list_agents(self, project_id: str) -> dict[str, dict[str, Any]]:
        agents_path = self._agents_path(project_id)
        if not agents_path.exists():
            return {}
        try:
            with open(agents_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            log.error("agent_registry.load_error", project_id=project_id, error=str(e))
            return {}

    def upsert(
        self,
        project_id: str,
        name: str,
        agent: dict[str, Any],
    ) -> list[str]:
        """Returns list of validation errors. Writes agents.toml and prompts/{name}.md."""
        errors: list[str] = []

        # Validate name
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            errors.append(f"invalid_name: '{name}' must be alphanumeric with dashes/underscores")

        # Validate backend if specified
        backend = agent.get("backend")
        if backend and backend not in self._known_backends and backend != "local":
            errors.append(f"backend_not_found: '{backend}'")

        # Validate tools
        tools = agent.get("tools", [])
        for tool in tools:
            if tool not in self._known_tools:
                errors.append(f"invalid_tool: '{tool}'")

        # Validate system_prompt
        system_prompt = agent.get("system_prompt", "")
        if not system_prompt.strip():
            errors.append("prompt_empty: system_prompt cannot be empty")

        if errors:
            return errors

        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir = project_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        # Write prompt file
        prompt_file = f"prompts/{name}.md"
        self._prompt_path(project_id, name).write_text(system_prompt)

        # Load existing agents and update
        agents = self.list_agents(project_id)
        agent_data = {k: v for k, v in agent.items() if k != "system_prompt"}
        agent_data["prompt_file"] = prompt_file
        agents[name] = agent_data

        with open(self._agents_path(project_id), "wb") as f:
            tomli_w.dump(agents, f)

        log.info("agent_registry.upserted", project_id=project_id, name=name)
        return []

    def delete(self, project_id: str, name: str) -> None:
        agents = self.list_agents(project_id)
        if name not in agents:
            raise KeyError(f"Agent not found: {name}")
        del agents[name]

        with open(self._agents_path(project_id), "wb") as f:
            tomli_w.dump(agents, f)

        prompt_path = self._prompt_path(project_id, name)
        if prompt_path.exists():
            prompt_path.unlink()

        log.info("agent_registry.deleted", project_id=project_id, name=name)

    def get_prompt(self, project_id: str, name: str) -> str | None:
        prompt_path = self._prompt_path(project_id, name)
        if not prompt_path.exists():
            return None
        return prompt_path.read_text()
