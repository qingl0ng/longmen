"""Tool registry."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .app_runner import AppRunnerTool
from .build import BuildTool
from .delete import DeleteTool

if TYPE_CHECKING:
    from .base import BaseTool
from .file_read import GrepTool, ListDirTool, ReadFileTool, SymbolsTool, TreeTool
from .file_write import SearchReplaceTool, WriteFileTool
from .git import GitAddTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from .project_detect import ProjectDetectTool
from .shell import ShellTool
from .sql import SQLQueryTool
from .test_runner import TestRunnerTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool

# Build base TOOL_REGISTRY (tools that are always available)
_BASE_TOOL_REGISTRY: dict[str, BaseTool] = {
    "shell": ShellTool(),
    "read_file": ReadFileTool(),
    "list_dir": ListDirTool(),
    "grep": GrepTool(),
    "write_file": WriteFileTool(),
    "search_replace": SearchReplaceTool(),
    "tree": TreeTool(),
    "symbols": SymbolsTool(),
    "git_status": GitStatusTool(),
    "git_diff": GitDiffTool(),
    "git_log": GitLogTool(),
    "git_add": GitAddTool(),
    "git_commit": GitCommitTool(),
    "detect_project": ProjectDetectTool(),
    "build": BuildTool(),
    "run_tests": TestRunnerTool(),
    "run_app": AppRunnerTool(),
    "sql_query": SQLQueryTool(),
    "delete_tool": DeleteTool(),
}

# Conditional tools (web_search)
_TOOL_REGISTRY: dict[str, BaseTool] = dict(_BASE_TOOL_REGISTRY)

# web_search tool will be added during server startup via update_tool_registry()
# which handles API key priority (env var > config file)
KNOWN_TOOLS: set[str] = set(_TOOL_REGISTRY.keys())

# Export TOOL_REGISTRY for backward compatibility
TOOL_REGISTRY: dict[str, BaseTool] = _TOOL_REGISTRY


def update_tool_registry(
    web_config_brave_key: str = "",
    search_enabled: bool = True,
    search_count: int = 5,
    fetch_enabled: bool = True,
    fetch_timeout: int = 15,
    fetch_max_redirects: int = 5,
    fetch_blocked_domains: list[str] | None = None,
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    rag_client: Any = None,
) -> None:
    """Update TOOL_REGISTRY based on web and RAG configuration.

    Call this when config is reloaded to add/remove web_search, web_fetch, and rag_search.
    Environment variable BRAVE_API_KEY always takes precedence over config file value.
    """
    global _TOOL_REGISTRY, KNOWN_TOOLS, TOOL_REGISTRY
    _TOOL_REGISTRY = dict(_BASE_TOOL_REGISTRY)

    # Priority: environment variable > config file
    brave_key = os.environ.get("BRAVE_API_KEY", "") or web_config_brave_key

    # Add web_search only when enabled and a valid API key is present
    if search_enabled and brave_key:
        _TOOL_REGISTRY["web_search"] = WebSearchTool(
            search_count=search_count,
            brave_api_key=brave_key,
            user_agent=user_agent,
        )

    # Add web_fetch only when enabled
    if fetch_enabled:
        _TOOL_REGISTRY["web_fetch"] = WebFetchTool(
            fetch_timeout=fetch_timeout,
            fetch_max_redirects=fetch_max_redirects,
            user_agent=user_agent,
            fetch_blocked_domains=fetch_blocked_domains or [],
        )

    # Add rag_search when a RAGClient is provided
    if rag_client is not None:
        from .rag_search import RAGSearchTool
        _TOOL_REGISTRY["rag_search"] = RAGSearchTool()

    KNOWN_TOOLS = set(_TOOL_REGISTRY.keys())
    TOOL_REGISTRY = _TOOL_REGISTRY


def get_tool_registry() -> dict[str, BaseTool]:
    """Return the current TOOL_REGISTRY."""
    return _TOOL_REGISTRY


def get_known_tools() -> set[str]:
    """Return the current KNOWN_TOOLS set."""
    return KNOWN_TOOLS


def get_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    """Return OpenAI-format function schemas for the requested tools."""
    return [TOOL_REGISTRY[n].schema() for n in tool_names if n in TOOL_REGISTRY]


async def execute_tool(name: str, root_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}
    return await tool.execute(root_path, **arguments)
