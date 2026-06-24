"""Git tools — git_status, git_diff, git_log, git_add, git_commit."""

from __future__ import annotations

import subprocess
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)


def _run_git(args: list[str], cwd: str) -> tuple[str, str, int]:
    """Run a git command; return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "git command timed out", 1
    except FileNotFoundError:
        return "", "git not found in PATH", 1


def _is_git_repo(root_path: str) -> bool:
    _, _, rc = _run_git(["rev-parse", "--git-dir"], root_path)
    return rc == 0


class GitStatusTool(BaseTool):
    name = "git_status"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": (
                    "Show the current git status of the project: which files are staged, "
                    "modified but unstaged, and untracked. Returns 'git status --short' "
                    "output plus the current branch name. "
                    "Errors: 'not a git repository' if the project root is not a git repo."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def execute(self, root_path: str, **kwargs: Any) -> dict[str, Any]:
        if not _is_git_repo(root_path):
            return {"error": "not a git repository"}

        branch_out, _, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root_path)
        branch = branch_out.strip() or "unknown"

        status_out, status_err, rc = _run_git(["status", "--short"], root_path)
        if rc != 0:
            return {"error": status_err.strip() or "git status failed"}

        if status_out.strip():
            output = f"On branch {branch}\n{status_out.rstrip()}"
        else:
            output = f"On branch {branch}\nnothing to commit, working tree clean"
        return {"stdout": output}


class GitDiffTool(BaseTool):
    name = "git_diff"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": (
                    "Show a unified diff of changes in the project. By default shows unstaged "
                    "changes (working tree vs HEAD). Pass 'ref' to compare against a specific "
                    "commit or branch. Pass 'path' to limit the diff to one file. "
                    "Errors: 'not a git repository', 'unknown ref: <ref>'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Optional. Limit the diff to this file only."
                                " Path relative to project root."
                            ),
                        },
                        "ref": {
                            "type": "string",
                            "description": (
                                "Optional. Compare working tree against this git ref"
                                " instead of HEAD."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        root_path: str,
        path: str | None = None,
        ref: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not _is_git_repo(root_path):
            return {"error": "not a git repository"}

        args = ["diff"]
        if ref:
            _, _, rc = _run_git(["rev-parse", "--verify", ref], root_path)
            if rc != 0:
                return {"error": f"unknown ref: {ref}"}
            args.append(ref)

        if path:
            try:
                safe = self._safe_path(root_path, path)
                args += ["--", safe]
            except ValueError as e:
                return {"error": str(e)}

        out, err, rc = _run_git(args, root_path)
        if rc != 0:
            return {"error": err.strip() or "git diff failed"}

        return {"stdout": out if out else "(no differences)"}


class GitLogTool(BaseTool):
    name = "git_log"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": (
                    "Show recent commit history for the project or a specific file. "
                    "Returns one line per commit: short hash, author, date, and message. "
                    "Errors: 'not a git repository'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of most-recent commits to show. Default: 10.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional. Only show commits that touched this file.",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        root_path: str,
        count: int = 10,
        path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not _is_git_repo(root_path):
            return {"error": "not a git repository"}

        count = min(max(1, int(count)), 100)
        args = [
            "log",
            f"-{count}",
            "--format=%h  %an  %ad  %s",
            "--date=short",
        ]

        if path:
            try:
                safe = self._safe_path(root_path, path)
                args += ["--", safe]
            except ValueError as e:
                return {"error": str(e)}

        out, err, rc = _run_git(args, root_path)
        if rc != 0:
            return {"error": err.strip() or "git log failed"}

        return {"stdout": out.rstrip() if out.strip() else "(no commits)"}


class GitAddTool(BaseTool):
    name = "git_add"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_add",
                "description": (
                    "Stage one or more files to be included in the next git_commit call. "
                    "Must be called before git_commit. Use paths=['.'] to stage all changes. "
                    "Errors: 'path not found', 'path outside project root', 'not a git repository'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of file paths to stage, each relative to the project root. "
                                "Must not be empty. Use ['.'] to stage all changes."
                            ),
                        },
                    },
                    "required": ["paths"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        paths: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not _is_git_repo(root_path):
            return {"error": "not a git repository"}
        if not paths:
            return {"error": "paths must not be empty"}

        resolved: list[str] = []
        for p in paths:
            if p == ".":
                resolved.append(".")
            else:
                try:
                    resolved.append(self._safe_path(root_path, p))
                except ValueError as e:
                    return {"error": str(e)}

        _, err, rc = _run_git(["add", *resolved], root_path)
        if rc != 0:
            return {"error": err.strip() or "git add failed"}

        status_out, _, _ = _run_git(["status", "--short"], root_path)
        staged = [
            line[3:].strip()
            for line in status_out.splitlines()
            if line and line[0] in ("A", "M", "D", "R")
        ]
        summary = ", ".join(staged) if staged else "(nothing new staged)"
        return {"stdout": f"Staged: {summary}"}


class GitCommitTool(BaseTool):
    name = "git_commit"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": (
                    "Record all staged changes as a new commit. Precondition: at least one "
                    "file must be staged with git_add first. "
                    "Errors: 'nothing to commit — stage files with git_add first', "
                    "'not a git repository'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "Commit message. Use imperative mood, present tense. "
                                "Keep under 72 characters for the first line."
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
        }

    async def execute(  # type: ignore[override]
        self,
        root_path: str,
        message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not _is_git_repo(root_path):
            return {"error": "not a git repository"}
        if not message.strip():
            return {"error": "commit message must not be empty"}

        status_out, _, _ = _run_git(["status", "--short"], root_path)
        staged = [
            line for line in status_out.splitlines() if line and line[0] in ("A", "M", "D", "R")
        ]
        if not staged:
            return {"error": "nothing to commit — stage files with git_add first"}

        out, err, rc = _run_git(["commit", "-m", message], root_path)
        if rc != 0:
            return {"error": err.strip() or out.strip() or "git commit failed"}

        hash_out, _, _ = _run_git(["rev-parse", "--short", "HEAD"], root_path)
        commit_hash = hash_out.strip()
        return {"stdout": f"[{commit_hash}] {message}"}
