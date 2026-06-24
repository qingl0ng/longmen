"""Project type detection — scan build files to determine project type and commands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from .base import BaseTool

log = structlog.get_logger(__name__)

# Module-level cache: resolved_root_path -> ProjectType
_project_type_cache: dict[str, ProjectType] = {}


@dataclass
class ProjectType:
    types: list[str] = field(default_factory=list)
    build_cmd: str | None = None
    test_cmd: str | None = None
    test_framework: str | None = None
    package_manager: str | None = None
    detected_files: list[str] = field(default_factory=list)


def get_cached_project_type(root_path: str) -> ProjectType | None:
    """Return cached project type for root_path, or None if not yet detected."""
    return _project_type_cache.get(str(Path(root_path).expanduser().resolve()))


def _detect_catch2(root: Path) -> bool:
    """Scan CMakeLists.txt files (root + subdirs up to 2 levels) for Catch2 usage."""
    patterns = [
        re.compile(r"FetchContent_Declare\s*\(\s*[Cc]atch2", re.IGNORECASE),
        re.compile(r"find_package\s*\(\s*Catch2", re.IGNORECASE),
        re.compile(r"include\s*\(\s*Catch\b", re.IGNORECASE),
        re.compile(r"include\s*\(\s*CatchAddTests\b", re.IGNORECASE),
        re.compile(r"catch_discover_tests\s*\(", re.IGNORECASE),
        re.compile(r"catch2_discover_tests\s*\(", re.IGNORECASE),
    ]

    cmake_files: list[Path] = [root / "CMakeLists.txt"]
    try:
        for subdir in root.iterdir():
            if subdir.is_dir():
                cmake_files.append(subdir / "CMakeLists.txt")
                try:
                    for subsubdir in subdir.iterdir():
                        if subsubdir.is_dir():
                            cmake_files.append(subsubdir / "CMakeLists.txt")
                except PermissionError:
                    pass
    except PermissionError:
        pass

    for cmake_file in cmake_files:
        if not cmake_file.exists():
            continue
        try:
            content = cmake_file.read_text(errors="replace")
            for pattern in patterns:
                if pattern.search(content):
                    return True
        except Exception:
            pass
    return False


def detect_project_type(root_path: str) -> ProjectType:
    """Scan root_path for build files and return a ProjectType (also caches result)."""
    root = Path(root_path).expanduser().resolve()
    pt = ProjectType()

    # ── Python ───────────────────────────────────────────────────────────────
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            if data.get("tool", {}).get("poetry") is not None:
                pt.types.append("python-poetry")
                pt.detected_files.append("pyproject.toml")
                pt.build_cmd = "poetry install"
                pt.test_cmd = "poetry run pytest"
                pt.test_framework = "pytest"
                pt.package_manager = "poetry"
            elif "build-system" in data:
                pt.types.append("python-pip")
                pt.detected_files.append("pyproject.toml")
                pt.build_cmd = "pip install -e ."
                pt.test_cmd = "pytest"
                pt.test_framework = "pytest"
                pt.package_manager = "pip"
        except Exception as exc:
            log.warning("project_detect.pyproject_parse_error", error=str(exc))

    if not any(t.startswith("python") for t in pt.types):
        if (root / "setup.py").exists():
            pt.types.append("python-pip")
            pt.detected_files.append("setup.py")
            pt.build_cmd = "pip install -e ."
            pt.test_cmd = "pytest"
            pt.test_framework = "pytest"
            pt.package_manager = "pip"
        elif (root / "setup.cfg").exists():
            pt.types.append("python-pip")
            pt.detected_files.append("setup.cfg")
            pt.build_cmd = "pip install -e ."
            pt.test_cmd = "pytest"
            pt.test_framework = "pytest"
            pt.package_manager = "pip"
        elif (root / "requirements.txt").exists():
            pt.types.append("python-pip")
            pt.detected_files.append("requirements.txt")
            pt.build_cmd = "pip install -r requirements.txt"
            pt.test_cmd = "pytest"
            pt.test_framework = "pytest"
            pt.package_manager = "pip"

    # ── CMake ─────────────────────────────────────────────────────────────────
    cmake_file = root / "CMakeLists.txt"
    if cmake_file.exists():
        pt.detected_files.append("CMakeLists.txt")
        if _detect_catch2(root):
            pt.types.append("cmake-catch2")
            pt.test_framework = "catch2"
            pt.test_cmd = "ctest --test-dir build --output-on-failure"
        else:
            pt.types.append("cmake")
            pt.test_framework = pt.test_framework or "ctest"
            pt.test_cmd = pt.test_cmd or "ctest --test-dir build"
        pt.build_cmd = pt.build_cmd or "cmake -B build && cmake --build build"

    # ── Makefile (only if no CMake) ───────────────────────────────────────────
    if (root / "Makefile").exists() and not cmake_file.exists():
        pt.types.append("make")
        pt.detected_files.append("Makefile")
        pt.build_cmd = pt.build_cmd or "make"
        pt.test_cmd = pt.test_cmd or "make test"

    # ── Node.js / pnpm / yarn / npm ───────────────────────────────────────────
    package_json = root / "package.json"
    pnpm_lock = root / "pnpm-lock.yaml"
    yarn_lock = root / "yarn.lock"

    if pnpm_lock.exists():
        pt.types.append("pnpm")
        pt.detected_files.append("pnpm-lock.yaml")
        pt.package_manager = pt.package_manager or "pnpm"
        pt.build_cmd = pt.build_cmd or "pnpm install"
        pt.test_cmd = pt.test_cmd or "pnpm test"
    elif yarn_lock.exists():
        pt.types.append("yarn")
        pt.detected_files.append("yarn.lock")
        pt.package_manager = pt.package_manager or "yarn"
        pt.build_cmd = pt.build_cmd or "yarn install"
        pt.test_cmd = pt.test_cmd or "yarn test"
    elif package_json.exists():
        try:
            pkg = json.loads(package_json.read_text())
            dev_deps = pkg.get("devDependencies", {})
            all_deps = {**pkg.get("dependencies", {}), **dev_deps}
            if "jest" in dev_deps or "jest" in all_deps:
                pt.types.append("npm-jest")
                pt.test_framework = pt.test_framework or "jest"
                pt.test_cmd = pt.test_cmd or "npx jest"
            elif "vitest" in dev_deps or "vitest" in all_deps:
                pt.types.append("npm-vitest")
                pt.test_framework = pt.test_framework or "vitest"
                pt.test_cmd = pt.test_cmd or "npx vitest run"
            elif "mocha" in dev_deps or "mocha" in all_deps:
                pt.types.append("npm-mocha")
                pt.test_framework = pt.test_framework or "mocha"
                pt.test_cmd = pt.test_cmd or "npx mocha"
            else:
                pt.types.append("npm")
                pt.test_cmd = pt.test_cmd or "npm test"
            pt.detected_files.append("package.json")
            pt.package_manager = pt.package_manager or "npm"
            pt.build_cmd = pt.build_cmd or "npm install"
        except Exception:
            pt.types.append("npm")
            pt.detected_files.append("package.json")
            pt.package_manager = pt.package_manager or "npm"
            pt.build_cmd = pt.build_cmd or "npm install"
            pt.test_cmd = pt.test_cmd or "npm test"

    # ── Rust / Cargo ──────────────────────────────────────────────────────────
    if (root / "Cargo.toml").exists():
        pt.types.append("cargo")
        pt.detected_files.append("Cargo.toml")
        pt.build_cmd = pt.build_cmd or "cargo build"
        pt.test_cmd = pt.test_cmd or "cargo test"
        pt.test_framework = pt.test_framework or "cargo-test"
        pt.package_manager = pt.package_manager or "cargo"

    # ── Go ────────────────────────────────────────────────────────────────────
    if (root / "go.mod").exists():
        pt.types.append("go")
        pt.detected_files.append("go.mod")
        pt.build_cmd = pt.build_cmd or "go build ./..."
        pt.test_cmd = pt.test_cmd or "go test ./..."
        pt.test_framework = pt.test_framework or "go-test"

    # ── Docker ────────────────────────────────────────────────────────────────
    if (root / "Dockerfile").exists():
        project_name = root.name.lower().replace(" ", "-")
        pt.types.append("docker")
        pt.detected_files.append("Dockerfile")
        if not pt.build_cmd:
            pt.build_cmd = f"docker build -t {project_name} ."

    for compose_name in (
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
    ):
        if (root / compose_name).exists():
            pt.types.append("docker-compose")
            pt.detected_files.append(compose_name)
            if not pt.build_cmd:
                pt.build_cmd = "docker compose up --build"
            break

    # ── SQL ───────────────────────────────────────────────────────────────────
    migrations_dir = root / "migrations"
    has_sql = bool(list(root.glob("*.sql"))) or (
        migrations_dir.is_dir() and bool(list(migrations_dir.glob("*.sql")))
    )
    if has_sql:
        pt.types.append("sql")

    # Cache and return
    _project_type_cache[str(root)] = pt
    return pt


def build_system_prompt_section(pt: ProjectType) -> str:
    """Return the '## Build & test tools' system prompt section for a detected project."""
    project_types = ", ".join(pt.types) if pt.types else "(unknown)"
    lines = [
        "## Build & test tools",
        "",
        f"Project: {project_types}",
        f"Build: {pt.build_cmd or '(none)'}",
        f"Test: {pt.test_cmd or '(none)'}",
        f"Framework: {pt.test_framework or '(none)'}",
        "",
        "Use these tools instead of shell for build/test work — they return structured results:",
        "- `build` — compile/install; returns errors as file:line:message",
        "- `run_tests` — run tests; returns pass/fail counts and failure details",
        "- `run_app` — run a script or server; captures stdout/stderr",
        "- `sql_query` — query the database; returns a formatted table",
        "",
        "Workflow when fixing code:",
        "1. Edit with write_file or search_replace",
        "2. build → read errors by file:line → fix → repeat until build passes",
        "3. run_tests → read failure messages → fix → repeat until tests pass",
        "",
        "Focused test runs:",
        '- run_tests(path="tests/auth/")',
        '- run_tests(test_name="test_login")',
        '- run_tests(path="tests/auth/test_login.py", test_name="test_expired_token")',
        "",
        "Server startup check:",
        '- run_app(command="flask run", wait_for="Running on http")',
        "",
        "Never use shell to build or run tests.",
    ]
    return "\n".join(lines)


class ProjectDetectTool(BaseTool):
    name = "detect_project"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "detect_project",
                "description": (
                    "Detect the project's build system and test framework by scanning build files "
                    "(pyproject.toml, CMakeLists.txt, package.json, Cargo.toml, go.mod, etc.). "
                    "Returns build_cmd, test_cmd, test_framework, package_manager,"
                    " and detected types. "
                    "Call once per conversation before using `build` or `run_tests`. "
                    "Call again only if build files changed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def execute(self, root_path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            pt = detect_project_type(root_path)
        except Exception as e:
            log.error("project_detect.error", error=str(e))
            return {"error": f"Detection failed: {e}"}

        if not pt.types:
            return {
                "stdout": "No recognized build files found in project root.",
                "stderr": "",
                "exit_code": 0,
                "project_type": None,
            }

        lines = [
            f"Project type: {', '.join(pt.types)}",
            f"Build: {pt.build_cmd or '(none)'}",
            f"Test: {pt.test_cmd or '(none)'}",
            f"Test framework: {pt.test_framework or '(none)'}",
            f"Package manager: {pt.package_manager or '(none)'}",
            f"Detected from: {', '.join(pt.detected_files)}",
        ]
        return {
            "stdout": "\n".join(lines),
            "stderr": "",
            "exit_code": 0,
            "project_type": {
                "types": pt.types,
                "build_cmd": pt.build_cmd,
                "test_cmd": pt.test_cmd,
                "test_framework": pt.test_framework,
                "package_manager": pt.package_manager,
                "detected_files": pt.detected_files,
            },
        }
