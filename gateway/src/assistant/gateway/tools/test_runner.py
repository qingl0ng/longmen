"""Test runner tool — runs tests and parses structured results."""

from __future__ import annotations

import contextlib
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import structlog

from .base import BaseTool
from .project_detect import get_cached_project_type
from .shell import ShellTool

log = structlog.get_logger(__name__)

_shell = ShellTool()


# ── Result types ──────────────────────────────────────────────────────────────


def _make_result(
    passed: int,
    failed: int,
    skipped: int,
    duration_s: float | None,
    failures: list[dict[str, Any]],
    passed_names: list[str] | None = None,
) -> dict[str, Any]:
    total = passed + failed + skipped
    parts: list[str] = []
    dur = f" in {duration_s:.1f}s" if duration_s is not None else ""
    parts.append(f"Tests: {passed} passed, {failed} failed, {skipped} skipped ({total} total){dur}")

    if failures:
        parts.append("\nFAILED tests:")
        for f_ in failures:
            name = f_.get("name", "unknown")
            section = f_.get("section", "")
            file_ = f_.get("file", "")
            line = f_.get("line", 0)
            msg = f_.get("message", "")
            expr = f_.get("expression", "")
            expanded = f_.get("expanded", "")

            parts.append(f"  {name}")
            if section:
                parts.append(f"    Section: {section}")
            if file_ and line:
                parts.append(f"    {file_}:{line}")
            elif file_:
                parts.append(f"    {file_}")
            if msg:
                parts.append(f"    {msg}")
            if expr:
                parts.append(f"    {expr}")
            if expanded:
                parts.append(f"    with expansion: {expanded}")

    if passed_names is not None:
        parts.append(f"\nPassed: {len(passed_names)} tests")
        for n in passed_names:
            parts.append(f"  {n}")
    elif passed > 0:
        parts.append(f"\nPassed: {passed} tests (not shown — use verbose=true to see all)")

    return {
        "stdout": "\n".join(parts),
        "stderr": "",
        "exit_code": 0 if failed == 0 else 1,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failures": failures,
    }


# ── Framework parsers ─────────────────────────────────────────────────────────


def _parse_pytest(output: str, verbose: bool = False) -> dict[str, Any]:
    """Parse pytest --tb=short -q output."""
    passed = failed = skipped = 0
    duration_s: float | None = None
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    # Summary line: "X passed, Y failed, Z skipped in N.NNs"
    summary_re = re.compile(
        r"(?:(\d+) passed)?(?:,\s*)?(?:(\d+) failed)?(?:,\s*)?(?:(\d+) (?:skipped|warning))?"
        r".*?in ([\d.]+)s",
        re.IGNORECASE,
    )
    for m in summary_re.finditer(output):
        p_val = m.group(1)
        f_val = m.group(2)
        s_val = m.group(3)
        d_val = m.group(4)
        if p_val or f_val:
            passed = int(p_val or 0)
            failed = int(f_val or 0)
            skipped = int(s_val or 0)
            duration_s = float(d_val) if d_val else None

    # Parse PASSED/FAILED lines: "tests/file.py::test_name PASSED"
    test_re = re.compile(r"^(\S+::[\w\[\]-]+)\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)
    current_failures: dict[str, dict[str, Any]] = {}
    for m in test_re.finditer(output):
        name = m.group(1)
        status = m.group(2)
        if status in ("FAILED", "ERROR"):
            current_failures[name] = {"name": name, "message": "", "file": "", "line": 0}
        elif status == "PASSED" and verbose and passed_names is not None:
            passed_names.append(name)

    # Parse short tracebacks for FAILED tests
    # Pattern: "FAILED tests/file.py::test_name - ErrorType: message"
    fail_detail_re = re.compile(r"^FAILED\s+(\S+::[\w\[\]-]+)\s+-\s+(.+)$", re.MULTILINE)
    for m in fail_detail_re.finditer(output):
        name = m.group(1)
        msg = m.group(2).strip()
        if name in current_failures:
            current_failures[name]["message"] = msg
        else:
            current_failures[name] = {"name": name, "message": msg, "file": "", "line": 0}

    # Try to extract file:line from traceback blocks
    # Pattern after "FAILED": file.py:42: AssertionError
    tb_loc_re = re.compile(r"(\S+\.py):(\d+):\s*(\w+Error[^\n]*)", re.MULTILINE)
    for _name, data in current_failures.items():
        if not data.get("file"):
            for m in tb_loc_re.finditer(output):
                if not data.get("file"):
                    data["file"] = m.group(1)
                    data["line"] = int(m.group(2))
                    if not data.get("message"):
                        data["message"] = m.group(3)

    failures = list(current_failures.values())

    # Fallback counts from summary if test lines weren't found
    if not any([passed, failed, skipped]):
        # Try alternate format: "X failed" or "X passed"
        m2 = re.search(r"(\d+) failed", output)
        if m2:
            failed = int(m2.group(1))
        m3 = re.search(r"(\d+) passed", output)
        if m3:
            passed = int(m3.group(1))

    return _make_result(passed, failed, skipped, duration_s, failures, passed_names)


def _parse_jest_json(raw_json: str, verbose: bool = False) -> dict[str, Any]:
    """Parse jest --json output."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return {"stdout": "Failed to parse jest JSON output", "stderr": "", "exit_code": 1}

    passed = data.get("numPassedTests", 0)
    failed = data.get("numFailedTests", 0)
    skipped = data.get("numPendingTests", 0)
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    for suite in data.get("testResults", []):
        file_path = suite.get("testFilePath", "")
        for test in suite.get("testResults", []):
            status = test.get("status", "")
            name = test.get("fullName", test.get("title", "unknown"))
            if status == "failed":
                msgs = test.get("failureMessages", [])
                msg = msgs[0].splitlines()[0] if msgs else ""
                failures.append({"name": name, "file": file_path, "line": 0, "message": msg})
            elif status == "passed" and verbose and passed_names is not None:
                passed_names.append(name)

    return _make_result(passed, failed, skipped, None, failures, passed_names)


def _parse_vitest_json(raw_json: str, verbose: bool = False) -> dict[str, Any]:
    """Parse vitest --reporter=json output."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return {"stdout": "Failed to parse vitest JSON output", "stderr": "", "exit_code": 1}

    passed = failed = skipped = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    for file_result in data.get("testResults", []):
        for test in file_result.get("assertionResults", []):
            status = test.get("status", "")
            name = " > ".join(test.get("ancestorTitles", []) + [test.get("title", "unknown")])
            if status == "passed":
                passed += 1
                if verbose and passed_names is not None:
                    passed_names.append(name)
            elif status == "failed":
                failed += 1
                msgs = test.get("failureMessages", [])
                msg = msgs[0].splitlines()[0] if msgs else ""
                failures.append(
                    {
                        "name": name,
                        "file": file_result.get("name", ""),
                        "line": 0,
                        "message": msg,
                    }
                )
            else:
                skipped += 1

    return _make_result(passed, failed, skipped, None, failures, passed_names)


def _parse_ctest(output: str, verbose: bool = False) -> dict[str, Any]:
    """Parse ctest --output-on-failure output."""
    passed = failed = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    # Pattern: " 1/3  Test  #1: TestAuth .......................... Passed    0.10 sec"
    test_re = re.compile(
        r"^\s*(?:\d+/\d+\s+)?Test\s+#\d+:\s+(\S+)\s+\.+\s+(Passed|Failed)\s+([\d.]+)\s+sec",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in test_re.finditer(output):
        name = m.group(1)
        status = m.group(2).lower()
        if status == "passed":
            passed += 1
            if verbose and passed_names is not None:
                passed_names.append(name)
        else:
            failed += 1
            failures.append({"name": name, "file": "", "line": 0, "message": ""})

    # Also try simpler "N - name: Passed/Failed" format
    if not (passed or failed):
        simple_re = re.compile(
            r"^\s*(\d+)\s*-\s*(\S+)\s*:\s*(Passed|Failed)", re.MULTILINE | re.IGNORECASE
        )
        for m in simple_re.finditer(output):
            name = m.group(2)
            status = m.group(3).lower()
            if status == "passed":
                passed += 1
            else:
                failed += 1
                failures.append({"name": name, "file": "", "line": 0, "message": ""})

    return _make_result(passed, failed, 0, None, failures, passed_names)


def _parse_catch2_xml(xml_content: str, verbose: bool = False) -> dict[str, Any]:
    """Parse Catch2 XML (--reporter xml) output."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return {"stdout": f"Failed to parse Catch2 XML: {e}", "stderr": "", "exit_code": 1}

    passed = failed = skipped = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None
    duration_s: float | None = None

    for tc in root.iter("TestCase"):
        tc_name = tc.get("name", "unknown")
        tc_failed = False

        # Sections with failures
        for section in tc.iter("Section"):
            sec_name = section.get("name", "")
            for expr in section.findall(".//Expression[@success='false']"):
                tc_failed = True
                filename = expr.get("filename", "")
                line = int(expr.get("line", 0) or 0)
                original = (expr.findtext("Original") or "").strip()
                expanded = (expr.findtext("Expanded") or "").strip()
                failures.append(
                    {
                        "name": tc_name,
                        "section": sec_name,
                        "file": filename,
                        "line": line,
                        "expression": original,
                        "expanded": expanded,
                        "message": "",
                    }
                )

        # Direct expressions (outside sections)
        for expr in tc.findall("Expression[@success='false']"):
            tc_failed = True
            filename = expr.get("filename", "")
            line = int(expr.get("line", 0) or 0)
            original = (expr.findtext("Original") or "").strip()
            expanded = (expr.findtext("Expanded") or "").strip()
            failures.append(
                {
                    "name": tc_name,
                    "section": "",
                    "file": filename,
                    "line": line,
                    "expression": original,
                    "expanded": expanded,
                    "message": "",
                }
            )

        if not tc_failed:
            passed += 1
            if verbose and passed_names is not None:
                passed_names.append(tc_name)
        else:
            failed += 1

    # Try to get overall duration from OverallResults
    for or_elem in root.iter("OverallResults"):
        with contextlib.suppress(ValueError, TypeError):
            duration_s = float(or_elem.get("duration", 0) or 0)

    return _make_result(passed, failed, skipped, duration_s, failures, passed_names)


def _parse_catch2_text(output: str, verbose: bool = False) -> dict[str, Any]:
    """Parse Catch2 default text output (fallback when XML not available)."""
    passed = failed = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None
    duration_s: float | None = None

    # Summary: "test cases: 20 | 18 passed | 2 failed"
    summary_re = re.compile(
        r"test cases:\s*(\d+)\s*\|\s*(\d+)\s*passed\s*\|\s*(\d+)\s*failed",
        re.IGNORECASE,
    )
    m = summary_re.search(output)
    if m:
        passed = int(m.group(2))
        failed = int(m.group(3))

    # Failure blocks: dashes + test name, then file:line: FAILED:
    # -------------------------------------------------------------------------------
    # TestCaseName
    #   SectionName (optional)
    # -------------------------------------------------------------------------------
    # file.cpp:42: FAILED:
    #   REQUIRE( expr )
    # with expansion:
    #   0 > 0
    block_sep = re.compile(r"^-{50,}$", re.MULTILINE)
    sections = block_sep.split(output)
    for section_text in sections:
        lines = section_text.strip().splitlines()
        if not lines:
            continue
        tc_name = lines[0].strip() if lines else ""
        sec_name = (
            lines[1].strip() if len(lines) > 1 and not lines[1].strip().startswith("-") else ""
        )

        # Look for FAILED assertions in this block
        fail_re = re.compile(r"^(\S+\.(?:cpp|cc|cxx|h|hpp)):(\d+):\s*FAILED:", re.MULTILINE)
        expr_re = re.compile(r"REQUIRE\((.+?)\)|CHECK\((.+?)\)", re.MULTILINE)
        expand_re = re.compile(r"with expansion:\s*\n\s*(.+)", re.MULTILINE)

        for fm in fail_re.finditer(section_text):
            file_ = fm.group(1)
            line = int(fm.group(2))
            rest = section_text[fm.end() : fm.end() + 400]
            expr_m = expr_re.search(rest)
            expand_m = expand_re.search(rest)
            original = (expr_m.group(1) or expr_m.group(2) or "").strip() if expr_m else ""
            expanded = expand_m.group(1).strip() if expand_m else ""
            failures.append(
                {
                    "name": tc_name,
                    "section": sec_name,
                    "file": file_,
                    "line": line,
                    "expression": original,
                    "expanded": expanded,
                    "message": "",
                }
            )

    return _make_result(passed, failed, 0, duration_s, failures, passed_names)


def _parse_cargo_test(output: str, verbose: bool = False) -> dict[str, Any]:
    """Parse cargo test output."""
    passed = failed = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    test_re = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)", re.MULTILINE)
    for match in test_re.finditer(output):
        name = match.group(1)
        status = match.group(2)
        if status == "ok":
            passed += 1
            if verbose and passed_names is not None:
                passed_names.append(name)
        elif status == "FAILED":
            failed += 1
            failures.append({"name": name, "file": "", "line": 0, "message": ""})

    # Try to extract panic messages for failures
    # "---- test_name stdout ----\n   message\n"
    for failure in failures:
        pattern = re.compile(
            rf"---- {re.escape(failure['name'])} stdout ----\n(.*?)(?=\n----|\Z)",
            re.DOTALL,
        )
        m: re.Match[str] | None = pattern.search(output)
        if m is not None:
            failure["message"] = m.group(1).strip()[:200]

    return _make_result(passed, failed, 0, None, failures, passed_names)


def _parse_go_test(output: str, verbose: bool = False) -> dict[str, Any]:
    """Parse go test -v output."""
    passed = failed = 0
    failures: list[dict[str, Any]] = []
    passed_names: list[str] | None = [] if verbose else None

    pass_re = re.compile(r"^--- PASS:\s+(\S+)\s+", re.MULTILINE)
    fail_re = re.compile(r"^--- FAIL:\s+(\S+)\s+", re.MULTILINE)

    for m in pass_re.finditer(output):
        passed += 1
        if verbose and passed_names is not None:
            passed_names.append(m.group(1))

    for m in fail_re.finditer(output):
        name = m.group(1)
        failed += 1
        # Capture output between this FAIL and next test line
        rest = output[m.end() : m.end() + 500]
        next_test = re.search(r"^---", rest, re.MULTILINE)
        block = rest[: next_test.start()].strip() if next_test else rest.strip()
        failures.append({"name": name, "file": "", "line": 0, "message": block[:200]})

    return _make_result(passed, failed, 0, None, failures, passed_names)


# ── Command builders ──────────────────────────────────────────────────────────


def _build_test_command(
    base_cmd: str,
    framework: str | None,
    path: str | None,
    test_name: str | None,
    verbose: bool,
) -> str:
    """Augment the base test command with framework-specific flags, path, and test_name."""
    cmd = base_cmd

    if framework == "pytest" or "pytest" in base_cmd:
        # Add pytest-specific flags
        if "--tb" not in cmd:
            cmd += " --tb=short"
        if not verbose and "-v" not in cmd:
            cmd += " -q"
        if path:
            cmd += f" {path}"
        if test_name:
            cmd += f" -k {test_name}"

    elif framework == "jest":
        if "--json" not in cmd:
            cmd += " --json"
        if path:
            cmd += f" {path}"
        if test_name:
            cmd += f' --testNamePattern "{test_name}"'

    elif framework == "vitest":
        if "--reporter" not in cmd:
            cmd += " --reporter=json"
        if path:
            cmd += f" {path}"
        if test_name:
            cmd += f' --reporter=json --testNamePattern "{test_name}"'

    elif framework == "cargo-test":
        if test_name:
            cmd += f" {test_name}"
        if path:
            cmd += f" --manifest-path {path}"

    elif framework == "go-test":
        if "-v" not in cmd:
            cmd += " -v"
        if path:
            # Replace ./... with path
            cmd = re.sub(r"\./\.\.\.", path, cmd)
            if path not in cmd:
                cmd += f" {path}"
        if test_name:
            cmd += f" -run {test_name}"

    elif framework in ("ctest", "catch2"):
        if "--output-on-failure" not in cmd:
            cmd += " --output-on-failure"
        if test_name:
            cmd += f" -R {test_name}"

    else:
        # Generic: append path and test_name
        if path:
            cmd += f" {path}"
        if test_name:
            cmd += f" {test_name}"

    return cmd


# ── Tool ──────────────────────────────────────────────────────────────────────


class TestRunnerTool(BaseTool):
    name = "run_tests"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": (
                    "Run the test suite and return a structured summary: pass/fail counts, "
                    "failing test names, assertion errors, and file:line locations. "
                    "Prefer this over shell for tests. "
                    "Requires detect_project to have run first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "Override the test command. Omit to use the auto-detected test_cmd."
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Run only tests in this file or directory"
                                " (relative to project root)."
                                ' Examples: "tests/auth/", "tests/auth/test_login.py",'
                                ' "./pkg/auth/...".'
                            ),
                        },
                        "test_name": {
                            "type": "string",
                            "description": (
                                "Run only tests whose name matches this string. "
                                "Maps to -k (pytest), --testNamePattern (jest), -run (go), "
                                'substring filter (cargo). Example: "test_invalid_password".'
                            ),
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": (
                                "Show passing test names in addition to failures. Default: false."
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
        command: str | None = None,
        path: str | None = None,
        test_name: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pt = get_cached_project_type(root_path)
        framework = pt.test_framework if pt else None

        if command:
            base_cmd = command
        elif pt and pt.test_cmd:
            base_cmd = pt.test_cmd
        else:
            return {
                "stdout": (
                    "No test command available. "
                    "Run detect_project first or provide a command override."
                ),
                "stderr": "",
                "exit_code": 1,
            }

        test_cmd = _build_test_command(base_cmd, framework, path, test_name, verbose)
        result = await _shell.execute(root_path, command=test_cmd, timeout=300)
        raw = "\n".join(filter(None, [result.get("stdout", ""), result.get("stderr", "")]))

        # For jest/vitest, try to extract JSON from output
        if framework == "jest":
            json_match = re.search(r"^\{.*\}$", raw, re.DOTALL | re.MULTILINE)
            if json_match:
                parsed = _parse_jest_json(json_match.group(0), verbose)
            else:
                # Fall back to text parsing
                parsed = _parse_pytest(raw, verbose)  # rough fallback
        elif framework == "vitest":
            json_match = re.search(r"^\{.*\}$", raw, re.DOTALL | re.MULTILINE)
            if json_match:
                parsed = _parse_vitest_json(json_match.group(0), verbose)
            else:
                parsed = _parse_pytest(raw, verbose)
        elif framework == "catch2":
            # Try XML first
            xml_match = re.search(r"<\?xml|<Catch2TestRun", raw)
            if xml_match:
                parsed = _parse_catch2_xml(raw[xml_match.start() :], verbose)
            else:
                parsed = _parse_catch2_text(raw, verbose)
        elif framework == "ctest":
            parsed = _parse_ctest(raw, verbose)
        elif framework == "cargo-test":
            parsed = _parse_cargo_test(raw, verbose)
        elif framework == "go-test":
            parsed = _parse_go_test(raw, verbose)
        else:
            # Default: treat as pytest
            parsed = _parse_pytest(raw, verbose)

        return parsed
