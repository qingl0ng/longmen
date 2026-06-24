"""Regex-based fact extraction from conversation text.

Facts are extracted before compaction and appended to the summary block so that
key references (file paths, URLs, decisions) survive even after the model-generated
summary compresses the conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# File paths: sequences like /foo/bar.py, src/auth/login.py, or Dockerfile
_RE_FILE_PATH = re.compile(
    r"""
    (?:^|[\s"'`=(,])          # preceded by whitespace or delimiters
    (                          # capture the path
        (?:\.{1,2}/)?          # optional leading ./  or ../
        [\w./\-]+              # path segments
        (?:
            \.                 # either has an extension…
            [a-zA-Z0-9]{1,10}  # …extension (py, ts, cpp, toml, …)
            (?:/[\w./\-]*)?    # optional trailing path (e.g. dir/)
            |
            /[\w./\-]*         # …or contains a slash (bare filename in a dir)
        )
    )
    (?:$|[\s"'`),:\]])        # followed by whitespace or delimiters
    """,
    re.VERBOSE,
)

# Well-known extensionless filenames (e.g. Dockerfile, Makefile, Jenkinsfile)
_RE_KNOWN_BARE_FILES = re.compile(
    r"""
    (?:^|[\s"'`=(,/])
    (
        (?:Docker|Container)file[\w.\-]*
        | Makefile[\w.\-]*
        | Jenkinsfile[\w.\-]*
        | Vagrantfile[\w.\-]*
        | Procfile
        | \.env[\w.\-]*
        | \.gitignore | \.gitattributes | \.gitmodules
        | \.dockerignore
        | \.eslintrc | \.babelrc | \.prettierrc
        | CLAUDE\.md | README | CHANGELOG | LICENSE | AUTHORS
    )
    (?:$|[\s"'`),:\]])
    """,
    re.VERBOSE,
)

# URLs: http/https
_RE_URL = re.compile(
    r"https?://[^\s\"'<>)}\]]+",
    re.IGNORECASE,
)

# Error codes: E1234, error[E0308], HTTP 404, exit code 1
_RE_ERROR_CODE = re.compile(
    r"""
    (?:
        error\[?\w+\]?         # Rust/mypy: error[E0308]
        | E\d{4}               # Python: E0308
        | \bHTTP\s+\d{3}\b     # HTTP status codes
        | \bexit\s+code\s+\d+\b  # exit codes
        | exit_code:\s*[1-9]\d*   # exit_code: N
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Function/class names: backtick-quoted, or def/class declarations
_RE_IDENTIFIER = re.compile(
    r"""
    (?:
        `([\w]+(?:\.[\w]+)*(?:\(\))?)` # backtick-quoted identifier
        | \bdef\s+([\w]+)              # def func_name
        | \bclass\s+([\w]+)            # class ClassName
        | \bfn\s+([\w]+)               # Rust fn
        | \bfunc\s+([\w]+)             # Go func
    )
    """,
    re.VERBOSE,
)

# Decisions: lines containing decision keywords
_RAG_HEADER_RE = re.compile(r'^RAG Search: "(.+?)" \(\d+ of \d+ results\)')
_RAG_SOURCE_RE = re.compile(r'^\[\d+\] (.+)$')

_RE_DECISION = re.compile(
    r"""
    (?:^|[.\n])
    [^\n]*
    (?:
        \bdecided\b | \bchose\b | \bwill\s+use\b | \bapproach:\b
        | \bgoing\s+with\b | \busing\b\s+\w+\s+for\b
        | \bwill\s+implement\b | \busing\s+RAII\b
    )
    [^\n]+
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Configuration values: key = value, --flag value
_RE_CONFIG = re.compile(
    r"""
    (?:
        [\w_]+\s*=\s*[\w\d"'./\-:]+   # key = value
        | --[\w\-]+=[\w\d"'./\-:]+    # --flag=value
        | --[\w\-]+\s+\d+              # --flag 60
    )
    """,
    re.VERBOSE,
)

# Minimum lengths to avoid noise
_MIN_PATH_LEN = 5
_MIN_DECISION_LEN = 20


# ---------------------------------------------------------------------------
# ExtractedFacts
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFacts:
    file_paths: set[str] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)
    error_codes: list[str] = field(default_factory=list)
    identifiers: set[str] = field(default_factory=set)
    decisions: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    custom: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FactExtractor
# ---------------------------------------------------------------------------


class FactExtractor:
    """Extract structured facts from conversation text using regex patterns."""

    def _extract_rag_facts(self, text: str, facts: ExtractedFacts) -> None:
        for line in text.splitlines():
            m = _RAG_HEADER_RE.match(line)
            if m:
                facts.decisions.append(f'RAG search: "{m.group(1)}"')
                continue
            m = _RAG_SOURCE_RE.match(line)
            if m:
                facts.identifiers.add(m.group(1))

    def extract(self, text: str) -> ExtractedFacts:
        facts = ExtractedFacts()

        if text.startswith("RAG Search:"):
            self._extract_rag_facts(text, facts)
            return facts

        # File paths (with extension or containing a slash)
        for match in _RE_FILE_PATH.finditer(text):
            path = match.group(1).strip()
            if len(path) >= _MIN_PATH_LEN and ("/" in path or path.startswith(".")):
                facts.file_paths.add(path)

        # Well-known extensionless filenames
        for match in _RE_KNOWN_BARE_FILES.finditer(text):
            path = match.group(1).strip()
            if path:
                facts.file_paths.add(path)

        # URLs
        for match in _RE_URL.finditer(text):
            url = match.group(0).rstrip(".,;)")
            facts.urls.add(url)

        # Error codes (deduplicated, preserve first occurrence order)
        seen_errors: set[str] = set()
        for match in _RE_ERROR_CODE.finditer(text):
            code = match.group(0).strip()
            if code not in seen_errors:
                seen_errors.add(code)
                facts.error_codes.append(code)

        # Identifiers (function/class names)
        for match in _RE_IDENTIFIER.finditer(text):
            # One of the groups will be non-None
            for g in match.groups():
                if g:
                    facts.identifiers.add(g)

        # Decisions
        seen_decisions: set[str] = set()
        for match in _RE_DECISION.finditer(text):
            decision = match.group(0).strip().lstrip(".\n ")
            if len(decision) >= _MIN_DECISION_LEN:
                # Deduplicate and limit to first 80 chars
                key = decision[:80]
                if key not in seen_decisions:
                    seen_decisions.add(key)
                    facts.decisions.append(decision[:120])

        # Primary: matches the Reason line prepended to revise_plan tool results
        plain_revision_reason_pattern = re.compile(
            r"^Reason:\s+(.+?)$",
            re.MULTILINE,
        )
        for match in plain_revision_reason_pattern.finditer(text):
            reason_text = match.group(1).strip()
            decision = f"Plan revised: {reason_text}"
            if len(decision) >= _MIN_DECISION_LEN:
                key = decision[:80]
                if key not in seen_decisions:
                    seen_decisions.add(key)
                    facts.decisions.append(decision[:120])

        # Fallback: matches the summary line (covers tool results without a Reason line)
        plain_revision_pattern = re.compile(
            r"Plan revised:\s+(.+?)(?:\n|$)",
        )
        for match in plain_revision_pattern.finditer(text):
            description = match.group(1).strip()
            decision = f"Plan revised: {description}"
            if len(decision) >= _MIN_DECISION_LEN:
                key = decision[:80]
                if key not in seen_decisions:
                    seen_decisions.add(key)
                    facts.decisions.append(decision[:120])

        # Configuration values: key = value, --flag value
        for match in _RE_CONFIG.finditer(text):
            config = match.group(0).strip()
            if config:
                facts.custom["config"] = config

        return facts

    def extract_from_messages(
        self,
        messages: list[dict[str, str | list[dict[str, str]]]],
        commands: list[str] | None = None,
    ) -> ExtractedFacts:
        """Extract facts from a list of OpenAI-format message dicts."""
        text_parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

        facts = self.extract("\n".join(text_parts))
        if commands:
            facts.commands_run = commands
        return facts

    def format_facts(self, facts: ExtractedFacts) -> str:
        """Format extracted facts as a structured block to append to a compacted summary."""
        lines: list[str] = ["[Extracted references — do not hallucinate alternatives]:"]

        if facts.file_paths:
            paths_sorted = sorted(facts.file_paths)
            lines.append(f"Files: {', '.join(paths_sorted)}")

        if facts.urls:
            urls_sorted = sorted(facts.urls)
            lines.append(f"URLs: {', '.join(urls_sorted)}")

        if facts.error_codes:
            lines.append(f"Errors encountered: {', '.join(facts.error_codes)}")

        if facts.identifiers:
            idents_sorted = sorted(facts.identifiers)
            lines.append(f"Functions/Classes: {', '.join(idents_sorted)}")

        if facts.decisions:
            lines.append("Decisions: " + " | ".join(facts.decisions))

        if facts.commands_run:
            lines.append(f"Commands: {', '.join(facts.commands_run)}")

        if len(lines) == 1:
            # No facts extracted
            return ""

        return "\n".join(lines)
