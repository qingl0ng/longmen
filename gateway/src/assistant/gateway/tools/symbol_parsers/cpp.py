"""C++ symbol parser — regex-based, covers common patterns in headers and source files."""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_INCLUDE_RE = re.compile(r"^\s*#\s*include\s+([<\"][^>\"]+[>\"])")
_NAMESPACE_RE = re.compile(r"^\s*namespace\s*([\w:]+)?\s*\{")
_TEMPLATE_RE = re.compile(r"^\s*template\s*<")
_CLASS_STRUCT_RE = re.compile(
    r"^\s*(class|struct)\s+(\w+)"
    r"(?:\s+final)?"
    r"(?:\s*:\s*((?:[^{;])+?))?"
    r"\s*\{"
)
_ENUM_RE = re.compile(r"^\s*enum\s+(?:(class|struct)\s+)?(\w+)")
_USING_RE = re.compile(r"^\s*using\s+(\w+)\s*=")

_CONTROL_FLOW_WORDS: frozenset[str] = frozenset(
    [
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "break",
        "continue",
        "return",
        "goto",
        "try",
        "catch",
        "throw",
        "delete",
        "new",
        "sizeof",
        "decltype",
        "static_assert",
        "assert",
        "ASSERT",
        "CHECK",
        "DCHECK",
        "EXPECT",
        "REQUIRE",
    ]
)


def _is_func_line(stripped: str) -> bool:
    """Return True if the line looks like a function/method declaration."""
    if not stripped or "(" not in stripped:
        return False
    # Skip comments and preprocessor
    if stripped[0] in ("/", "#", "*", "!"):
        return False
    # Skip pure brace lines
    if re.match(r"^[{};\s]*$", stripped):
        return False
    # Skip access specifiers
    if re.match(r"^(public|private|protected)\s*:", stripped):
        return False
    # Skip single-word lines ending with :
    if re.match(r"^\w+\s*:$", stripped):
        return False
    # Get first word (strip ~ for destructor: ~ClassName())
    first_m = re.match(r"(\w+)", stripped.lstrip("~"))
    if not first_m:
        return False
    first_word = first_m.group(1)
    if first_word in _CONTROL_FLOW_WORDS:
        return False
    # Must have an identifier immediately before (
    paren_pos = stripped.index("(")
    before_paren = stripped[:paren_pos].rstrip()
    if not before_paren:
        return False
    last_char = before_paren[-1]
    if not (last_char.isalnum() or last_char in ("_", ">")):
        return False
    # Skip variable assignments: `type name = expr(...)`
    return " = " not in before_paren


def _clean_sig(stripped: str) -> str:
    """Remove trailing { and ; from a function declaration line."""
    sig = stripped.rstrip()
    # Remove trailing {
    if sig.endswith("{"):
        sig = sig[:-1].rstrip()
    # Remove trailing ;
    if sig.endswith(";"):
        sig = sig[:-1].rstrip()
    if len(sig) > 85:  # noqa: PLR2004
        sig = sig[:82] + "..."
    return sig


def parse(path: str) -> str:
    """Parse C++ file and return formatted symbol listing."""
    try:
        source = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"  [Error reading file: {e}]"

    lines = source.splitlines()

    # Collect includes
    includes: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = _INCLUDE_RE.match(line)
        if m:
            includes.append((idx + 1, m.group(1)))

    parts: list[str] = []

    if includes:
        names = [inc[1] for inc in includes[:10]]
        suffix = ", ..." if len(includes) > 10 else ""  # noqa: PLR2004
        lo, hi = includes[0][0], includes[-1][0]
        rng = f"lines {lo}-{hi}" if lo != hi else f"line {lo}"
        parts.append(f"  #include: {', '.join(names)}{suffix} ({rng})")
        parts.append("")

    # State machine
    brace_depth = 0
    # Each entry: (depth_at_open, kind, name)
    # kind: 'namespace' | 'class' | 'struct' | 'function'
    scope_stack: list[tuple[int, str, str]] = []
    pending_template: str | None = None
    pending_template_lineno: int = 0

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        if not stripped:
            pending_template = None
            continue
        # Skip comments
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        # Skip preprocessor (includes already handled)
        if stripped.startswith("#"):
            continue

        opens = stripped.count("{")
        closes = stripped.count("}")
        new_depth = brace_depth + opens - closes

        indent = "  " * (len(scope_stack) + 1)
        # Are we inside a function/method body?
        in_func = bool(scope_stack and scope_stack[-1][1] == "function")

        # Template prefix — captures and waits for next declaration
        if _TEMPLATE_RE.match(stripped):
            if not in_func:
                pending_template = stripped
                pending_template_lineno = lineno
            brace_depth = new_depth
            continue

        matched = False

        # Namespace
        if not in_func:
            m = _NAMESPACE_RE.match(stripped)
            if m and opens > 0:
                name = m.group(1) or "(anonymous)"
                if pending_template:
                    parts.append(f"{indent}{pending_template}  # line {pending_template_lineno}")
                    pending_template = None
                parts.append(f"{indent}namespace {name} {{  # line {lineno}")
                parts.append("")
                scope_stack.append((brace_depth, "namespace", name))
                brace_depth = new_depth
                matched = True

        # Class or struct
        if not matched and not in_func:
            m = _CLASS_STRUCT_RE.match(stripped)
            if m:
                kind = m.group(1)  # 'class' or 'struct'
                name = m.group(2)
                bases = m.group(3)
                base_str = f" : {bases.strip()}" if bases else ""
                if pending_template:
                    parts.append(f"{indent}{pending_template}  # line {pending_template_lineno}")
                    pending_template = None
                parts.append(f"{indent}{kind} {name}{base_str} {{  # line {lineno}")
                scope_stack.append((brace_depth, kind, name))
                brace_depth = new_depth
                matched = True

        # Enum
        if not matched and not in_func:
            m = _ENUM_RE.match(stripped)
            if m:
                is_enum_class = bool(m.group(1))
                name = m.group(2)
                label = "enum class" if is_enum_class else "enum"
                parts.append(f"{indent}{label} {name}  # line {lineno}")
                if opens > 0:
                    scope_stack.append((brace_depth, "function", ""))
                brace_depth = new_depth
                matched = True

        # Using alias
        if not matched and not in_func:
            m = _USING_RE.match(stripped)
            if m:
                name = m.group(1)
                parts.append(f"{indent}using {name} = ...  # line {lineno}")
                brace_depth = new_depth
                matched = True

        # Function / method
        if not matched and not in_func and _is_func_line(stripped):
            sig = _clean_sig(stripped)
            parts.append(f"{indent}{sig}  # line {lineno}")
            if opens > closes:
                # Function with body — track to avoid showing inner calls
                scope_stack.append((brace_depth, "function", ""))
            brace_depth = new_depth
            matched = True

        if not matched:
            # Track nested scopes inside function bodies
            if in_func and opens > closes:
                scope_stack.append((brace_depth, "function", ""))
            brace_depth = new_depth

        # Pop scopes that are now closed
        while scope_stack and scope_stack[-1][0] >= brace_depth:
            closed = scope_stack.pop()
            if closed[1] == "namespace":
                ns_indent = "  " * (len(scope_stack) + 1)
                parts.append(f"{ns_indent}}} // namespace {closed[2]}")
                parts.append("")

        pending_template = None

    return "\n".join(parts).rstrip() if parts else "  (no symbols found)"
