"""TypeScript/TSX symbol parser — regex-based, covers common patterns."""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:type\s+)?(?:\{[^}]*\}|[\w*]+|\*\s+as\s+\w+)"
    r"(?:\s*,\s*(?:\{[^}]*\}|[\w*]+))*"
    r"\s+from\s+['\"]([^'\"]+)['\"]",
)
_IMPORT_SIDE_EFFECT_RE = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""")

_INTERFACE_RE = re.compile(r"^\s*(export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{")
_TYPE_ALIAS_RE = re.compile(r"^\s*(export\s+)?type\s+(\w+)\s*(?:<[^>]*>)?\s*=")
_CLASS_RE = re.compile(
    r"^\s*(export\s+(?:default\s+)?|abstract\s+)?"
    r"class\s+(\w+)"
    r"(?:\s+extends\s+(\w+(?:<[^>]*>)?))?"
    r"(?:\s+implements\s+([^{]+))?"
    r"\s*\{"
)
_FUNCTION_RE = re.compile(
    r"^\s*(export\s+(?:default\s+)?)?"
    r"(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*)?\s*\(([^)]*)\)"
    r"(?:\s*:\s*([^{]+))?"
    r"\s*\{"
)
_ENUM_RE = re.compile(r"^\s*(export\s+)?(?:const\s+)?enum\s+(\w+)\s*\{")
_ARROW_CONST_RE = re.compile(
    r"^\s*(export\s+)?(const|let)\s+(\w+)\s*"
    r"(?::\s*(?:React\.(?:FC|ComponentType|ReactElement|ReactNode)|[^=]+?))?"
    r"\s*=\s*(?:async\s+)?\(?"
)
# Method inside a class body
_METHOD_RE = re.compile(
    r"^\s*((?:public|private|protected|static|async|abstract|override|readonly)\s+)*"
    r"(?:async\s+)?"
    r"(\w+)\s*"
    r"(?:<[^>]*)?\s*\(([^)]*)\)"
    r"(?:\s*:\s*[^{;]+)?"
    r"\s*[{;]"
)

_COMPONENT_HINT_RE = re.compile(r"React\.(?:FC|ComponentType|memo|forwardRef)")
_RETURNS_JSX_RE = re.compile(r"(?:JSX\.Element|ReactElement|ReactNode)")

# Words that look like method names but aren't
_NOT_METHOD = frozenset(
    [
        "if",
        "for",
        "while",
        "switch",
        "return",
        "new",
        "delete",
        "throw",
        "catch",
        "finally",
        "typeof",
        "instanceof",
        "in",
        "of",
        "constructor",  # show separately? or include? Include it.
    ]
)


def parse(path: str) -> str:
    """Parse TypeScript/TSX file and return formatted symbol listing."""
    try:
        source = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"  [Error reading file: {e}]"

    is_tsx = path.endswith(".tsx")
    lines = source.splitlines()

    # Pass 1: collect imports grouped by source module
    imports: list[tuple[int, str, str]] = []  # (lineno, names, from_module)
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _IMPORT_RE.match(line)
        if m:
            from_mod = m.group(1)
            # Extract the names portion between import and from
            names_part = line[line.index("import") + 6 : line.lower().index(" from ")].strip()
            imports.append((i + 1, names_part, from_mod))
        else:
            m2 = _IMPORT_SIDE_EFFECT_RE.match(line)
            if m2:
                imports.append((i + 1, "(side-effect)", m2.group(1)))
        i += 1

    parts: list[str] = []

    # Format imports: group by module, show up to 5 lines
    if imports:
        shown = imports[:8]
        for lineno, names, mod in shown:
            if len(names) > 40:  # noqa: PLR2004
                names = names[:37] + "..."
            parts.append(f'  Imports: {names} from "{mod}"  # line {lineno}')
        if len(imports) > 8:  # noqa: PLR2004
            parts.append(f"  ... ({len(imports) - 8} more imports)")
        parts.append("")

    # Pass 2: extract structural elements
    brace_depth = 0
    # scope_stack: (close_depth, kind, name)  kind = 'class'|'interface'|'function'|'other'
    scope_stack: list[tuple[int, str, str]] = []

    i = 0
    while i < len(lines):
        lineno = i + 1
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith(("//", "/*", "*")):
            i += 1
            continue

        opens = stripped.count("{")
        closes = stripped.count("}")
        new_depth = brace_depth + opens - closes

        indent = "  " * (len(scope_stack) + 1)
        in_other = bool(scope_stack and scope_stack[-1][1] in ("function", "other"))
        in_class = bool(scope_stack and scope_stack[-1][1] == "class")
        in_interface = bool(scope_stack and scope_stack[-1][1] == "interface")

        matched = False

        # Interface
        if not in_other and not in_class:
            m = _INTERFACE_RE.match(line)
            if m:
                exported = bool(m.group(1))
                name = m.group(2)
                extends = m.group(3)
                ext_str = f" extends {extends.strip()}" if extends else ""
                exp_str = "export " if exported else ""
                parts.append(f"{indent}{exp_str}interface {name}{ext_str} {{  # line {lineno}")
                scope_stack.append((brace_depth, "interface", name))
                brace_depth = new_depth
                # Summarize fields
                field_count = 0
                j = i + 1
                while j < len(lines):
                    fl = lines[j].strip()
                    if fl.startswith("}"):
                        break
                    if fl and not fl.startswith("//") and ":" in fl:
                        field_count += 1
                    j += 1
                if field_count:
                    parts.append(f"{indent}  // {field_count} fields (lines {lineno + 1}-{j + 1})")
                matched = True
                i += 1
                continue

        # Type alias
        if not in_other and not in_class and not in_interface:
            m = _TYPE_ALIAS_RE.match(line)
            if m:
                exported = bool(m.group(1))
                name = m.group(2)
                # Get the value (rest of line after =)
                rest = line[line.index("=") + 1 :].strip().rstrip(";")
                if len(rest) > 50:  # noqa: PLR2004
                    rest = rest[:47] + "..."
                exp_str = "export " if exported else ""
                parts.append(f"{indent}{exp_str}type {name} = {rest}  # line {lineno}")
                if opens > 0:
                    scope_stack.append((brace_depth, "other", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        # Class
        if not in_other:
            m = _CLASS_RE.match(line)
            if m and "{" in stripped:
                is_exported = bool(m.group(1))
                name = m.group(2)
                extends = m.group(3)
                implements = m.group(4)
                ext_str = f" extends {extends}" if extends else ""
                impl_str = f" implements {implements.strip()}" if implements else ""
                exp_str = "export " if is_exported else ""
                cls_line = f"{indent}{exp_str}class {name}{ext_str}{impl_str} {{  # line {lineno}"
                parts.append(cls_line)
                scope_stack.append((brace_depth, "class", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        # Enum
        if not in_other and not in_class:
            m = _ENUM_RE.match(line)
            if m:
                exported = bool(m.group(1))
                name = m.group(2)
                exp_str = "export " if exported else ""
                parts.append(f"{indent}{exp_str}enum {name}  # line {lineno}")
                if opens > 0:
                    scope_stack.append((brace_depth, "other", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        # Function declaration
        if not in_other and not in_class and not in_interface:
            m = _FUNCTION_RE.match(line)
            if m:
                exported = bool(m.group(1))
                name = m.group(2)
                params = m.group(3) or ""
                ret_type = (m.group(4) or "").strip()
                ret_str = f": {ret_type}" if ret_type else ""
                exp_str = "export " if exported else ""
                is_async = "async " in line[: line.index("function")]
                async_str = "async " if is_async else ""
                parts.append(
                    f"{indent}{exp_str}{async_str}function {name}({params}){ret_str}"
                    f"  # line {lineno}"
                )
                if opens > 0:
                    scope_stack.append((brace_depth, "function", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        # Arrow function / exported const
        if not in_other and not in_class and not in_interface:
            m = _ARROW_CONST_RE.match(line)
            if m:
                exported = bool(m.group(1))
                name = m.group(3)
                # Detect React component: PascalCase in .tsx OR has React.FC type
                is_component = (
                    (is_tsx and name[0].isupper())
                    or bool(_COMPONENT_HINT_RE.search(line))
                    or bool(_RETURNS_JSX_RE.search(line))
                )

                exp_str = "export " if exported else ""
                comp_tag = " [component]" if is_component else ""
                # Get the type annotation if present
                type_ann = ""
                colon_pos = line.find(":", line.find(name) + len(name))
                eq_pos = line.find("=", line.find(name) + len(name))
                if 0 < colon_pos < eq_pos:
                    type_ann_raw = line[colon_pos + 1 : eq_pos].strip()
                    if len(type_ann_raw) > 30:  # noqa: PLR2004
                        type_ann_raw = type_ann_raw[:27] + "..."
                    type_ann = f": {type_ann_raw}"
                parts.append(f"{indent}{exp_str}const {name}{type_ann}{comp_tag}  # line {lineno}")
                if opens > 0:
                    scope_stack.append((brace_depth, "function", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        # Method inside class body
        if in_class and not in_other:
            m = _METHOD_RE.match(line)
            if m:
                modifiers = (m.group(1) or "").strip()
                name = m.group(2)
                params = m.group(3) or ""
                if name not in _NOT_METHOD:
                    mod_str = f"{modifiers} " if modifiers else ""
                    parts.append(f"{indent}{mod_str}{name}({params})  # line {lineno}")
                if opens > closes:
                    scope_stack.append((brace_depth, "function", name))
                brace_depth = new_depth
                matched = True
                i += 1
                continue

        if not matched:
            if opens > closes and in_other:
                scope_stack.append((brace_depth, "other", ""))
            brace_depth = new_depth

        # Pop closed scopes
        while scope_stack and scope_stack[-1][0] >= brace_depth:
            scope_stack.pop()

        i += 1

    return "\n".join(parts).rstrip() if parts else "  (no symbols found)"
