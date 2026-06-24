"""Python AST-based symbol parser."""

from __future__ import annotations

import ast


def parse(source: str) -> str:
    """Extract Python symbols using AST. Returns formatted string."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"  [SyntaxError: {e}]"

    parts: list[str] = []

    # Imports summary
    import_lines: list[int] = []
    import_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_lines.append(node.lineno)
            import_names.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            import_lines.append(node.lineno)
            if node.module:
                import_names.append(node.module.split(".")[0])

    if import_lines:
        unique: list[str] = list(dict.fromkeys(import_names))
        shown = unique[:6]
        suffix = ", ..." if len(unique) > 6 else ""  # noqa: PLR2004
        lo, hi = min(import_lines), max(import_lines)
        rng = f"lines {lo}-{hi}" if lo != hi else f"line {lo}"
        parts.append(f"  Imports: {', '.join(shown)}{suffix} ({rng})")
        parts.append("")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            parts.append(_class_line(node, 2))
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    parts.append(_func_line(child, 4))
                elif isinstance(child, ast.ClassDef):
                    parts.append(_class_line(child, 4))
                    for grandchild in child.body:
                        if isinstance(grandchild, ast.FunctionDef | ast.AsyncFunctionDef):
                            parts.append(_func_line(grandchild, 6))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(_func_line(node, 2))

    return "\n".join(parts)


def _class_line(node: ast.ClassDef, indent: int) -> str:
    p = " " * indent
    dec = _decs(node.decorator_list)
    bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
    base_str = f"({bases})" if bases else ""
    return f"{p}{dec}class {node.name}{base_str}:  # line {node.lineno}"


def _func_line(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: int) -> str:
    p = " " * indent
    dec = _decs(node.decorator_list)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    sig = _build_sig(node)
    if len(sig) > 88 - indent:  # noqa: PLR2004
        sig = sig[: 85 - indent] + "..."
    return f"{p}{dec}{prefix}def {sig}  # line {node.lineno}"


def _build_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    params: list[str] = []
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        di = i - defaults_offset
        if di >= 0:
            s += f" = {ast.unparse(args.defaults[di])}"
        params.append(s)

    if args.vararg:
        va = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            va += f": {ast.unparse(args.vararg.annotation)}"
        params.append(va)

    for i, arg in enumerate(args.kwonlyargs):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            s += f" = {ast.unparse(args.kw_defaults[i])}"  # type: ignore[arg-type]
        params.append(s)

    if args.kwarg:
        kw = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            kw += f": {ast.unparse(args.kwarg.annotation)}"
        params.append(kw)

    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({', '.join(params)}){ret}"


def _decs(decorator_list: list[ast.expr]) -> str:
    if not decorator_list:
        return ""
    names: list[str] = []
    for d in decorator_list:
        if isinstance(d, ast.Name):
            names.append(f"@{d.id}")
        elif isinstance(d, ast.Attribute):
            names.append(f"@{ast.unparse(d)}")
        elif isinstance(d, ast.Call):
            fn = d.func
            if isinstance(fn, ast.Name):
                names.append(f"@{fn.id}")
            elif isinstance(fn, ast.Attribute):
                names.append(f"@{ast.unparse(fn)}")
            else:
                names.append(f"@{ast.unparse(d)}")
        else:
            names.append(f"@{ast.unparse(d)}")
    return " ".join(names) + " "
