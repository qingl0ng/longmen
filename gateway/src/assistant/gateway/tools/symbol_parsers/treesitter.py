"""Tree-sitter universal fallback parser for languages without a dedicated parser."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

# Map file extension to (module_name, language_variant)
# language_variant is passed to _load_language() to handle packages
# that expose multiple grammars (e.g., tree-sitter-typescript).
_EXT_TO_GRAMMAR: dict[str, tuple[str, str]] = {
    ".js": ("tree_sitter_javascript", "default"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".tsx": ("tree_sitter_typescript", "tsx"),
    ".rs": ("tree_sitter_rust", "default"),
    ".go": ("tree_sitter_go", "default"),
    ".java": ("tree_sitter_java", "default"),
    ".c": ("tree_sitter_c", "default"),
    ".cpp": ("tree_sitter_cpp", "default"),
    ".cc": ("tree_sitter_cpp", "default"),
    ".cxx": ("tree_sitter_cpp", "default"),
    ".h": ("tree_sitter_cpp", "default"),
    ".hpp": ("tree_sitter_cpp", "default"),
    ".hxx": ("tree_sitter_cpp", "default"),
}

# Node types whose names are worth showing, per language family
_SYMBOL_TYPES: dict[str, list[str]] = {
    "tree_sitter_python": ["function_definition", "class_definition"],
    "tree_sitter_javascript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    ],
    "tree_sitter_typescript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "tree_sitter_rust": [
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
    ],
    "tree_sitter_go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "short_var_declaration",
    ],
    "tree_sitter_java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    ],
    "tree_sitter_c": ["function_definition", "struct_specifier", "enum_specifier"],
    "tree_sitter_cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "template_declaration",
    ],
}


def _load_language(module_name: str, variant: str) -> Any | None:
    """Load a tree-sitter Language object. Returns None if not available."""
    try:
        from tree_sitter import Language

        mod = importlib.import_module(module_name)

        # Try variant-specific function names first
        if variant == "tsx":
            for attr in ("language_tsx", "language_typescript", "language"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    try:
                        return Language(fn())
                    except Exception:  # noqa: BLE001
                        pass
        elif variant == "typescript":
            for attr in ("language_typescript", "language"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    try:
                        return Language(fn())
                    except Exception:  # noqa: BLE001
                        pass
        else:
            fn = getattr(mod, "language", None)
            if fn is not None:
                return Language(fn())
    except (ImportError, Exception):  # noqa: BLE001
        pass
    return None


def _get_node_name(node: Any) -> str | None:
    """Extract a display name from a tree-sitter node."""
    # Try common field names across languages
    for field in ("name", "identifier"):
        child = node.child_by_field_name(field)
        if child is not None and child.text:
            return str(child.text.decode("utf-8", errors="replace"))

    # Rust/Go: the 'name' field might be nested in 'declarator'
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        inner = declarator.child_by_field_name("declarator")
        target = inner if inner is not None else declarator
        if target.text:
            text: str = target.text.decode("utf-8", errors="replace")
            # Strip pointer/reference qualifiers
            return text.lstrip("*&").strip().split("(")[0].strip()

    return None


def _walk_symbols(node: Any, symbol_types: list[str]) -> list[tuple[int, str, str]]:
    """Walk AST and collect (lineno, node_type, name) for symbol nodes."""
    results: list[tuple[int, str, str]] = []

    def _walk(n: Any) -> None:
        if n.type in symbol_types:
            name = _get_node_name(n)
            if name:
                lineno = n.start_point[0] + 1
                results.append((lineno, n.type, name))
        for child in n.children:
            _walk(child)

    _walk(node)
    return results


def try_parse(path: str) -> str | None:
    """
    Try to parse the file with tree-sitter.

    Returns a formatted symbol string if successful, or None if:
    - No grammar is available for this file extension
    - tree-sitter is not installed
    - Parsing fails for any reason
    """
    ext = Path(path).suffix.lower()
    if ext not in _EXT_TO_GRAMMAR:
        return None

    module_name, variant = _EXT_TO_GRAMMAR[ext]
    language = _load_language(module_name, variant)
    if language is None:
        return None

    try:
        from tree_sitter import Parser

        parser = Parser(language)
        source_bytes = Path(path).read_bytes()
        tree = parser.parse(source_bytes)
    except Exception:  # noqa: BLE001
        return None

    symbol_types = _SYMBOL_TYPES.get(module_name, ["function_definition", "class_definition"])
    symbols = _walk_symbols(tree.root_node, symbol_types)

    if not symbols:
        return None

    lines = [f"  line {lineno:4d}: {name}  [{node_type}]" for lineno, node_type, name in symbols]
    return "\n".join(lines)
