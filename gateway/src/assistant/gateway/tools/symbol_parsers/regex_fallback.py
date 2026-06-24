"""Generic regex fallback symbol parser for unsupported languages."""

from __future__ import annotations

import re

_GENERIC_PAT = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?"
    r"(?:def |class |function |fn |impl |struct |enum |interface |type )\w",
    re.MULTILINE,
)


def parse(source: str) -> str:
    """Return a list of symbol lines found via simple regex matching."""
    found: list[str] = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if _GENERIC_PAT.match(stripped):
            found.append(f"  line {i:4d}: {stripped[:80]}")
    return "\n".join(found) if found else "  (no symbols found)"
