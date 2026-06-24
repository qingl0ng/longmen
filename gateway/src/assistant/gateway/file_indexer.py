"""Walk a project root and return gitignore-aware relative file paths."""

import os
from pathlib import Path

from .tools.file_read import _load_gitignore_patterns, _should_skip


def build_file_index(root_path: Path) -> list[str]:
    """Return sorted POSIX relative paths of all non-ignored files under root_path."""
    patterns = _load_gitignore_patterns(root_path)
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dp = Path(dirpath)
        # Prune skipped directories in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip(dp / d, root_path, patterns)
        ]
        for name in filenames:
            path = dp / name
            if not _should_skip(path, root_path, patterns):
                result.append(path.relative_to(root_path).as_posix())
    result.sort()
    return result
