"""List snapshot directory names without reading simulation or result contents."""
from __future__ import annotations

from pathlib import Path
import re


def is_snapshot_name(name: str) -> bool:
    """Recognize plt followed by digits, such as plt0008703."""
    return re.fullmatch(r"plt\d+", name) is not None


def list_snapshot_directories(directory: Path) -> list[Path]:
    """Return direct plt+digits directories in numeric order, without recursion.

    The returned paths retain their directory names, including symlink aliases.
    A matching broken symlink is an error. Header and payload completeness are
    checked by the snapshot processor, not inferred from a directory name.
    """
    snapshots = []
    for path in directory.iterdir():
        if not is_snapshot_name(path.name):
            continue
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"Broken snapshot symlink: {path}")
        if path.is_dir():
            snapshots.append(path)
    snapshots.sort(key=lambda path: (int(path.name[3:]), path.name))
    return snapshots
