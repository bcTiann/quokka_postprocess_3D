"""Resolve user paths at configuration and command-line entry points."""
from pathlib import Path


def resolve_path(
    path: str | Path,
    *,
    base_directory: Path | None = None,
) -> Path:
    """Return an absolute Path after expanding a leading ~.

    Relative paths use base_directory when supplied (the YAML directory),
    otherwise the current working directory (command-line arguments).
    This does not create files or require them to exist. $VAR stays literal.
    Example: resolve_path("../inputs/table.npz", base_directory=config.parent).
    """
    expanded_path = Path(path).expanduser()
    if base_directory is not None and not expanded_path.is_absolute():
        expanded_path = base_directory / expanded_path
    return expanded_path.resolve()
