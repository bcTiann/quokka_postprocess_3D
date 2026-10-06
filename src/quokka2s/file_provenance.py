"""Identify build and diagnostic input files without loading them into memory."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Return the file's SHA256 digest as a 64-character hexadecimal string.

    path names an existing file, such as a raw emission table or source file.
    Read at most 1 MiB at a time. Build reports use this identity; process-time
    table compatibility is checked separately from file hashes.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
