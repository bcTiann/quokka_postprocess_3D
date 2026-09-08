"""Opt-in, durable completed-point storage for long DESPOTIC builds.

Only a completed atomic rename commits a point. Temporary files left by an
interruption are not results. A directory belongs to one exact build manifest;
incompatible or damaged committed records are errors, never cache misses.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
from importlib.metadata import distribution
import json
import numbers
import os
from pathlib import Path
import platform
import tempfile

from .models import AttemptRecord, LineLumResult


def _encode(value):
    # Hex floats preserve every bit and represent NaN without nonstandard JSON.
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return {"$float": float(value).hex()}
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"$float"}:
            return float.fromhex(value["$float"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _canonical(value) -> bytes:
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_record(path: Path, payload) -> None:
    raw = _canonical(payload)
    envelope = json.dumps({"sha256": hashlib.sha256(raw).hexdigest(),
                           "payload": raw.decode("utf-8")}).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(envelope)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_record(path: Path):
    try:
        envelope = json.loads(path.read_bytes())
        if set(envelope) != {"sha256", "payload"}:
            raise ValueError("invalid record envelope")
        raw = envelope["payload"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
            raise ValueError("checksum mismatch")
        return _decode(json.loads(raw))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuntimeError(f"Corrupt DESPOTIC checkpoint: {path}: {exc}") from exc


def source_metadata() -> dict:
    """Hash the actual solver sources and local collision data before reuse."""
    from .solver import _configure_despotic_home

    _configure_despotic_home()
    local = Path(__file__).resolve().parent
    package = Path(distribution("despotic").locate_file("despotic"))
    lamda = Path(os.environ["DESPOTIC_HOME"]) / "LAMDA"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "python": platform.python_version(),
        "numpy": distribution("numpy").version,
        "project_sources": {
            name: digest(local / name) for name in (
                "builder.py", "checkpoint.py", "solver.py", "thermal_solver.py",
                "abundances.py", "models.py",
            )
        },
        "despotic_sources": {
            str(path.relative_to(package)): digest(path)
            for path in sorted(package.rglob("*.py"))
        },
        "lamda_data": {
            name: digest(lamda / name)
            for name in ("co.dat", "catom.dat", "c+.dat", "hco+.dat", "oatom.dat")
        },
    }


@dataclass(frozen=True)
class PointCheckpoints:
    directory: Path
    manifest_digest: str

    @classmethod
    @contextmanager
    def open(cls, directory: Path, manifest: dict):
        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # The parent holds this lock until every worker has finished. Workers
        # receive only the picklable store, and write disjoint point paths.
        with (directory / ".lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Checkpoint directory is already in use: {directory}") from exc
            try:
                expected = {"format_version": 1, **manifest}
                raw = _canonical(expected)
                path = directory / "manifest.json"
                if path.exists():
                    if _canonical(_read_record(path)) != raw:
                        raise RuntimeError(f"Incompatible DESPOTIC checkpoint manifest: {path}")
                else:
                    existing = [p for p in directory.iterdir()
                                if p.name != ".lock" and not p.name.startswith(".tmp-")]
                    if existing:
                        raise RuntimeError(f"Checkpoint directory has data but no manifest: {directory}")
                    _write_record(path, expected)
                yield cls(directory, hashlib.sha256(raw).hexdigest())
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _path(self, indices) -> Path:
        row, col, dvdr = indices
        return self.directory / f"row-{row:05d}" / f"point-{col:05d}-{dvdr:05d}.json"

    def load(self, indices, coordinates):
        path = self._path(indices)
        if not path.exists():
            return None
        payload = _read_record(path)
        try:
            if (payload["manifest_digest"] != self.manifest_digest
                    or payload["indices"] != list(indices)
                    or payload["coordinates"] != list(coordinates)):
                raise ValueError("point identity does not match the requested grid")
            result = payload["result"]
            if len(result) != 8 or not isinstance(result[-1], bool):
                raise ValueError("invalid point result schema")
            result[0] = {name: LineLumResult(**line) for name, line in result[0].items()}
            result[6] = dict(result[6])
            attempts = [AttemptRecord(**item) for item in payload["attempts"]]
            if any((item.row_idx, item.col_idx, item.dvdr_idx) != tuple(indices)
                   or (item.nH, item.colDen, item.dvdr) != tuple(coordinates)
                   for item in attempts):
                raise ValueError("attempt identity does not match the requested grid")
            return tuple(result), attempts
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RuntimeError(f"Corrupt DESPOTIC point checkpoint: {path}: {exc}") from exc

    def save(self, indices, coordinates, result, attempts) -> None:
        path = self._path(indices)
        if not path.parent.exists():
            path.parent.mkdir()
            _sync_directory(self.directory)
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite completed checkpoint: {path}")
        line_results, *remaining = result
        serial_result = [{name: asdict(line) for name, line in line_results.items()}, *remaining]
        # Energy-term insertion order is part of the final NPZ's names array.
        serial_result[6] = list(result[6].items())
        _write_record(path, {
            "manifest_digest": self.manifest_digest,
            "indices": indices,
            "coordinates": coordinates,
            "result": serial_result,
            "attempts": [asdict(item) for item in attempts],
        })
