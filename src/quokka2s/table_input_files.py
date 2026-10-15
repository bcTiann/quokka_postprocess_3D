"""Save small survey records outside simulation trees; reuse only current scans."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from quokka2s.table_input_statistics import input_definitions


def snapshot_source_paths(*, dataset: Path) -> list[Path]:
    """List source metadata and level-zero payload paths, without opening them."""
    paths = [dataset / "Header", dataset / "metadata.yaml", dataset / "Level_0/Cell_H"]
    paths.extend(sorted((dataset / "Level_0").glob("Cell_D_*")))
    return paths


def snapshot_file_state(*, dataset: Path) -> list[dict]:
    """Record source sizes/mtimes without reading or hashing binary contents.

    Header, metadata and Cell_H define the layout/units. Cell_D state detects
    changed payload files on a later invocation. Files are inspected only;
    no permission, content or metadata modification is requested.
    """
    state = []
    for path in snapshot_source_paths(dataset=dataset):
        info = path.stat()
        state.append({
            "path": str(path.resolve()),
            "size_bytes": info.st_size,
            "modified_ns": info.st_mtime_ns,
        })
    return state


class UnsafeSurveyOutput(ValueError):
    """An output resolves into source data or outside its allowed destination."""


class StaleInputStatistics(ValueError):
    """Saved statistics belong to changed data, definitions or selection."""


class SurveyOutputWriteError(RuntimeError):
    """Saving a survey record failed; further expensive scans must stop."""


class SurveyOutputDirectory:
    """Write small survey records within a checked user-owned output directory.

    directory is a user-owned output Path, never a simulation directory.
    protected_roots contains source trees and resolved input-file directories.
    allowed_output_root optionally restricts this directory to the batch's
    output root. Symlink checks apply before scanning and before every write.
    Only paths are retained; no datasets or cell arrays belong to this object.
    """

    def __init__(
        self,
        *,
        directory: Path,
        protected_roots: tuple[Path, ...],
        allowed_output_root: Path | None = None,
    ):
        self.directory = directory.resolve()
        self.protected_roots = tuple(root.resolve() for root in protected_roots)
        self.allowed_output_root = self.directory
        if allowed_output_root is not None:
            self.allowed_output_root = allowed_output_root.resolve()
        self.check_output_path(path=self.directory)

    def check_output_path(self, *, path: Path) -> None:
        """Reject a destination inside a source tree or outside this output root."""
        resolved = path.resolve()
        if not resolved.is_relative_to(self.directory):
            raise UnsafeSurveyOutput(f"Survey output symlink escapes its directory: {path}")
        if not resolved.is_relative_to(self.allowed_output_root):
            raise UnsafeSurveyOutput(f"Survey output leaves the allowed output root: {path}")
        for root in self.protected_roots:
            if resolved.is_relative_to(root):
                raise UnsafeSurveyOutput(f"Survey output points into a source tree: {path}")

    def write_text(self, *, path: Path, text: str) -> None:
        """Atomically save prepared JSON/CSV text within the checked output root."""
        self.check_output_path(path=path)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=".table-input-", dir=self.directory)
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w") as output:
                    self.check_output_path(path=temporary_path)
                    output.write(text)
                self.check_output_path(path=path)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError as error:
            raise SurveyOutputWriteError(f"Could not save survey output: {path}: {error}") from error

    def write_json(self, *, path: Path, values: dict) -> None:
        """Serialize a record before atomic writing; invalid JSON replaces nothing."""
        text = json.dumps(values, indent=2, allow_nan=False) + "\n"
        self.write_text(path=path, text=text)


class TableInputFiles(SurveyOutputDirectory):
    """Name and reuse one snapshot's records in a guarded survey directory."""

    def __init__(
        self,
        *,
        directory: Path,
        protected_roots: tuple[Path, ...],
        allowed_output_root: Path | None = None,
    ):
        super().__init__(
            directory=directory,
            protected_roots=protected_roots,
            allowed_output_root=allowed_output_root,
        )
        self.statistics_path = self.directory / "input_ranges.json"
        self.report_path = self.directory / "table_range_report.json"
        self.check_output_path(path=self.statistics_path)
        self.check_output_path(path=self.report_path)

    def read_completed_statistics(
        self,
        *,
        dataset: Path,
        source_state: list[dict],
        xy_region: dict | None,
    ) -> dict | None:
        """Reuse a complete scan only for the same source and physical definitions.

        Return None when no scan exists or it was partial. Changed source,
        selection or definitions require an explicit fresh output directory,
        so results from different inputs cannot be mixed silently.
        Tables are deliberately absent: new axes reuse the same physical ranges.
        """
        if not self.statistics_path.exists():
            return None
        self.check_output_path(path=self.statistics_path)
        with self.statistics_path.open("r") as saved:
            statistics = json.load(saved)
        requested_region = None
        if xy_region is not None:
            requested_region = {
                axis: list(bounds) if bounds is not None else None
                for axis, bounds in xy_region.items()
            }
        if (
            statistics["source"] != str(dataset.resolve())
            or statistics["source_file_state"] != source_state
            or statistics["definitions"] != input_definitions()
            or statistics["requested_xy_region"] != requested_region
        ):
            raise StaleInputStatistics("Saved input ranges use different inputs/rules; choose a new output directory")
        if statistics["scan_status"] != "complete":
            return None
        return statistics
