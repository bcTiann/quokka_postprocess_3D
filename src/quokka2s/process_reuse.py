"""Verify completed numerical results before skipping an expensive calculation."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from quokka2s.emission_results import read_emission_results
from quokka2s.file_provenance import file_sha256
from quokka2s.run_settings import ProcessSettings


def snapshot_file_state(dataset: Path) -> dict:
    """Stat snapshot files and linked payloads without reading their contents.

    AMReX Level_* directories may be symlinks. Follow their files while recording
    link destinations; visit each resolved directory once to stop cycles.
    """
    files = []
    directory_links = []
    pending = [dataset]
    visited = set()
    while pending:
        directory = pending.pop()
        resolved = directory.resolve(strict=True)
        if directory.is_symlink():
            directory_links.append({
                "path": str(directory.relative_to(dataset)),
                "target": str(resolved),
            })
        if resolved in visited:
            continue
        visited.add(resolved)
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                state = path.stat()
                files.append({
                    "path": str(path.relative_to(dataset)),
                    "resolved_path": str(path.resolve(strict=True)),
                    "size_bytes": state.st_size,
                    "modified_ns": state.st_mtime_ns,
                })
            elif path.is_symlink():
                raise FileNotFoundError(f"Broken snapshot file link: {path}")
    return {
        "files": sorted(files, key=lambda record: record["path"]),
        "directory_links": sorted(directory_links, key=lambda record: record["path"]),
    }


def processing_input_identity(config: ProcessSettings) -> dict:
    """Record source-file metadata and hashes of tables and calculation sources.

    Snapshot payloads are only stat'ed, never hashed or loaded. File sizes and
    nanosecond modification times detect ordinary source updates. Table and
    Python/Cython source hashes also detect replacements at the same paths.
    These records govern result reuse, not physical table compatibility.
    """
    if not (config.dataset / "Header").is_file():
        raise FileNotFoundError(f"Missing snapshot Header: {config.dataset / 'Header'}")
    package_directory = Path(__file__).parent
    source_hashes = {
        str(path.relative_to(package_directory)): file_sha256(path)
        for path in sorted(package_directory.rglob("*"))
        if path.suffix in (".py", ".pyx")
    }
    table_hashes = {
        name: file_sha256(getattr(config, name))
        for name in ("despotic_table", "cloudy_table", "dust_opacity_table")
    }
    return {
        "snapshot_files": snapshot_file_state(dataset=config.dataset),
        "table_sha256": table_hashes,
        "source_sha256": source_hashes,
    }


def reuse_completed_results(config: ProcessSettings) -> bool:
    """Return False for new output; reuse only intact, complete, matching output.

    Existing partial, failed, legacy or conflicting results raise an error and
    remain untouched. Explicit regions can be complete without full_snapshot.
    This is whole-snapshot reuse, never a slab checkpoint or an overwrite.
    """
    directory = config.output_dir
    if not directory.exists():
        return False
    advice = (
        f"Existing results were preserved: {directory}. "
        "Use plot to redraw saved products, or choose another results_dir for a new calculation."
    )
    try:
        status = json.loads((directory / "status.json").read_text())
        report = json.loads((directory / "emission_report.json").read_text())
        for record in (status, report):
            if record.get("status") != "completed" or record.get("processing_complete") is not True:
                raise ValueError("numerical processing is not complete")
        configuration = {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(config).items()
        }
        # JSON stores native-index tuples as lists in run_configuration.
        configuration = json.loads(json.dumps(configuration))
        if report.get("run_configuration") != configuration:
            raise ValueError("processing configuration differs from the saved run")
        if report.get("input_identity") != processing_input_identity(config):
            raise ValueError("input/source identity is missing or differs from the saved run")
        for name in ("images.npz", "spectra.npz", "phase_velocity.npz"):
            with ZipFile(directory / name) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"corrupt numerical product: {name}")
        products = read_emission_results(directory=directory)
        if not products.processing_complete:
            raise ValueError("saved image products are incomplete")
    except Exception as exc:
        raise ValueError(f"Cannot reuse results: {exc}. {advice}") from exc
    return True
