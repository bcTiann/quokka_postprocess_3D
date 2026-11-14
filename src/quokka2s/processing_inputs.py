"""Load run inputs: a native snapshot, two emission tables, and dust opacity."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.despotic.cell_fields import DespoticCellReader
from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.physics.composition import ABUNDANCE_SETUP
from quokka2s.physics.line_emissivity import CO_LINE_KEYS
from quokka2s.physics.dust_attenuation import (
    extinction_cross_sections,
    load_draine_extinction,
)
from quokka2s.snapshot_reader import Snapshot
from quokka2s.despotic.table_files import load_table
from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.despotic.snapshot_domain import validate_table_domain
from quokka2s.z_contiguous_cache import prepare_z_contiguous_cache, z_contiguous_cache_identity


def load_processing_inputs(config):
    """Open configured inputs, then create the new output directory.

    config contains the snapshot, table, dust-opacity, and output Paths from
    load_process_config(). Return Snapshot and the shared CellEmissionCalculator.
    This step reads metadata and emission tables, not snapshot cell fields.
    Native columns later load full-z tiles; the explicit z-contiguous source
    later loads its slabs through read_slab().
    """
    print('Inputs: checking snapshot and table paths...', flush=True)
    check_input_paths(config)
    region = {"x": config.x_index_range, "y": config.y_index_range}
    sidecar = None if config.prepared_z_fields is None else config.prepared_z_fields / 'snapshot_metadata.json'
    if sidecar is not None and sidecar.exists():
        from .snapshot_metadata import open_snapshot_metadata
        print('Inputs: restoring validated snapshot metadata; no yt load...', flush=True)
        snapshot = open_snapshot_metadata(
            dataset_path=config.dataset,
            metadata_path=sidecar,
            xy_region=region,
        )
    else:
        print('Inputs: opening snapshot metadata with yt...', flush=True)
        snapshot = open_snapshot(dataset_path=config.dataset, xy_region=region)
    print_snapshot_summary(snapshot=snapshot)
    emission_calculator = load_emission_calculator(config=config)

    # Create the output directory only after every input has loaded successfully.
    print(f'Inputs: creating numerical output directory: {config.output_dir}', flush=True)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    print('Inputs ready.', flush=True)
    return snapshot, emission_calculator


@contextmanager
def prepare_z_contiguous_snapshot(
    *, snapshot: Snapshot, dataset_path: Path, work_directory: Path, source_identity: dict,
    prepared_directory: Path | None = None, reader_threads: int = 8,
    read_workers: int = 5,
):
    """Prepare/reuse z fields, then own one bounded reader until processing exits.

    Snapshot and native metadata are initialized by the timed process. Only
    prepare_z_contiguous_cache()'s created conversion cost may be excluded.
    Reader construction, unit factors and existing-cache checks remain timed.
    """
    from .bounded_z_reader import BoundedEagerReader
    native_reader = snapshot.native_reader
    if prepared_directory is None:
        storage = TemporaryDirectory(prefix='.zfields-', dir=work_directory)
    else:
        prepared_directory = Path(prepared_directory).resolve(strict=True)
        storage = nullcontext(prepared_directory)
    with storage as directory:
        cache_directory = Path(directory) / 'fields' if prepared_directory is None else Path(directory)
        preparation = prepare_z_contiguous_cache(
            native_reader=native_reader,
            cache_directory=cache_directory,
            source_identity=source_identity,
        )
        if prepared_directory is not None and preparation['created']:
            raise AssertionError('An explicitly prepared cache cannot be converted during process')
        metadata_write = None
        if preparation['created']:
            from .snapshot_metadata import write_snapshot_metadata
            metadata_write = write_snapshot_metadata(
                snapshot=snapshot,
                dataset_path=dataset_path,
                metadata_path=cache_directory / 'snapshot_metadata.json',
                source_identity=source_identity,
            )
            preparation['identity'] = z_contiguous_cache_identity(
                cache_directory=cache_directory,
                source_identity=source_identity,
            )
        preparation['metadata_sidecar_write'] = metadata_write
        initialization_began = time.monotonic()
        reader = BoundedEagerReader(
            native_reader,
            cache_directory=preparation['directory'],
            reader_threads=reader_threads,
            read_workers=read_workers,
        )
        preparation['reader_initialization_seconds'] = time.monotonic() - initialization_began
        preparation['conversion_seconds'] = preparation['preprocessing_seconds'] if preparation['created'] else 0.0
        preparation['cache_reused'] = not preparation['created']
        preparation['reader_statistics'] = reader.read_statistics
        preparation['derivation_statistics'] = reader.derivation_statistics
        try:
            yield replace(snapshot, field_reader=reader), preparation
        finally:
            reader.close()


@contextmanager
def prepare_processing_snapshot(
    *, snapshot: Snapshot, dataset_path: Path, work_directory: Path,
    source_identity: dict, field_source: str = "native-columns",
    prepared_directory: Path | None = None, reader_threads: int = 8,
    read_workers: int = 5,
):
    """Own the explicitly selected field reader; never mix the two sources.

    The native-column reader retains metadata and small immutable neighbour
    faces between tiles. Each NumPy field read closes its file.
    Snapshot.native_reader remains the original metadata reader, so unit and
    global geometry access is intact.
    No native-column initialization/read cost is excluded from process timing.
    """
    if field_source == "z-contiguous":
        with prepare_z_contiguous_snapshot(
            snapshot=snapshot,
            dataset_path=dataset_path,
            work_directory=work_directory,
            source_identity=source_identity,
            prepared_directory=prepared_directory,
            reader_threads=reader_threads,
            read_workers=read_workers,
        ) as prepared:
            yield prepared
        return
    if field_source != "native-columns":
        raise ValueError("Unsupported processing field source: " + str(field_source))
    if prepared_directory is not None:
        raise ValueError("native-columns cannot use prepared_z_fields")
    from .native_column_reader import NativeColumnReader

    initialized = time.monotonic()
    with NativeColumnReader(snapshot.native_reader, read_workers=read_workers) as reader:
        initialization_seconds = time.monotonic() - initialized
        yield snapshot, {
            "native_column_reader": reader,
            "conversion_seconds": 0.0,
            "preprocessing_seconds": 0.0,
            "cache_reused": False,
            "reader_initialization_seconds": initialization_seconds,
        }


def print_snapshot_summary(snapshot: Snapshot) -> None:
    """Print native geometry and selected indices using snapshot metadata only.

    snapshot comes from open_snapshot(). Grid counts and x/y bounds are native
    cell indices; widths are converted from its unit-aware lengths to pc for
    display. This function reads no cell fields and changes no numerical arrays.
    Example: a 512x512x4096 box with x/y=64:66/32:34 selects 16,384 cells.
    """
    nx, ny, nz = snapshot.shape
    dx_pc, dy_pc, dz_pc = snapshot.cell_widths.to('pc').value
    x_start, x_stop = snapshot.xy_region['x']
    y_start, y_stop = snapshot.xy_region['y']
    print(
        f'Native grid: {nx} x {ny} x {nz} ({snapshot.cell_count:,} cells)',
        flush=True,
    )
    print(
        f'Cell widths [pc]: dx={dx_pc:.6g}, dy={dy_pc:.6g}, dz={dz_pc:.6g}',
        flush=True,
    )
    print(
        f'Selected region: x=[{x_start}, {x_stop}), y=[{y_start}, {y_stop}), '
        f'z=[0, {nz}); {snapshot.processing_cell_count:,} cells',
        flush=True,
    )


def check_input_paths(config):
    """Require available input paths, a snapshot Header, and a new output path."""
    if config.output_dir.exists():
        raise FileExistsError(
            f'Results already exist: {config.output_dir}. '
            'This selection has already been processed. '
            'Choose a different results_dir to keep another run.'
        )
    for name in ('dataset', 'despotic_table', 'cloudy_table', 'dust_opacity_table'):
        if not getattr(config, name).exists():
            raise FileNotFoundError(f'{name}: {getattr(config, name)}')
    snapshot_header = config.dataset / 'Header'
    if not snapshot_header.is_file():
        raise FileNotFoundError(
            'dataset must point to the snapshot directory containing Header '
            '(e.g. snapshots/plt0655228), not its parent. '
            f'Missing file: {snapshot_header}'
        )


def open_snapshot(dataset_path, xy_region=None) -> Snapshot:
    """Open a native uniform yt snapshot and retain its geometry.

    dataset_path is the configured Path. Cell fields are loaded later; Snapshot
    derives (Nx, Ny, Nz), unit-aware widths, and cell volume [cm^3] from yt.
    xy_region optionally selects native x/y index ranges, e.g.
    {"x": (64, 128), "y": (32, 96)}. Original geometry and full z are retained."""
    import yt

    ds = yt.load(str(dataset_path.resolve()))
    if ds.max_level != 0:
        raise ValueError('Expected the complete uniform snapshot at native resolution')
    return Snapshot(
        dataset=ds,
        xy_region=xy_region,
    )


def load_emission_calculator(config) -> CellEmissionCalculator:
    """Load shared readers, choose line order, and prepare dust cross-sections.

    config supplies the table Paths. Grid geometry belongs to Snapshot and
    the product accumulators, so loading the emission tables does not need it.
    Return a calculator holding two readers, line names, and named cross-sections
    [cm^2/H]. No cell queries run here."""
    from .despotic.runtime_table import load_runtime_despotic_lookup

    print(f'Tables: loading DESPOTIC: {config.despotic_table}', flush=True)
    despotic = load_runtime_despotic_lookup(
        table_path=config.despotic_table,
    )
    print(f'Tables: loading Cloudy: {config.cloudy_table}', flush=True)
    cloudy = CloudyLookup(
        path=config.cloudy_table,
    )
    line_keys = tuple(cloudy.line_keys) + CO_LINE_KEYS
    print(f'Tables: loading Draine dust opacity: {config.dust_opacity_table}', flush=True)
    dust_cross_section_cm2_H = prepare_line_dust_cross_sections(
        table_path=config.dust_opacity_table,
        line_keys=line_keys,
    )
    despotic_reader = DespoticCellReader(
        lookup=despotic,
    )
    cloudy_reader = CloudyCellReader(
        lookup=cloudy,
    )
    print(f'Emission setup ready: {len(line_keys)} lines.', flush=True)
    return CellEmissionCalculator(
        despotic_reader=despotic_reader,
        cloudy_reader=cloudy_reader,
        line_keys=line_keys,
        dust_cross_section_cm2_H=dust_cross_section_cm2_H,
    )


def load_despotic_lookup(table_path) -> DespoticLookup:
    """Load DESPOTIC and check its composition and recorded physical domain.

    table_path is the configured Path. The reference snapshot's mesh size is
    retained as build metadata; it does not restrict the new snapshot's size.
    Remaining table NaNs are handled later by DespoticCellReader.read_fields()."""
    from quokka2s.physics import settings as cfg

    table = load_table(table_path)
    if table.build_metadata['composition']['setup'] != ABUNDANCE_SETUP:
        raise ValueError('Expected native default DESPOTIC abundances')
    validate_table_domain(
        table=table,
        cfg=cfg,
    )
    return DespoticLookup(table)


def prepare_line_dust_cross_sections(table_path, line_keys) -> dict[str, float]:
    """Read Draine's table and return cross-sections keyed by line name.

    table_path is the configured Path; line_keys is the chosen output order.
    Values are [cm^2/H], e.g. result['halpha']; the HI cross-section is zero."""
    wavelength_micron, sigma_cm2_H = load_draine_extinction(table_path)
    cross_sections_cm2_H = extinction_cross_sections(
        line_keys=line_keys,
        wavelength_micron=wavelength_micron,
        sigma_cm2_H=sigma_cm2_H,
    )
    cross_section_by_line = {}
    for line_key, cross_section in zip(line_keys, cross_sections_cm2_H):
        cross_section_by_line[line_key] = float(cross_section)
    return cross_section_by_line
