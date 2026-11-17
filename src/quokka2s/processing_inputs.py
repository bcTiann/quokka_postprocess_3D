"""Load run inputs: a native snapshot, two emission tables, and dust opacity."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
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
from quokka2s.process_logging import quiet_yt_metadata


def load_processing_inputs(config, *, compact_log=False):
    """Open configured inputs, then create the new output directory.

    config contains the snapshot, table, dust-opacity, and output Paths from
    load_process_config(). Return Snapshot and the shared CellEmissionCalculator.
    This step reads metadata and emission tables, not snapshot cell fields.
    Native columns later load the selected full-z tiles directly from the snapshot.
    compact_log is used by process after its path header has been printed;
    independent figure tools retain their existing path and yt messages.
    """
    if compact_log:
        print('\nInputs', flush=True)
    prefix = '  ' if compact_log else 'Inputs: '
    print(f'{prefix}checking snapshot and table paths...', flush=True)
    check_input_paths(config)
    region = {"x": config.x_index_range, "y": config.y_index_range}
    metadata_message = 'opening snapshot metadata...' if compact_log else 'opening snapshot metadata with yt...'
    print(f'{prefix}{metadata_message}', flush=True)
    metadata_logs = quiet_yt_metadata() if compact_log else nullcontext()
    with metadata_logs:
        snapshot = open_snapshot(dataset_path=config.dataset, xy_region=region)
    print_snapshot_summary(snapshot=snapshot, compact_log=compact_log)
    emission_calculator = load_emission_calculator(config=config, compact_log=compact_log)

    # Create the output directory only after every input has loaded successfully.
    if compact_log:
        print('  creating numerical output directory...', flush=True)
    else:
        print(f'Inputs: creating numerical output directory: {config.output_dir}', flush=True)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    if compact_log:
        print('  ready', flush=True)
    else:
        print('Inputs ready.', flush=True)
    return snapshot, emission_calculator


@contextmanager
def prepare_processing_snapshot(*, snapshot: Snapshot, read_workers: int = 5):
    """Own the original-snapshot reader until this snapshot finishes.

    Read selected full-z XY tiles into bounded RAM. Global neighbouring faces
    remain available for gradients; no converted snapshot is written to disk.
    Initialization and every subsequent field read remain inside process time.
    """
    from .native_column_reader import NativeColumnReader

    initialized = time.monotonic()
    with NativeColumnReader(
        snapshot.native_reader,
        read_workers=read_workers,
        xy_region=snapshot.xy_region,
    ) as reader:
        initialization_seconds = time.monotonic() - initialized
        yield reader, initialization_seconds


def print_snapshot_summary(snapshot: Snapshot, *, compact_log=False) -> None:
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
    prefix = '  ' if compact_log else ''
    print(
        f'{prefix}Native grid: {nx} x {ny} x {nz} ({snapshot.cell_count:,} cells)',
        flush=True,
    )
    print(
        f'{prefix}Cell widths [pc]: dx={dx_pc:.6g}, dy={dy_pc:.6g}, dz={dz_pc:.6g}',
        flush=True,
    )
    print(
        f'{prefix}Selected region: x=[{x_start}, {x_stop}), y=[{y_start}, {y_stop}), '
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


def load_emission_calculator(config, *, compact_log=False) -> CellEmissionCalculator:
    """Load shared readers, choose line order, and prepare dust cross-sections.

    config supplies the table Paths. Grid geometry belongs to Snapshot and
    the product accumulators, so loading the emission tables does not need it.
    Return a calculator holding two readers, line names, and named cross-sections
    [cm^2/H]. No cell queries run here. compact_log omits paths already shown
    in process's header; standalone callers retain full path messages."""
    from .despotic.runtime_table import load_runtime_despotic_lookup

    if compact_log:
        print('  loading DESPOTIC...', flush=True)
    else:
        print(f'Tables: loading DESPOTIC: {config.despotic_table}', flush=True)
    despotic = load_runtime_despotic_lookup(
        table_path=config.despotic_table,
    )
    if compact_log:
        print('  loading Cloudy...', flush=True)
    else:
        print(f'Tables: loading Cloudy: {config.cloudy_table}', flush=True)
    cloudy = CloudyLookup(
        path=config.cloudy_table,
    )
    line_keys = tuple(cloudy.line_keys) + CO_LINE_KEYS
    if compact_log:
        print('  loading Draine dust opacity...', flush=True)
    else:
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
    if compact_log:
        print(f'  emission setup ready: {len(line_keys)} lines', flush=True)
    else:
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
