"""Load run inputs: a native snapshot, two emission tables, and dust opacity."""
from __future__ import annotations

import numpy as np

from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.despotic.cell_fields import DespoticCellReader
from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.physics.dust_attenuation import (
    extinction_cross_sections,
    load_draine_extinction,
)
from quokka2s.snapshot_reader import Snapshot
from quokka2s.despotic.table_files import load_table
from quokka2s.despotic.lookup import DespoticLookup

AXIS_NAMES = ('nH', 'NH', 'dVdr')


def validate_snapshot_domain(table, shape, cfg):
    """Check DESPOTIC metadata against the snapshot's physical setup.

    table is the loaded DespoticTable, shape is its target (Nx, Ny, Nz), and cfg
    is physics.settings. Reject incompatible geometry, hydrogen/column
    settings, or table axis endpoints; source-file identity is not required."""
    domain = (table.build_metadata or {}).get('snapshot_domain')
    if not domain or domain.get('selection') != 'all simulation cells':
        raise ValueError('DESPOTIC table lacks all-cell snapshot-domain metadata')
    checks = {
        'shape': list(shape), 'total_cells': int(np.prod(shape)),
        'X_H': float(cfg.X_H), 'column_mean': cfg.COLUMN_DENSITY_MEAN,
        'column_directions': cfg.COLUMN_DENSITY_DIRECTIONS,
    }
    for name, expected in checks.items():
        if domain.get(name) != expected:
            raise ValueError(f'DESPOTIC snapshot settings mismatch for {name}: '
                             f'table={domain.get(name)!r}, current={expected!r}')
    axes = (table.nH_values, table.col_density_values, table.dVdr_values)
    for name, axis in zip(AXIS_NAMES, axes):
        recorded = domain['axes'][name]
        if axis[0] != recorded['minimum'] or axis[-1] != recorded['maximum']:
            raise ValueError(f'Candidate {name} bounds differ from recorded snapshot extrema')


def load_processing_inputs(config):
    """Open configured inputs, then create the new output directory.

    config contains the snapshot, table, dust-opacity, and output Paths from
    load_process_config(). Return Snapshot and the shared CellEmissionCalculator;
    cell arrays are loaded later by read_slab()."""
    check_input_paths(config)
    snapshot = open_snapshot(
        dataset_path=config.dataset,
        xy_region={
            "x": config.x_index_range,
            "y": config.y_index_range,
        },
    )
    emission_calculator = load_emission_calculator(config=config, snapshot=snapshot)

    # Create the output directory only after every input has loaded successfully.
    config.output_dir.mkdir(parents=True, exist_ok=False)
    return snapshot, emission_calculator


def check_input_paths(config):
    """Require available input paths, a snapshot Header, and a new output path."""
    if config.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    for name in ('dataset', 'despotic_table', 'cloudy_table', 'dust_opacity_table'):
        if not getattr(config, name).exists():
            raise FileNotFoundError(f'{name}: {getattr(config, name)}')
    snapshot_header = config.dataset / 'Header'
    if not snapshot_header.is_file():
        raise FileNotFoundError(f'Snapshot Header: {snapshot_header}')


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


def load_emission_calculator(config, snapshot: Snapshot) -> CellEmissionCalculator:
    """Load shared readers, choose line order, and prepare dust cross-sections.

    config supplies table Paths; snapshot supplies the native grid geometry.
    Return a calculator holding two readers, line names, and named cross-sections
    [cm^2/H]. No cell queries run here."""
    despotic = load_despotic_lookup(
        table_path=config.despotic_table,
        snapshot=snapshot,
    )
    cloudy = CloudyLookup(
        path=config.cloudy_table,
    )
    line_keys = tuple(cloudy.line_keys) + ('co10', 'co21')
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
    return CellEmissionCalculator(
        despotic_reader=despotic_reader,
        cloudy_reader=cloudy_reader,
        line_keys=line_keys,
        dust_cross_section_cm2_H=dust_cross_section_cm2_H,
    )


def load_despotic_lookup(table_path, snapshot: Snapshot) -> DespoticLookup:
    """Load DESPOTIC and check its composition and snapshot-domain metadata.

    table_path is the configured Path; snapshot supplies its target geometry.
    Remaining table NaNs are handled later by DespoticCellReader.read_fields()."""
    from quokka2s.physics import settings as cfg

    table = load_table(table_path)
    if table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected native default DESPOTIC abundances')
    validate_snapshot_domain(
        table=table,
        shape=snapshot.shape,
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
