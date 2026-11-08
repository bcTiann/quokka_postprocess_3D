"""Load the DESPOTIC fields needed by snapshot processing from the saved NPZ.

Temperature, three line luminosities and three abundances retain their table
values and independent NaN support. Solver attempts, energy terms and unused
line quantities stay on disk. Table builders and diagnostics use the full
table loader in table_files; this module supplies only the processing subset.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.despotic.snapshot_domain import validate_table_domain
from quokka2s.despotic.table_files import SUPPORTED_TABLE_VERSIONS, _scalar_string
from quokka2s.physics import settings
from quokka2s.physics.composition import ABUNDANCE_SETUP, reject_superseded_composition


ABUNDANCE_SPECIES = frozenset(('e-', 'H+', 'H'))
LINE_SPECIES = frozenset(('CO', 'CO21', 'C+'))
REQUIRED_SPECIES = ABUNDANCE_SPECIES | LINE_SPECIES


@dataclass(frozen=True)
class RuntimeLineGrid:
    """Luminosity per H [erg/s/H], shape (N_nH, N_NH, N_gradient)."""

    lumPerH: np.ndarray


@dataclass(frozen=True)
class RuntimeSpecies:
    """A queried line or H/e abundance; unqueried quantities are absent."""

    name: str
    abundance: np.ndarray | None
    line: RuntimeLineGrid | None
    is_emitter: bool


@dataclass(frozen=True)
class RuntimeDespoticTable:
    """Seven process fields and the axes/build settings that define them.

    Field shapes are (N_nH, N_NH, N_gradient). Axes have units cm^-3,
    cm^-2 and s^-1; tg_final is in K. Build settings retain the source
    composition and table-domain metadata used by compatibility checks.
    """

    species_data: MappingProxyType
    tg_final: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    dVdr_values: np.ndarray
    build_metadata: MappingProxyType | None
    chemistry_network: str
    escape_geometry: str
    temperature_mode: str


class RuntimeDespoticLookup(DespoticLookup):
    """Use the shared interpolation/query code with seven registered fields."""

    def _register_thermal_fields(self):
        self._register_field(token='tg_final', values=self.table.tg_final)

    def _register_species_fields(self):
        for name, record in self._species_records.items():
            if record.abundance is not None:
                self._register_field(
                    token=f'species:{name}:abundance',
                    values=record.abundance,
                )
            if record.line is not None:
                self._register_field(
                    token=f'species:{name}:line:lumPerH',
                    values=record.line.lumPerH,
                )

    def _register_energy_fields(self):
        # Snapshot products do not query solver heating/cooling grids.
        return None


def load_runtime_despotic_table(path: str | Path) -> RuntimeDespoticTable:
    """Read required fields and compatibility metadata from the source NPZ."""
    with np.load(Path(path), allow_pickle=True) as source:
        version = int(np.asarray(source['version']).flat[0])
        if version not in SUPPORTED_TABLE_VERSIONS:
            raise ValueError(
                f'Unsupported DESPOTIC table version {version}; '
                f'supported: {SUPPORTED_TABLE_VERSIONS}'
            )
        species_names = [str(value) for value in source['species_names']]
        emitter_flags = np.asarray(source['species_is_emitter'], dtype=bool)
        species = {}
        for name, is_emitter in zip(species_names, emitter_flags):
            if name not in REQUIRED_SPECIES:
                continue
            abundance = None
            if name in ABUNDANCE_SPECIES:
                abundance = np.asarray(source[f'{name}_abundance'], dtype=float)
            line = None
            if name in LINE_SPECIES and is_emitter:
                line = RuntimeLineGrid(
                    lumPerH=np.asarray(source[f'{name}_lumPerH'], dtype=float),
                )
            species[name] = RuntimeSpecies(
                name=name,
                abundance=abundance,
                line=line,
                is_emitter=bool(is_emitter),
            )
        build_metadata = None
        if 'build_metadata_json' in source.files:
            build_metadata = json.loads(
                str(np.asarray(source['build_metadata_json']).item()),
            )
        if build_metadata:
            reject_superseded_composition(
                build_metadata.get('composition', {}).get('setup'),
            )
        table = RuntimeDespoticTable(
            species_data=MappingProxyType(species),
            tg_final=np.asarray(source['tg_final'], dtype=float),
            nH_values=np.asarray(source['nH_values'], dtype=float),
            col_density_values=np.asarray(source['col_density_values'], dtype=float),
            dVdr_values=np.asarray(source['dVdr_values'], dtype=float),
            build_metadata=MappingProxyType(build_metadata) if build_metadata is not None else None,
            chemistry_network=_scalar_string(source, 'chemistry_network', 'GOW'),
            escape_geometry=_scalar_string(source, 'escape_geometry', 'LVG'),
            temperature_mode=_scalar_string(source, 'temperature_mode', 'iterateDust'),
        )
    expected_shape = table.tg_final.shape
    for record in table.species_data.values():
        if record.abundance is not None and record.abundance.shape != expected_shape:
            raise ValueError('DESPOTIC abundance shape differs from temperature grid')
        if record.line is not None and record.line.lumPerH.shape != expected_shape:
            raise ValueError('DESPOTIC line shape differs from temperature grid')
    return table


def load_runtime_despotic_lookup(table_path: str | Path) -> RuntimeDespoticLookup:
    """Check source composition/domain, then build the process interpolators."""
    table = load_runtime_despotic_table(table_path)
    if table.build_metadata['composition']['setup'] != ABUNDANCE_SETUP:
        raise ValueError('Expected native default DESPOTIC abundances')
    validate_table_domain(table=table, cfg=settings)
    return RuntimeDespoticLookup(table)
