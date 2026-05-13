"""Build the sphere-geometry counterpart of the LVG DESPOTIC table.

Same architecture as ``build_table.py`` (3D `(nH, NH, dVdr)` axes,
self-consistent T from ``setChemEq(evolveTemp="iterateDust")``), but
line emission uses ``escapeProbGeom='sphere'``. Sphere τ does not
depend on dVdr, so each line field is computed once per (nH, NH) and
broadcast across the dVdr axis. Result: drop-in replacement for the
LVG table — pipeline lookup interface is unchanged.

Run:
    python -m quokka2s.tables.build_table_sphere
"""
from __future__ import annotations
from pathlib import Path
from . import LogGrid, build_table, save_table
from .builder import SpeciesSpec
from despotic.chemistry import NL99, NL99_GC, GOW
from ..pipeline.prep import config as cfg

N_H_RANGE     = (1e-4,  1e6)
COL_DEN_RANGE = (1e15,  1e24)
DVDR_RANGE    = (1e-19, 1e-12)
points = 35

nH_grid   = LogGrid(*N_H_RANGE,     num_points=points)
col_grid  = LogGrid(*COL_DEN_RANGE, num_points=points)
dVdr_grid = LogGrid(*DVDR_RANGE,    num_points=points)

SPECIES_SPECS = (
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("O", True),
    SpeciesSpec("e-", False),
    SpeciesSpec("H+", False),
    SpeciesSpec("H2", False),
    SpeciesSpec("H", False),
)


table = build_table(
    nH_grid,
    col_grid,
    dVdr_grid,
    species_specs=SPECIES_SPECS,
    show_progress=True,
    chem_network=GOW,
    full_parallel=False,
    workers=-1,
    escape_geom='sphere',
)
table_path = Path(cfg.DESPOTIC_TABLE_PATH_SPHERE)
table_path.parent.mkdir(parents=True, exist_ok=True)
save_table(table, table_path)
print(f"Saved sphere table to {table_path}")
