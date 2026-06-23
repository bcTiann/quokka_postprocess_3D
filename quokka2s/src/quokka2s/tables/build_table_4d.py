# ============================================================================
# DEPRECATED 2026-06-23 — kept in-tree for reference (wrap-don't-delete).
# 4D (nH,NH,dVdr,μγ) DESPOTIC table path was retired in cache schema v5
# (2026-06-13); no pipeline task references any 4D table.  To revive:
# restore the 4D exports in tables/__init__.py and un-wrap these defs.
# ============================================================================
r'''
from __future__ import annotations
from pathlib import Path
from . import LogGrid, build_table_4d, save_table_4d
from .builder import SpeciesSpec
from despotic.chemistry import NL99, NL99_GC, GOW
from ..pipeline.prep import config as cfg

# Same (nH, N_H, dVdr) ranges as the 3D iterateDust table, plus a T axis.
# T grid matches the legacy fixed-T table (1 K – 2e7 K, 35 log points).
N_H_RANGE     = (1e-4,  1e6)
COL_DEN_RANGE = (1e15,  1e24)
DVDR_RANGE    = (1e-19, 1e-12)
T_RANGE       = (1.0,   2e7)
points = 35

nH_grid   = LogGrid(*N_H_RANGE,     num_points=points)
col_grid  = LogGrid(*COL_DEN_RANGE, num_points=points)
dVdr_grid = LogGrid(*DVDR_RANGE,    num_points=points)
T_grid    = LogGrid(*T_RANGE,       num_points=points)

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

if __name__ == "__main__":
    table = build_table_4d(
        nH_grid,
        col_grid,
        dVdr_grid,
        T_grid,
        species_specs=SPECIES_SPECS,
        show_progress=True,
        chem_network=GOW,
        workers=-1,
    )
    table_path = Path(cfg.DESPOTIC_TABLE_4D_PATH)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    save_table_4d(table, table_path)
    print(f"[build_table_4d] saved → {table_path}")
'''
