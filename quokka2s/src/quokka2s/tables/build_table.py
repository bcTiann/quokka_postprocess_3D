from __future__ import annotations
from pathlib import Path
from . import LogGrid, build_table, save_table
from .builder import SpeciesSpec
from despotic.chemistry import NL99, NL99_GC, GOW
from ..pipeline.prep import config as cfg

N_H_RANGE = (1e-4,  1e6)
COL_DEN_RANGE = (1e15, 1e24)
T_RANGE = (1.0, 2e5)
points = 35

nH_grid = LogGrid(*N_H_RANGE, num_points=points)
col_grid = LogGrid(*COL_DEN_RANGE, num_points=points)
T_grid = LogGrid(*T_RANGE, num_points=points)

# tg_guesses = [1000.0, ]
# SPECIES = ('CO', 'C+', "C", 'HCO+', 'O')
# ABUNDANCES = ('H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C', 
#               'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-')



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
    T_grid, 
    species_specs=SPECIES_SPECS, 
    show_progress=True, 
    chem_network=GOW, 
    full_parallel=False,
    workers=-1,
)
table_path = Path(cfg.DESPOTIC_TABLE_PATH)
table_path.parent.mkdir(parents=True, exist_ok=True)
save_table(table, table_path)
