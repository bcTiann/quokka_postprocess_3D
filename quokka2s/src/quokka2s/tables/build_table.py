from __future__ import annotations
from pathlib import Path
from quokka2s.tables import LogGrid, build_table, save_table
from quokka2s.tables.builder import SpeciesSpec
from despotic.chemistry import NL99, NL99_GC
from quokka2s.pipeline.prep import config as cfg

N_H_RANGE = (1e-4,  1e6)
COL_DEN_RANGE = (1e15, 1e24)
T_RANGE = (2.0, 5e6)
points = 25

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
    chem_network=NL99_GC, 
    full_parallel=False,
    workers=-1,
)
table_path = Path(cfg.DESPOTIC_TABLE_PATH)
table_path.parent.mkdir(parents=True, exist_ok=True)
save_table(table, table_path)
