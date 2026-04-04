from __future__ import annotations
from pathlib import Path
from quokka2s.tables import LogGrid, build_table, save_table
from quokka2s.tables.builder import SpeciesSpec
N_H_RANGE = (1e-5, 1e5)
COL_DEN_RANGE = (1e15, 1e24)
points = 25
nH_grid = LogGrid(*N_H_RANGE, num_points=points)
col_grid = LogGrid(*COL_DEN_RANGE, num_points=points)
tg_guesses = [10.0, ]
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
    # SpeciesSpec("H+", False),
)


table = build_table(nH_grid, col_grid, tg_guesses, species_specs=SPECIES_SPECS, show_progress=True, workers=-1)
output_dir = Path("output_tables_testNL99")
output_dir.mkdir(parents=True, exist_ok=True)
save_table(table, output_dir / "despotic_table.npz")


