from __future__ import annotations
import os
import time
from pathlib import Path
from . import LogGrid, build_table, save_table
from .builder import SpeciesSpec
from despotic.chemistry import NL99, NL99_GC, GOW
from ..pipeline.prep import config as cfg

# --- Grid (same for all networks) ---
N_H_RANGE     = (1e-4,  1e6)
COL_DEN_RANGE = (1e15,  1e24)
DVDR_RANGE    = (1e-19, 1e-12)   # s^-1; calibrated from plt263168 |∇·v|/3 (P0.01–P99.99)
points = 35

nH_grid   = LogGrid(*N_H_RANGE,     num_points=points)
col_grid  = LogGrid(*COL_DEN_RANGE, num_points=points)
dVdr_grid = LogGrid(*DVDR_RANGE,    num_points=points)

# --- Per-network species lists ---
# Emitters (need LAMDA data): CO, C, C+, HCO+, O — supported by all three networks.
# Abundance-only: intersection of the desired set {e-, H+, H2, H} with each
# network's specListExtended.
#   GOW     : has H+ and atomic H → all 4
#   NL99    : no H+, no atomic H  → e-, H2 only
#   NL99_GC : has H+ and atomic H → all 4 (same as GOW)
_EMITTERS = (
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("O", True),
)
_GOW_SPECS = _EMITTERS + (
    SpeciesSpec("e-", False),
    SpeciesSpec("H+", False),
    SpeciesSpec("H2", False),
    SpeciesSpec("H", False),
)
_NL99_SPECS = _EMITTERS + (
    SpeciesSpec("e-", False),
    SpeciesSpec("H2", False),
)
_NL99_GC_SPECS = _GOW_SPECS

_NETWORK_REGISTRY = {
    'GOW':     (GOW,     _GOW_SPECS),
    'NL99':    (NL99,    _NL99_SPECS),
    'NL99_GC': (NL99_GC, _NL99_GC_SPECS),
}

# --- Selection via env (default = current pipeline behaviour) ---
#   TABLE_NETWORK = GOW (default) | NL99 | NL99_GC
#   TABLE_OUT     = full path to despotic_table.npz (default cfg.DESPOTIC_TABLE_PATH)
_NETWORK_NAME = os.environ.get('TABLE_NETWORK', 'GOW')
if _NETWORK_NAME not in _NETWORK_REGISTRY:
    raise ValueError(f"Unknown TABLE_NETWORK={_NETWORK_NAME!r}; "
                     f"valid: {list(_NETWORK_REGISTRY)}")
_CHEM_NETWORK, SPECIES_SPECS = _NETWORK_REGISTRY[_NETWORK_NAME]
_OUT = os.environ.get('TABLE_OUT', cfg.DESPOTIC_TABLE_PATH)

_spec_summary = ', '.join(s.name + ('(em)' if s.is_emitter else '') for s in SPECIES_SPECS)
print(f'[build_table] network = {_NETWORK_NAME}')
print(f'[build_table] species = {_spec_summary}')
print(f'[build_table] grid    = {points}^3  '
      f'(nH 1e-4..1e6, NH 1e15..1e24, dVdr 1e-19..1e-12)')
print(f'[build_table] output  = {_OUT}')

# --- Build ---
_t0 = time.time()
table = build_table(
    nH_grid,
    col_grid,
    dVdr_grid,
    species_specs=SPECIES_SPECS,
    show_progress=True,
    chem_network=_CHEM_NETWORK,
    full_parallel=False,
    workers=-1,
)
_t1 = time.time()

table_path = Path(_OUT)
table_path.parent.mkdir(parents=True, exist_ok=True)
save_table(table, table_path)

# --- README sidecar (metadata next to the npz) ---
readme = table_path.parent / 'README.txt'
with open(readme, 'w') as f:
    f.write('DESPOTIC 3D table\n')
    f.write('=================\n')
    f.write(f'network         : {_NETWORK_NAME}\n')
    f.write(f'escape geometry : LVG\n')
    f.write(f'evolveTemp      : iterateDust\n')
    f.write(f'grid            : nH 1e-4..1e6, NH 1e15..1e24, dVdr 1e-19..1e-12, '
            f'{points}^3\n')
    f.write(f'species         : {_spec_summary}\n')
    f.write(f'build time      : {(_t1 - _t0) / 3600:.2f} h ({_t1 - _t0:.0f} s)\n')
    f.write(f'completed at    : '
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_t1))}\n')
    f.write(f'output file     : {table_path}\n')
print(f'[build_table] saved -> {table_path}  '
      f'({(_t1 - _t0) / 3600:.2f}h)  README sidecar -> {readme}')
