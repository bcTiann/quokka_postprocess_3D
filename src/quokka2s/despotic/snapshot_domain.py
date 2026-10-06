"""Match DESPOTIC's recorded table domain to the simulation's physical setup."""

import numpy as np


# Coordinate order in table metadata and snapshot-domain measurements.
AXIS_NAMES = ("nH", "NH", "dVdr")


def validate_snapshot_domain(table, shape, cfg) -> dict:
    """Return the recorded domain after checking its geometry and conventions.

    table is a DespoticTable from load_table(). shape is the original snapshot
    shape, e.g. (256, 256, 2048), even when process selects a smaller x-y region.
    cfg supplies X_H, COLUMN_DENSITY_MEAN and COLUMN_DENSITY_DIRECTIONS from
    physics.settings. Require the recorded axis endpoints to match the table.
    Raw-table failure-mask checks belong to the coverage tool, not this check.
    """
    domain = (table.build_metadata or {}).get("snapshot_domain")
    if not domain or domain.get("selection") != "all simulation cells":
        raise ValueError("DESPOTIC table lacks all-cell snapshot-domain metadata")

    expected_settings = {
        "shape": list(shape),
        "total_cells": int(np.prod(shape)),
        "X_H": float(cfg.X_H),
        "column_mean": cfg.COLUMN_DENSITY_MEAN,
        "column_directions": cfg.COLUMN_DENSITY_DIRECTIONS,
    }
    for name, expected in expected_settings.items():
        if domain.get(name) != expected:
            raise ValueError(
                f"DESPOTIC snapshot settings mismatch for {name}: "
                f"table={domain.get(name)!r}, current={expected!r}"
            )

    axes = (table.nH_values, table.col_density_values, table.dVdr_values)
    for name, axis in zip(AXIS_NAMES, axes):
        recorded = domain["axes"][name]
        if axis[0] != recorded["minimum"] or axis[-1] != recorded["maximum"]:
            raise ValueError(f"Candidate {name} bounds differ from recorded snapshot extrema")
    return domain
