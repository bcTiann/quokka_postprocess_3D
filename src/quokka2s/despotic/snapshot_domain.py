"""Check DESPOTIC's recorded physical conventions and table-axis bounds."""


# Coordinate order in table metadata and snapshot-domain measurements.
AXIS_NAMES = ("nH", "NH", "dVdr")


def validate_table_domain(table, cfg) -> dict:
    """Check recorded physical settings and return the table's reference domain.

    table is a DespoticTable from load_table(). Its recorded snapshot shape and
    cell count describe the reference used to choose the axes; they do not
    restrict the resolution of a new snapshot. This function needs no snapshot.
    cfg supplies X_H, COLUMN_DENSITY_MEAN and COLUMN_DENSITY_DIRECTIONS from
    physics.settings. Require the recorded axis endpoints to match the table.
    Actual cell-coordinate coverage is checked later during table queries.
    Example: a table built using a 256x256x2048 snapshot can be loaded when
    processing a 512x512x4096 snapshot; its recorded shape stays unchanged.
    """
    domain = (table.build_metadata or {}).get("snapshot_domain")
    if not domain or domain.get("selection") != "all simulation cells":
        raise ValueError("DESPOTIC table lacks all-cell snapshot-domain metadata")

    expected_settings = {
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
