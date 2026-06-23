"""Public entry points for DESPOTIC table utilities.

The 4D (nH, N_H, dVdr, μγ) table surface — DespoticTable4D / TableLookup4D /
build_table_4d / save_table_4d / load_table_4d — was deprecated 2026-06-23:
the pipeline uses only the 3D table and no task references any 4D table.  Those
defs remain wrapped in-tree (see the DEPRECATED banners in models.py / io.py /
lookup.py / builder.py and the standalone build_table_4d.py).  To revive, re-add
them to the imports + __all__ here and un-wrap the defs.
"""

from .models import (
    LogGrid,
    LineLumResult,
    SpeciesLineGrid,
    AttemptRecord,
    DespoticTable,
)
from .builder import build_table, plot_table
from .io import load_table, save_table
from .diagnostics import plot_failure_overlay, summarize_failures, plot_sampling_histogram
from .lookup import TableLookup
from .plotting import plot_table_overview

__all__ = [
    "LogGrid",
    "LineLumResult",
    "SpeciesLineGrid",
    "AttemptRecord",
    "DespoticTable",
    "TableLookup",
    "build_table",
    "save_table",
    "load_table",
    "plot_table",
    "plot_failure_overlay",
    "summarize_failures",
    "plot_sampling_histogram",
    "plot_table_overview",
]
