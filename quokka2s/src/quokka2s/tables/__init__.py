"""Public entry points for DESPOTIC table utilities."""

from .models import (
    LogGrid,
    LineLumResult,
    SpeciesLineGrid,
    AttemptRecord,
    DespoticTable,
    DespoticTable4D,
)
from .builder import build_table, build_table_4d, plot_table
from .io import load_table, save_table, load_table_4d, save_table_4d
from .diagnostics import plot_failure_overlay, summarize_failures, plot_sampling_histogram
from .lookup import TableLookup, TableLookup4D
from .plotting import plot_table_overview

__all__ = [
    "LogGrid",
    "LineLumResult",
    "SpeciesLineGrid",
    "AttemptRecord",
    "DespoticTable",
    "DespoticTable4D",
    "TableLookup",
    "TableLookup4D",
    "build_table",
    "build_table_4d",
    "save_table",
    "load_table",
    "save_table_4d",
    "load_table_4d",
    "plot_table",
    "plot_failure_overlay",
    "summarize_failures",
    "plot_sampling_histogram",
    "plot_table_overview",
]
