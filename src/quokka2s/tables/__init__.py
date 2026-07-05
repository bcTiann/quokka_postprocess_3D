"""Canonical 3D GOW/LVG DESPOTIC table API."""

from .builder import GOW_LVG_SPECIES, SpeciesSpec, build_gow_lvg_table
from .io import load_table, save_table
from .lookup import TableLookup
from .models import AttemptRecord, DespoticTable, LineLumResult, LogGrid, SpeciesLineGrid, SpeciesRecord

__all__ = [
    "LogGrid", "LineLumResult", "SpeciesLineGrid", "SpeciesRecord",
    "AttemptRecord", "DespoticTable", "SpeciesSpec", "GOW_LVG_SPECIES",
    "TableLookup", "build_gow_lvg_table", "save_table", "load_table",
]
