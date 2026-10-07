"""Bundled input locations shared by configuration and numerical preparation."""
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAINE_TABLE = (
    REPOSITORY_ROOT / "vendor/draine/kext_albedo_WD_MW_3.1_60_D03.all"
)
