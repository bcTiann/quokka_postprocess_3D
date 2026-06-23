"""Export pipeline task entry points.

Only the tasks used by the current pipeline — plus a few toggle-able plotting
utilities — are exported here.  Retired tasks live in ``tasks/archive/`` as
inert reference (see archive/README.md).

Note: ``integrated_spectrum.py`` and ``temperature_lext_diff.py`` stay in this
package (and are NOT exported as tasks below) because live tasks import shared
code from them — ``SPECIES_CFG`` / ``V_RANGE_KMS`` and the
``_glob_one_taskcache`` / ``_load_results`` / ``_expected_sibling_key`` helpers
respectively — even though their own task classes are not registered.
"""
# ── Active pipeline tasks ────────────────────────────────────────────────
from .velocity_phase import VelocityPhaseTask
from .species_spectrum import SpeciesSpectrumTask
from .phase_spectrum_overlay import PhaseSpectrumOverlayTask
from .multi_field_slices import MultiFieldSlicesTask
from .phase_hist import PhaseHistTask, PhaseHistNHRhoTask
from .phase_combined_plot import PhaseCombinedPlotTask

# ── Toggle-able plotting utilities (kept; not in the default run) ────────
from .temperature_slices import TemperatureSlicesTask
from .temperature_projection import TemperatureProjectionTask
from .density_projection import DensityProjectionTask
from .temperature_compare import TemperatureCompareTask
from .table_diagnostics import TableDiagnosticsTask


__all__ = [
    "VelocityPhaseTask",
    "SpeciesSpectrumTask",
    "PhaseSpectrumOverlayTask",
    "MultiFieldSlicesTask",
    "PhaseHistTask",
    "PhaseHistNHRhoTask",
    "PhaseCombinedPlotTask",
    "TemperatureSlicesTask",
    "TemperatureProjectionTask",
    "DensityProjectionTask",
    "TemperatureCompareTask",
    "TableDiagnosticsTask",
]
