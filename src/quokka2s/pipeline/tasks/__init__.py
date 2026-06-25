"""Export pipeline task entry points.

Active tasks follow the Build_X (compute + store) / Plot_X (plot) split:
``--mode compute`` runs the Build tasks, ``--mode plot`` runs the Plot tasks,
``--mode all`` runs Builds (registered first) then Plots.  A few toggle-able
plotting utilities stay on the legacy single-``AnalysisTask`` lifecycle.
Retired tasks live in ``tasks/archive/`` as inert reference.

Note: ``integrated_spectrum.py`` and ``temperature_lext_diff.py`` stay in this
package (NOT exported as tasks) because live tasks import shared code from them
(``SPECIES_CFG`` / ``V_RANGE_KMS`` and the ``_glob_one_taskcache`` /
``_load_results`` / ``_expected_sibling_key`` helpers).
"""
# ── Build tasks (compute + store) ─────────────────────────────────────────
from .velocity_phase import Build_VelocityPhase
from .species_spectrum import Build_SpeciesSpectrum
from .phase_hist import Build_PhaseHist, Build_PhaseHistNHRho
from .multi_field_slices import Build_MultiFieldSlices

# ── Plot tasks (read Build results + render) ──────────────────────────────
from .velocity_phase import Plot_VelocityPhase
from .species_spectrum import Plot_SpeciesSpectrum
from .multi_field_slices import Plot_MultiFieldSlices
from .phase_combined_plot import Plot_PhaseCombined
from .phase_spectrum_overlay import Plot_PhaseSpectrumOverlay

# ── Toggle-able plotting utilities (legacy lifecycle; not in the default run) ──
from .temperature_slices import TemperatureSlicesTask
from .temperature_projection import TemperatureProjectionTask
from .density_projection import DensityProjectionTask
from .temperature_compare import TemperatureCompareTask
from .table_diagnostics import TableDiagnosticsTask


__all__ = [
    # Build
    "Build_VelocityPhase",
    "Build_SpeciesSpectrum",
    "Build_PhaseHist",
    "Build_PhaseHistNHRho",
    "Build_MultiFieldSlices",
    # Plot
    "Plot_VelocityPhase",
    "Plot_SpeciesSpectrum",
    "Plot_MultiFieldSlices",
    "Plot_PhaseCombined",
    "Plot_PhaseSpectrumOverlay",
    # toggle-able utilities
    "TemperatureSlicesTask",
    "TemperatureProjectionTask",
    "DensityProjectionTask",
    "TemperatureCompareTask",
    "TableDiagnosticsTask",
]
