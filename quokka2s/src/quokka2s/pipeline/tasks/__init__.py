"""Export pipeline task entry points."""
from .density_projection import DensityProjectionTask
from .halpha import HalphaTask
from .emitter import EmitterTask
from .co_spectrum_1d import COLine1DTask
from .cplus_spectrum_1d import CplusLine1DTask
from .hcoplus_spectrum_1d import HCOplusLine1DTask
from .triple_spectrum_1D import TripleLineTask
from .temperature_compare import TemperatureCompareTask
from .sigmaNT_check import SigmaNTCheckTask
from .table_diagnostics import TableDiagnosticsTask
from .emitter_compare import EmitterCompareTask
from .integrated_spectrum import IntegratedSpectrumTask
from .binned_pixel_grid import BinnedPixelGridTask
from .phase_sigmaV import PhaseSigmaVTask
from .phase_spectrum_overlay import PhaseSpectrumOverlayTask
from .phase_resolved_spectrum import PhaseResolvedSpectrumTask
from .velocity_phase import VelocityPhaseTask
from .species_spectrum import SpeciesSpectrumTask
from .spaxel_sigma import SpaxelSigmaTask
from .sigma_sfr_overlay import SigmaSFROverlayTask
from .temperature_slices import TemperatureSlicesTask
from .temperature_projection import TemperatureProjectionTask
from .temperature_lext_diff import TemperatureLextDiffTask
from .emitter_lext_diff import EmitterLextDiffTask
from .trust_region import TrustRegionTask
from .multi_field_slices import MultiFieldSlicesTask
from .phase_plots import PhasePlotTask
from .phase_colden import PhaseColdenTask
# PhaseCombinedTask deprecated 2026-06-19 — replaced by
# PhaseHistTask + PhaseHistNHRhoTask + PhaseCombinedPlotTask below.
# from .phase_combined import PhaseCombinedTask
from .phase_hist import PhaseHistTask, PhaseHistNHRhoTask
from .phase_combined_plot import PhaseCombinedPlotTask


__all__ = [
    "DensityProjectionTask",
    "HalphaTask",
    "EmitterTask",
    "COLine1DTask",
    "CplusLine1DTask",
    "HCOplusLine1DTask",
    "TripleLineTask",
    "TemperatureCompareTask",
    "SigmaNTCheckTask",
    "TableDiagnosticsTask",
    "EmitterCompareTask",
    "IntegratedSpectrumTask",
    "BinnedPixelGridTask",
    "PhaseSigmaVTask",
    "PhaseSpectrumOverlayTask",
    "PhaseResolvedSpectrumTask",
    "VelocityPhaseTask",
    "SpeciesSpectrumTask",
    "SpaxelSigmaTask",
    "SigmaSFROverlayTask",
    "TemperatureSlicesTask",
    "TemperatureProjectionTask",
    "TemperatureLextDiffTask",
    "EmitterLextDiffTask",
    "TrustRegionTask",
    "MultiFieldSlicesTask",
    "PhasePlotTask",
    "PhaseColdenTask",
    # "PhaseCombinedTask",   # deprecated 2026-06-19
    "PhaseHistTask",
    "PhaseHistNHRhoTask",
    "PhaseCombinedPlotTask",
    # "HalphaWithDustTask",
    # "HalphaComparisonTask",
]
