"""Export pipeline task entry points."""
from .density_projection import DensityProjectionTask
from .halpha import HalphaTask
from .emitter import EmitterTask
from .CO_specturm_1D import COLine1DTask
from .Cplus_specturm_1D import CplusLine1DTask
from .HCOplus_specturm_1D import HCOplusLine1DTask
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
from .spaxel_sigma import SpaxelSigmaTask
from .sigma_sfr_overlay import SigmaSFROverlayTask
from .temperature_slices import TemperatureSlicesTask


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
    "SpaxelSigmaTask",
    "SigmaSFROverlayTask",
    "TemperatureSlicesTask",
    # "HalphaWithDustTask",
    # "HalphaComparisonTask",
]
