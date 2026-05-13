"""Entry point for the H-alpha analysis pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add the project root to sys.path for relative imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ..base import Pipeline, PipelineConfig
from ..prep import config as cfg
from ..prep import physics_fields as phys
from . import (
    DensityProjectionTask,
    HalphaTask,
    EmitterTask,
    COLine1DTask,
    CplusLine1DTask,
    HCOplusLine1DTask,
    TripleLineTask,
    TemperatureCompareTask,
    SigmaNTCheckTask,
    TableDiagnosticsTask,
    EmitterCompareTask,
    IntegratedSpectrumTask,
    BinnedPixelGridTask,
    PhaseSigmaVTask,
    PhaseSpectrumOverlayTask,
    PhaseResolvedSpectrumTask,
    SpaxelSigmaTask,
    SigmaSFROverlayTask,
    TemperatureSlicesTask,
)





def build_pipeline() -> Pipeline:
    """Configure and assemble the pipeline with the desired tasks."""
    pipeline_config = PipelineConfig(
        dataset_path=cfg.YT_DATASET_PATH,
        output_dir=Path(cfg.OUTPUT_DIR),
        figure_units="kpc",
        projection_axis="x",
        field_setup=phys.add_all_fields,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
    )

    pipeline = Pipeline(pipeline_config)
    pipeline.register_task(TemperatureSlicesTask(pipeline_config, n_slices=2))
    # pipeline.register_task(TableDiagnosticsTask(pipeline_config))
    pipeline.register_task(DensityProjectionTask(pipeline_config, axis="x", figure_units='kpc'))

    # pipeline.register_task(HalphaTask(pipeline_config, axis="x", figure_units='kpc'))


    pipeline.register_task(EmitterTask(pipeline_config, axis="x", figure_units='kpc'))

    

    pipeline.register_task(PhaseSigmaVTask(pipeline_config))
    # pipeline.register_task(SpaxelSigmaTask(pipeline_config))
    # pipeline.register_task(SigmaSFROverlayTask(pipeline_config))

    pipeline.register_task(IntegratedSpectrumTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(BinnedPixelGridTask(pipeline_config, species='CO',      bin_size=8, max_panels_per_side=10))
    # pipeline.register_task(BinnedPixelGridTask(pipeline_config, species='C+',      bin_size=8, max_panels_per_side=10))
    # pipeline.register_task(BinnedPixelGridTask(pipeline_config, species='H_alpha', bin_size=8, max_panels_per_side=10))
    # pipeline.register_task(BinnedPixelGridTask(pipeline_config, species='HI',      bin_size=8, max_panels_per_side=10))
    pipeline.register_task(PhaseSpectrumOverlayTask(pipeline_config))                # R = ∞ (no LSF)
    pipeline.register_task(PhaseResolvedSpectrumTask(pipeline_config))
    
    
    
    # pipeline.register_task(PhaseSpectrumOverlayTask(pipeline_config, R=1e5))         # near-ideal
    # pipeline.register_task(PhaseSpectrumOverlayTask(pipeline_config, R=1e4))         # heavy smearing
    # pipeline.register_task(PhaseSpectrumOverlayTask(pipeline_config, R=1e3))         # extreme smearing
   
   
    # pipeline.register_task(COLine1DTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(CplusLine1DTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(HCOplusLine1DTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(TripleLineTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(TemperatureCompareTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(EmitterCompareTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(SigmaNTCheckTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(HalphaWithDustTask(pipeline_config, axis="x"))
    # pipeline.register_task(HalphaComparisonTask(pipeline_config, axis="x"))
    return pipeline



def main() -> None:
    print("Pipeline Start")
    print("="*50)
    print(f"OUTPUT_DIR = {cfg.OUTPUT_DIR}")
    print("="*50)
    pipeline = build_pipeline()
    pipeline.run()



if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\nTotal analysis time: {elapsed/60:.2f} minutes")
