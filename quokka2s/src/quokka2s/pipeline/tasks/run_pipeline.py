"""Entry point for the quokka2s analysis pipeline.

CLI:
    python -m quokka2s.pipeline.tasks.run_pipeline                  # full run
    python -m quokka2s.pipeline.tasks.run_pipeline --mode compute   # only physics
    python -m quokka2s.pipeline.tasks.run_pipeline --mode plot      # only plotting
    python -m quokka2s.pipeline.tasks.run_pipeline --force          # ignore intermediates
    python -m quokka2s.pipeline.tasks.run_pipeline --task EmitterTask --task PhaseSigmaVTask
    python -m quokka2s.pipeline.tasks.run_pipeline --clean-intermediates  # rm both stores and exit
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Add the project root to sys.path for relative imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ..base import Pipeline, PipelineConfig
from ..cache import cache_root_for_dataset
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
    TemperatureProjectionTask,
    TemperatureLextDiffTask,
    EmitterLextDiffTask,
    TrustRegionTask,
    MultiFieldSlicesTask,
    PhasePlotTask,
    PhaseColdenTask,
    PhaseCombinedTask,
)





def build_pipeline(force: bool = False) -> Pipeline:
    """Configure and assemble the pipeline with the desired tasks."""
    pipeline_config = PipelineConfig(
        dataset_path=cfg.YT_DATASET_PATH,
        output_dir=Path(cfg.OUTPUT_DIR),
        figure_units="kpc",
        projection_axis="x",
        field_setup=phys.add_all_fields,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
        force_recompute=force,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )

    pipeline = Pipeline(pipeline_config)
    pipeline.register_task(TemperatureProjectionTask(pipeline_config, slice_axis='x', figure_units='kpc'))
    pipeline.register_task(TemperatureSlicesTask(pipeline_config, n_slices=4, figure_units='kpc'))
    # Cross-run comparison; reads two TemperatureSlicesTask intermediates.
    # Skips with a warning if either L_ext source dir is missing.
    pipeline.register_task(TemperatureLextDiffTask(pipeline_config,
                                                   L_ext_baseline=0.0, L_ext_compare=9.0))
    pipeline.register_task(EmitterLextDiffTask(pipeline_config,
                                               L_ext_baseline=0.0, L_ext_compare=9.0))
    pipeline.register_task(TrustRegionTask(pipeline_config))
    pipeline.register_task(MultiFieldSlicesTask(pipeline_config, slice_axis='x', figure_units='kpc',
                                                share_lext_partners=(0.0, 9.0, 99.0)))
    pipeline.register_task(PhasePlotTask(pipeline_config))
    pipeline.register_task(PhaseColdenTask(pipeline_config))
    pipeline.register_task(PhaseCombinedTask(pipeline_config, share_lext_partners=(0.0, 9.0)))
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



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Run the quokka2s post-processing pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--mode', choices=('all', 'compute', 'plot'),
                    default='all',
                    help='all (default): compute+plot;  compute: skip plotting;  '
                         'plot: load cached compute() results and replot only.')
    ap.add_argument('--force', action='store_true',
                    help='Ignore the intermediates store; recompute everything. '
                         'Intermediate files are rewritten so subsequent runs benefit.')
    ap.add_argument('--task', action='append', default=[],
                    help='Run only the named task class (repeatable). e.g. '
                         '--task EmitterTask --task PhaseSigmaVTask')
    ap.add_argument('--clean-intermediates', action='store_true',
                    help='Delete the field intermediates (per-dataset) and task '
                         'intermediates (per-output) directories, then exit.')
    return ap.parse_args()


def clean_intermediates() -> None:
    """Remove both intermediate-data stores for the current dataset/output config."""
    field_store = cache_root_for_dataset(cfg.YT_DATASET_PATH)
    task_store  = Path(cfg.OUTPUT_DIR) / 'task_intermediates'
    for label, p in [('field intermediates', field_store),
                     ('task intermediates',  task_store)]:
        if p.exists():
            shutil.rmtree(p)
            print(f'[clean] removed {label}: {p}')
        else:
            print(f'[clean] {label} was not present: {p}')


def main() -> None:
    args = parse_args()

    if args.clean_intermediates:
        clean_intermediates()
        return

    print('Pipeline Start')
    print('=' * 50)
    print(f'OUTPUT_DIR = {cfg.OUTPUT_DIR}')
    print(f'mode = {args.mode}   force = {args.force}   '
          f'task_filter = {args.task or "all"}')
    print('=' * 50)

    pipeline = build_pipeline(force=args.force)
    pipeline.run(mode=args.mode, task_filter=args.task or None)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\nTotal analysis time: {elapsed/60:.2f} minutes")
