"""Entry point for the quokka2s analysis pipeline.

CLI:
    python -m quokka2s.pipeline.tasks.run_pipeline                  # full run
    python -m quokka2s.pipeline.tasks.run_pipeline --mode compute   # only physics
    python -m quokka2s.pipeline.tasks.run_pipeline --mode plot      # only plotting
    python -m quokka2s.pipeline.tasks.run_pipeline --force          # ignore intermediates
    python -m quokka2s.pipeline.tasks.run_pipeline --task Build_VelocityPhase --task Plot_VelocityPhase
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
    # Build tasks (--mode compute)
    Build_VelocityPhase,
    Build_SpeciesSpectrum,
    Build_PhaseHist,
    Build_PhaseHistNHRho,
    Build_MultiFieldSlices,
    # Plot tasks (--mode plot)
    Plot_VelocityPhase,
    Plot_SpeciesSpectrum,
    Plot_MultiFieldSlices,
    Plot_PhaseCombined,
    Plot_PhaseSpectrumOverlay,
    # toggle-able plotting utilities (commented registrations below)
    TemperatureSlicesTask,
    TemperatureProjectionTask,
    DensityProjectionTask,
    TemperatureCompareTask,
    TableDiagnosticsTask,
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
    # Build/Plot split (2026-06-24): every task is either a Build_X (compute +
    # store) or a Plot_X (read stored result + render).  `--mode compute` runs
    # only the Build tasks, `--mode plot` only the Plot tasks, `--mode all` runs
    # Builds (registered first) then Plots.  Retired tasks live in tasks/archive/;
    # toggle-able plotting utilities are listed (commented) at the end.
    #
    # MultiFieldSlices: 10 evenly-spaced x-slices (matching the GOW_10x dir);
    # the SAME kwargs go to Build_ and Plot_ so they pair.  L=15 only
    # (share_lext_partners=()) per [[l15-only]].
    _MFS_INDICES = (12, 38, 63, 89, 114, 140, 165, 191, 216, 242)
    _MFS_KW = dict(slice_axis='x', slice_indices=_MFS_INDICES, figure_units='kpc',
                   subdir='multi_field_slices_10x', share_lext_partners=())
    T_2R = 'temperature_two_regime'

    # ── Build tasks (--mode compute).  VelocityPhase first: SpeciesSpectrum
    #    reads its σ_gas; PhaseCombined reads all the PhaseHist results. ──
    pipeline.register_task(Build_VelocityPhase(pipeline_config))
    pipeline.register_task(Build_SpeciesSpectrum(pipeline_config))
    # weight_field, T_field, tag  (display symbol lives in Plot_PhaseCombined._SYMBOL)
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'mass', 'temperature_quokka',   tag='mass_T_QK'))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'mass', 'temperature_despotic', tag='mass_T_DSP'))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'mass', T_2R,                    tag='mass_T_2R'))
    pipeline.register_task(Build_PhaseHistNHRho(pipeline_config))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'CO_luminosity',      T_2R, tag='CO_T_2R'))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'C+_luminosity',      T_2R, tag='Cplus_T_2R'))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'H_alpha_luminosity', T_2R, tag='Halpha_T_2R'))
    pipeline.register_task(Build_PhaseHist(pipeline_config, 'HI_luminosity',      T_2R, tag='HI_T_2R'))
    pipeline.register_task(Build_MultiFieldSlices(pipeline_config, **_MFS_KW))

    # ── Plot tasks (--mode plot) ──
    pipeline.register_task(Plot_VelocityPhase(pipeline_config))
    pipeline.register_task(Plot_SpeciesSpectrum(pipeline_config))
    pipeline.register_task(Plot_MultiFieldSlices(pipeline_config, **_MFS_KW))
    pipeline.register_task(Plot_PhaseCombined(pipeline_config))
    pipeline.register_task(Plot_PhaseSpectrumOverlay(pipeline_config))                # R = ∞ (no LSF)

    # ── Optional plotting utilities (legacy lifecycle; uncomment to include) ──
    # pipeline.register_task(TemperatureSlicesTask(pipeline_config, n_slices=4, figure_units='kpc'))
    # pipeline.register_task(TemperatureProjectionTask(pipeline_config, slice_axis='x', figure_units='kpc'))
    # pipeline.register_task(DensityProjectionTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(TemperatureCompareTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(TableDiagnosticsTask(pipeline_config))

    # (Retired tasks moved to tasks/archive/ on 2026-06-24.)
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
                         '--task Build_VelocityPhase --task Plot_VelocityPhase')
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
