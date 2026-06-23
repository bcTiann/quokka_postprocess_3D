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
    # active pipeline
    VelocityPhaseTask,
    SpeciesSpectrumTask,
    PhaseSpectrumOverlayTask,
    MultiFieldSlicesTask,
    PhaseHistTask,
    PhaseHistNHRhoTask,
    PhaseCombinedPlotTask,
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
    # Active pipeline: velocity/phase analysis + species spectra + the phase
    # overlay, the 8 PhaseHist tasks feeding PhaseCombinedPlot, and the
    # 10-slice MultiFieldSlices figure.  Retired tasks live in tasks/archive/;
    # toggle-able plotting utilities are listed (commented) at the end.
    pipeline.register_task(VelocityPhaseTask(pipeline_config))
    pipeline.register_task(SpeciesSpectrumTask(pipeline_config))
    pipeline.register_task(PhaseSpectrumOverlayTask(pipeline_config))                # R = ∞ (no LSF)

    # MultiFieldSlicesTask: 10 evenly-spaced x-slices (idx 12, 38, 63, 89, 114,
    # 140, 165, 191, 216, 242) matching the GOW_10x convention, L=15 only
    # (per [[l15-only]]).  share_lext_partners=() per the same convention.
    # Output: output_dir/multi_field_slices_10x/multi_field_slices_idxNNNN.png × 10.
    # np.linspace(12, 242, 10).round().astype(int) — pre-computed to avoid
    # importing numpy in this entry-point file.
    _MFS_INDICES = (12, 38, 63, 89, 114, 140, 165, 191, 216, 242)
    pipeline.register_task(MultiFieldSlicesTask(pipeline_config,
                                                 slice_axis='x',
                                                 slice_indices=_MFS_INDICES,
                                                 figure_units='kpc',
                                                 subdir='multi_field_slices_10x',
                                                 share_lext_partners=()))
    # Old single-task PhaseCombinedTask is replaced by 8 PhaseHist compute
    # tasks + 1 PhaseCombinedPlotTask (2026-06-19).  See phase_hist.py +
    # phase_combined_plot.py.
    T_2R = 'temperature_two_regime'
    pipeline.register_task(PhaseHistTask(pipeline_config, 'mass', 'temperature_quokka',
                                         tag='mass_T_QK',
                                         units_label=r'$\log_{10}\,M_{\rm bin}$ [g]'))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'mass', 'temperature_despotic',
                                         tag='mass_T_DSP',
                                         units_label=r'$\log_{10}\,M_{\rm bin}$ [g]'))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'mass', T_2R,
                                         tag='mass_T_2R',
                                         units_label=r'$\log_{10}\,M_{\rm bin}$ [g]'))
    pipeline.register_task(PhaseHistNHRhoTask(pipeline_config))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'CO_luminosity', T_2R,
                                         tag='CO_T_2R',
                                         units_label=r'$\log_{10}\,L_{\rm CO}$ [erg s$^{-1}$]'))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'C+_luminosity', T_2R,
                                         tag='Cplus_T_2R',
                                         units_label=r'$\log_{10}\,L_{\rm C^+}$ [erg s$^{-1}$]'))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'H_alpha_luminosity', T_2R,
                                         tag='Halpha_T_2R',
                                         units_label=r'$\log_{10}\,L_{\rm H\alpha}$ [erg s$^{-1}$]'))
    pipeline.register_task(PhaseHistTask(pipeline_config, 'HI_luminosity', T_2R,
                                         tag='HI_T_2R',
                                         units_label=r'$\log_{10}\,L_{\rm HI}$ [erg s$^{-1}$]'))
    pipeline.register_task(PhaseCombinedPlotTask(pipeline_config))

    # ── Optional plotting utilities (kept; uncomment to include) ────────
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
