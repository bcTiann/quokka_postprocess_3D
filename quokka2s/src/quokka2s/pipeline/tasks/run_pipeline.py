"""Entry point for the H-alpha analysis pipeline."""
from __future__ import annotations

from pathlib import Path
import time

from quokka2s.pipeline.base import Pipeline, PipelineConfig
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep import physics_fields as phys
from quokka2s.pipeline.tasks import (
    DensityProjectionTask,
    HalphaTask,
    EmitterTask,
)





def build_pipeline() -> Pipeline:
    """Configure and assemble the pipeline with the desired tasks."""
    pipeline_config = PipelineConfig(
        dataset_path=cfg.YT_DATASET_PATH,
        output_dir=Path(cfg.OUTPUT_DIR),
        figure_units="kpc",
        projection_axis="x",
        field_setup=phys.add_all_fields,
    )

    pipeline = Pipeline(pipeline_config)
    # pipeline.register_task(DensityProjectionTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(HalphaTask(pipeline_config, axis="x", figure_units='kpc'))
    pipeline.register_task(EmitterTask(pipeline_config, axis="x", figure_units='kpc'))
    # pipeline.register_task(HalphaWithDustTask(pipeline_config, axis="x"))
    # pipeline.register_task(HalphaComparisonTask(pipeline_config, axis="x"))
    return pipeline



def main() -> None:
    pipeline = build_pipeline()
    pipeline.run()



if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\nTotal analysis time: {elapsed/60:.2f} minutes")
