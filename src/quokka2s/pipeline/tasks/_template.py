"""Template for a new pipeline task.  Copy this file, rename the class, and
fill in the two methods (``compute`` and ``plot``).

Lifecycle
---------
Each task runs in two phases inside ``AnalysisTask.run()``:

    1. compute(context) -> dict
         Load whatever 3D / 2D slabs you need via ``context.provider`` AS
         LOCAL VARIABLES (never store on ``self``).  Slabs whose field is
         in ``pipeline.cache.CACHED_FIELDS`` hit the disk intermediate;
         everything else is computed fresh by yt the first time, then
         (for cached fields) written to disk.
         Do the numerical work and return a small dict — keep it to
         scalars, 1D/2D arrays.  Avoid returning 3D cubes (the task
         intermediate would balloon).
         When ``compute()`` returns, every local variable goes out of scope
         and Python frees the memory.  This is how the pipeline keeps a
         clean memory profile across tasks.

    2. plot(context, results)
         Read the dict from compute() and write figures into
         ``context.config.output_dir``.  Should NOT touch the dataset
         directly — that way ``--mode plot`` can re-run plotting from a
         cached results dict without paying for compute().

Conventions
-----------
* Store only small config on ``self`` (axis, R, bin_size).  Never
  ``self._foo_3d = ...`` — that creates cross-task memory accumulation
  because task objects live in ``pipeline._tasks`` for the whole run.
* Output file names: ``<self.config.output_dir>/<SomeDescriptiveName>.png``.
* Use ``PHASE_COLOR`` / ``PHASE_LABEL_LINE`` from ``pipeline.utils`` when
  splitting by ISM phase, so colour and threshold conventions stay
  consistent across tasks.
* Keep ``compute()`` deterministic.  If you need RNG, seed it explicitly.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext


class MyTemplateTask(AnalysisTask):
    """One-line description of what this task analyses / produces.

    Output:
        <output_dir>/MyTemplate.png
    """

    def __init__(self, config, my_param: float = 1.0, axis: str = 'x'):
        super().__init__(config)
        # Store only small config.  Their repr is hashed into the
        # task-intermediate filename, so different parameter combinations get
        # separate cache files automatically.
        self.my_param = float(my_param)
        self.axis     = axis

    # ────────────────────────────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        """Load fields locally, compute, return a small dict.

        Every variable defined here goes out of scope when compute() returns,
        so 3D arrays are freed automatically.  No self._foo state.
        """
        p = context.provider
        rho_u, extent = p.get_slab_z(('gas', 'density'))
        T_u,   _      = p.get_slab_z(('gas', 'temperature_despotic'))
        rho = rho_u.value
        T   = T_u.in_units('K').value
        del rho_u, T_u   # 2 GB released immediately

        # ... your physics here ...
        rho_mean    = float(np.mean(rho))
        T_median    = float(np.median(T))
        log_rho_2d  = np.log10(rho.mean(axis=0)).astype(np.float32)

        return {
            'rho_mean_cgs': rho_mean,
            'T_median_K':   T_median,
            'log_rho_2d':   log_rho_2d,      # 2D, OK to put in dict
            'extent':       extent[self.axis],
            'meta': {
                'shape':    list(rho.shape),
                'my_param': self.my_param,
            },
        }
        # rho, T (3D, ~2 GB) go out of scope here → GC'd

    # ────────────────────────────────────────────────────────────────────────
    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        """Render the figure from the compute() dict.  No data access here."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(results['log_rho_2d'].T, origin='lower', cmap='viridis')
        ax.set_title(
            f'{self.__class__.__name__}  '
            f'(my_param={self.my_param}, axis={self.axis})\n'
            f'⟨ρ⟩={results["rho_mean_cgs"]:.2e} g/cm³,  '
            f'median(T)={results["T_median_K"]:.0f} K'
        )
        out = context.config.output_dir / 'MyTemplate.png'
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
