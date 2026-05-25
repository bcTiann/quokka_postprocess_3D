"""Core pipeline abstractions for quokka2s."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Callable
import yt

from ..data_handling import YTDataProvider, make_downsampled_dataset



@dataclass
class PipelinePlotContext:
    """
    Shared context for pipeline tasks.

    Holds the yt dataset/provider plus a mutable results dict.  No long-lived
    in-memory caches across tasks — anything expensive that needs to survive
    a task boundary lives in the disk intermediate (see ``pipeline.cache``).
    """

    ds: yt.Dataset
    provider: YTDataProvider
    results: Dict[str, Any] = field(default_factory=dict)
    config: Optional[PipelineConfig] = None

@dataclass
class PipelineConfig:
    """
    Global configuration shared by every task in the pipeline.
    Handles dataset loading and optional physics-field registration.
    """
    dataset_path: str
    output_dir: Path
    figure_units: str = "pc"
    projection_axis: str = "x",
    field_setup: Optional[Callable[[yt.Dataset], None]] = None
    downsample_factor: int = 1
    # Cache controls.  Populated by build_pipeline() / argparse.
    despotic_table_path: Optional[str] = None
    cache_enabled: bool = True
    force_recompute: bool = False
    # Lateral (x/y) box-exterior column-density extension, in kpc.  Folded into
    # the cache key so L_ext=0 and L_ext=9 runs don't poison each other.
    column_extension_lateral_kpc: float = 0.0
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def load_dataset(self) -> yt.Dataset:
        """Load the yt dataset and register derived fields if requested."""
        ds = yt.load(self.dataset_path)
        if self.downsample_factor > 1:
            ds = make_downsampled_dataset(ds, self.downsample_factor)
        if self.field_setup:
            self.field_setup(ds)
        return ds
    
    def ensure_output_dir(self) -> None:
        """Create the output directory (and parents) if missing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


    
class AnalysisTask:
    """
    Base class for analysis tasks in pipeline.

    Lifecycle (only two domain methods):
        compute()  → load fields locally, calculate, return small results dict
        plot()     → consume the dict, write figures to output_dir

    Tasks must NOT store 3D arrays on ``self``.  Only small config (axis,
    bin_size, R, ...) belongs on self.  All 3D field loads live as local
    variables inside ``compute()`` so they're released the moment compute()
    returns.

    Execution modes (see run()):
        'all'      → compute → save task intermediate → plot
        'compute'  → compute → save task intermediate; skip plot
        'plot'     → load task intermediate; if missing, fall back to compute;
                     then plot.  Lets you iterate on figure styling without
                     re-running physics.
    """

    name: str

    def __init__(self, config: Mapping[str, Any], name: Optional[str] = None) -> None:
        self.config = config
        self.name = name or self.__class__.__name__ # Use class name as default task name, or you can provide a custom name

    def compute(self, context: PipelinePlotContext) -> Dict[str, Any]:
        """
        Load needed fields + calculate + return a small dict for plot().
        Local variables go out of scope on return so 3D arrays are freed.
        """
        return {}


    def plot(self, context: PipelinePlotContext, results: Dict[str, Any]) -> None:
        """Generate figures or other artifacts using compute() outputs."""


    # ── Level 2 task-result cache ──────────────────────────────────────────
    def _cache_filename(self) -> str:
        """Per-task HDF5 file name in ``<output_dir>/.task_cache/``.

        Instances of the same class with different __init__ args get different
        files (e.g. ``BinnedPixelGridTask(species='CO')`` vs ``species='C+'``).
        """
        import hashlib, json
        relevant_attrs = {
            k: repr(v) for k, v in sorted(self.__dict__.items())
            if k != 'config' and not k.startswith('_')
        }
        h = hashlib.sha1(json.dumps(relevant_attrs, sort_keys=True).encode()).hexdigest()
        return f'{self.__class__.__name__}_{h[:8]}.h5'

    def _cache_path(self, context: PipelinePlotContext) -> Path:
        return Path(context.config.output_dir) / 'task_intermediates' / self._cache_filename()

    def _l2_cache_key(self, context: PipelinePlotContext) -> Optional[str]:
        """Cache key for this task's results.  None disables Level 2."""
        cfg = context.config
        if not getattr(cfg, 'cache_enabled', True):
            return None
        if cfg.despotic_table_path is None:
            return None
        from .cache import compute_cache_key
        # L2 key = L1 key + task identity (class + init args).
        return compute_cache_key(
            dataset_path        = cfg.dataset_path,
            despotic_table_path = cfg.despotic_table_path,
            downsample_factor   = cfg.downsample_factor,
            column_extension_lateral_kpc = cfg.column_extension_lateral_kpc,
        ) + ':' + self._cache_filename()

    def _save_results(self, context: PipelinePlotContext, results: Dict[str, Any]) -> None:
        key = self._l2_cache_key(context)
        if key is None:
            return
        from .cache import save_results_dict
        save_results_dict(self._cache_path(context), results, cache_key=key)
        print(f'[task-intermediate] save  {self.name}  →  {self._cache_path(context).name}')

    def _load_results(self, context: PipelinePlotContext) -> Optional[Dict[str, Any]]:
        key = self._l2_cache_key(context)
        if key is None:
            return None
        if getattr(context.config, 'force_recompute', False):
            return None
        from .cache import load_results_dict
        results = load_results_dict(self._cache_path(context), expected_cache_key=key)
        if results is not None:
            print(f'[task-intermediate] load  {self.name}  ←  {self._cache_path(context).name}')
        return results

    def run(self, context: PipelinePlotContext, mode: str = 'all') -> None:
        """
        Execute the analysis task using the provided context.

        Parameters
        ----------
        context : PipelinePlotContext
            The shared context containing the dataset and results cache.
        mode : str
            'all'     → compute + plot (default).
            'compute' → compute and save task intermediate; skip plotting.
            'plot'    → load task intermediate; if missing, compute+save first.
        """
        if mode not in ('all', 'compute', 'plot'):
            raise ValueError(f'unknown task mode: {mode!r}')

        results: Optional[Dict[str, Any]] = None

        if mode == 'plot':
            results = self._load_results(context)

        if results is None:
            results = self.compute(context)
            self._save_results(context, results)

        if mode in ('plot', 'all'):
            self.plot(context, results)



class Pipeline:
    """Sequential pipeline runner."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._tasks: List[AnalysisTask] = []


    def build_context(self) -> PipelinePlotContext:
        """Load dataset/provider once and build the shared context."""
        self.config.ensure_output_dir()
        ds = self.config.load_dataset()

        # Resolve the field-intermediate cache root + identity key from the
        # dataset and table identities. Pass to the provider so get_slab_z
        # can skip expensive derived-field recomputes across runs.
        cache_root = None
        cache_key  = None
        if self.config.cache_enabled and self.config.despotic_table_path is not None:
            from .cache import cache_root_for_dataset, compute_cache_key
            cache_root = cache_root_for_dataset(self.config.dataset_path)
            cache_key  = compute_cache_key(
                dataset_path        = self.config.dataset_path,
                despotic_table_path = self.config.despotic_table_path,
                downsample_factor   = self.config.downsample_factor,
                column_extension_lateral_kpc = self.config.column_extension_lateral_kpc,
            )

        provider = YTDataProvider(
            ds,
            cache_root      = cache_root,
            cache_key       = cache_key,
            force_recompute = self.config.force_recompute,
        )

        return PipelinePlotContext(ds=ds, provider=provider, config=self.config)

    def register_task(self, task: AnalysisTask) -> None:
        self._tasks.append(task)

    def run(self, mode: str = 'all',
            task_filter: Optional[List[str]] = None) -> None:
        """
        Execute the pipeline.

        Parameters
        ----------
        mode : str
            Passed through to each task: 'all' | 'compute' | 'plot'.
        task_filter : list[str] | None
            If set, only run tasks whose class name matches one of these.

        Prints a ``[N/M] TaskName ...`` banner before each task and a
        cumulative wall-clock summary at the end so it's easy to see where
        the pipeline is and which task dominates the run time.
        """
        import time as _time

        context = self.build_context()
        tasks_to_run = [t for t in self._tasks
                        if not task_filter or t.__class__.__name__ in task_filter]
        n_total = len(tasks_to_run)

        if n_total == 0:
            print('(no tasks to run — empty filter?)')
            return

        pipeline_start = _time.perf_counter()
        per_task_seconds: Dict[str, float] = {}

        for idx, task in enumerate(tasks_to_run, start=1):
            name = task.__class__.__name__
            cum_min = (_time.perf_counter() - pipeline_start) / 60.0
            banner = f' [{idx}/{n_total}] {name}   (cumulative {cum_min:5.2f} min) '
            print('\n' + '=' * len(banner))
            print(banner)
            print('=' * len(banner))
            t0 = _time.perf_counter()
            task.run(context, mode=mode)
            # Drop yt's covering_grid in-memory cache so the next task starts
            # clean.  Anything expensive lives in the disk intermediates; the
            # next task that needs it will reload from disk (1-2 sec).
            context.provider._cached_grid = None
            dt = _time.perf_counter() - t0
            per_task_seconds[name] = dt
            print(f'[{idx}/{n_total}] {name} done in {dt/60:.2f} min '
                  f'(cumulative {(_time.perf_counter()-pipeline_start)/60:.2f} min)')

        total = _time.perf_counter() - pipeline_start
        print()
        print('=' * 60)
        print(f'Pipeline complete in {total/60:.2f} min total')
        print('Per-task wall-clock (slowest first):')
        for n, dt in sorted(per_task_seconds.items(), key=lambda kv: -kv[1]):
            print(f'  {n:<32s} {dt/60:6.2f} min   ({100*dt/total:5.1f} %)')
        print('=' * 60)

            
