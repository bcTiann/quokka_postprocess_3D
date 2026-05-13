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
    Shared context for pipline tasks.
    Holds the yt dataset/provider plus a mutable cache for task outputs
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
    Each task performs a specific analysis step and can store results in the shared context.
    """

    name: str

    def __init__(self, config: Mapping[str, Any], name: Optional[str] = None) -> None:
        self.config = config
        self.name = name or self.__class__.__name__ # Use class name as default task name, or you can provide a custom name

    def prepare(self, context: PipelinePlotContext) -> None:
        """Optional hook before compute() (e.g., heavy data loads)."""

    def compute(self, context: PipelinePlotContext) -> Dict[str, Any]:
        """
        Main calculation step.
        Returns a dictionary passed to plot().
        """
        return {}
    

    def plot(self, context: PipelinePlotContext, results: Dict[str, Any]) -> None:
        """Generate figures or other artifacts using compute() outputs."""


    def run(self, context: PipelinePlotContext) -> None:
        """
        Execute the analysis task using the provided context.

        Parameters
        ----------
        context : PipelinePlotContext
            The shared context containing the dataset and results cache.
        """
        self.prepare(context)
        results = self.compute(context)
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
        provider = YTDataProvider(ds)
        return PipelinePlotContext(ds=ds, provider=provider, config=self.config)

    def register_task(self, task: AnalysisTask) -> None:
        self._tasks.append(task)

    def run(self) -> None:
        context = self.build_context()
        for task in self._tasks:
            task.run(context)

            
