"""Data models for the canonical 3D GOW/LVG DESPOTIC table."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LogGrid:
    min_value: float
    max_value: float
    num_points: int

    def __post_init__(self) -> None:
        if self.min_value <= 0 or self.max_value <= 0:
            raise ValueError("LogGrid bounds must be positive")
        if self.min_value >= self.max_value:
            raise ValueError("LogGrid min_value must be smaller than max_value")
        if self.num_points < 2:
            raise ValueError("LogGrid num_points must be at least 2")

    def sample(self) -> np.ndarray:
        return np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)


@dataclass(frozen=True)
class LineLumResult:
    freq: float
    intIntensity: float
    intTB: float
    lumPerH: float
    tau: float
    tauDust: float


@dataclass(frozen=True)
class SpeciesLineGrid:
    freq: np.ndarray
    intIntensity: np.ndarray
    intTB: np.ndarray
    lumPerH: np.ndarray
    tau: np.ndarray
    tauDust: np.ndarray
    abundance: np.ndarray


@dataclass(frozen=True)
class SpeciesRecord:
    name: str
    abundance: np.ndarray
    line: SpeciesLineGrid | None
    is_emitter: bool = False

    def require_line(self) -> SpeciesLineGrid:
        if self.line is None:
            raise ValueError(f"Species '{self.name}' has no line data")
        return self.line


@dataclass(frozen=True)
class AttemptRecord:
    row_idx: int
    col_idx: int
    nH: float
    colDen: float
    tg_guess: float
    final_Tg: float
    converged: bool
    message: str | None = None
    duration: float | None = None
    dvdr_idx: int | None = None
    dvdr: float | None = None


@dataclass(frozen=True)
class DespoticTable:
    species_data: Mapping[str, SpeciesRecord]
    tg_final: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    dVdr_values: np.ndarray
    mu_values: np.ndarray
    cv_values: np.ndarray
    Eint_values: np.ndarray
    failure_mask: np.ndarray | None = None
    energy_terms: Mapping[str, np.ndarray] | None = None
    attempts: tuple[AttemptRecord, ...] = field(default_factory=tuple)
    chemistry_network: str = "GOW"
    escape_geometry: str = "LVG"
    temperature_mode: str = "iterateDust"

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_data", MappingProxyType(dict(self.species_data)))
        expected = self.tg_final.shape
        for name, values in (
            ("mu_values", self.mu_values),
            ("cv_values", self.cv_values),
            ("Eint_values", self.Eint_values),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape {values.shape} does not match tg_final {expected}")
        if self.failure_mask is not None and self.failure_mask.shape != expected:
            raise ValueError("failure_mask shape must match tg_final")
        if self.energy_terms is not None:
            object.__setattr__(self, "energy_terms", MappingProxyType(dict(self.energy_terms)))

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.species_data)

    @property
    def abundances(self) -> Mapping[str, np.ndarray]:
        return {name: record.abundance for name, record in self.species_data.items()}

    def require_species(self, name: str) -> SpeciesRecord:
        try:
            return self.species_data[name]
        except KeyError as exc:
            raise ValueError(f"Species '{name}' not found; available: {', '.join(self.species)}") from exc

    def clone_species_fields(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                field: np.array(getattr(record.line, field), copy=True)
                for field in ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
            }
            for name, record in self.species_data.items()
            if record.line is not None
        }
