"""Augment an existing GOW/LVG table with the CO J=2->1 line.

The expensive chemistry and thermal equilibrium do not need to be repeated.
The canonical table already stores the final gas temperature, collider
abundances, CO abundance, and LVG inputs for every grid point.  This module
reconstructs that final state, solves only the CO level populations, and saves
the second transition under the fixed ``CO21`` table token.  The legacy ``CO``
token remains CO(1-0).

For converged cells we replay the current solver's final sphere-default
``dEdt()`` call and then request the final LVG line ladder. Failed or
garbage-temperature cells are left NaN in the raw output and filled in the
clean output using the same convex-hull-only interpolation policy as the
existing table cleaner.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from .io import load_table, save_table
from .models import DespoticTable, SpeciesLineGrid, SpeciesRecord
from .solver import (
    CO21_TABLE_TOKEN,
    LINE_RESULT_FIELDS,
    LVG_GEOMETRY,
    _configure_despotic_home,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "output_tables_3D_GOW_LVG"
DEFAULT_RAW_SOURCE = TABLE_DIR / "despotic_table.npz"
DEFAULT_CLEAN_SOURCE = TABLE_DIR / "despotic_table_clean.npz"
DEFAULT_RAW_OUTPUT = TABLE_DIR / "despotic_table_co10_co21.npz"
DEFAULT_CLEAN_OUTPUT = TABLE_DIR / "despotic_table_co10_co21_clean.npz"

CO10_TRANSITION = (1, 0)
CO21_TRANSITION = (2, 1)
CO21_FREQUENCY_HZ = 230.538e9
ORTHO_H2_FRACTION = 0.2
CO10_REPLAY_RTOL = 1.0e-6
CO10_REPLAY_BRIGHT_FRACTION = 1.0e-6
CO10_REPLAY_FAINT_ATOL_FRACTION = 1.0e-8
MAX_VALID_TG_K = 1.0e6


def _transition_by_levels(
    transitions: list[Mapping[str, float]],
    upper: int,
    lower: int,
) -> Mapping[str, float]:
    for transition in transitions:
        matches_upper = int(transition.get("upper", -1)) == upper
        matches_lower = int(transition.get("lower", -1)) == lower
        if matches_upper and matches_lower:
            return transition
    raise ValueError(f"DESPOTIC did not return CO transition {upper}->{lower}")


def _source_invalid_mask(table: DespoticTable) -> np.ndarray:
    """Match the canonical cleaner's failure/NaN/garbage-cell policy."""
    tg = np.asarray(table.tg_final, dtype=float)
    invalid = ~np.isfinite(tg) | (tg > MAX_VALID_TG_K)
    if table.failure_mask is not None:
        invalid |= table.failure_mask
    return invalid


def _compute_strip(
    row_idx: int,
    col_idx: int,
    nH: float,
    col_den: float,
    dvdr: np.ndarray,
    tg: np.ndarray,
    co_abundance: np.ndarray,
    h_abundance: np.ndarray,
    h2_abundance: np.ndarray,
    hp_abundance: np.ndarray,
    e_abundance: np.ndarray,
    expected_co10_lum_per_h: np.ndarray,
    valid: np.ndarray,
) -> tuple[int, int, dict[str, np.ndarray], list[tuple[float, float]], list[str]]:
    """Compute CO(2-1) for one fixed ``(nH, N_H)`` strip."""
    _configure_despotic_home()
    from despotic import cloud

    line_fields = {
        field: np.full(dvdr.shape, np.nan, dtype=float)
        for field in LINE_RESULT_FIELDS
    }
    validation_pairs: list[tuple[float, float]] = []
    failures: list[str] = []

    for dvdr_idx in np.flatnonzero(valid):
        cell = cloud(noWarn=True)
        cell.nH = float(nH)
        cell.colDen = float(col_den)
        cell.Tg = float(tg[dvdr_idx])
        cell.dVdr = float(dvdr[dvdr_idx])
        cell.sigmaNT = 2.0e5

        x_h2 = float(h2_abundance[dvdr_idx])
        cell.comp.xoH2 = ORTHO_H2_FRACTION * x_h2
        cell.comp.xpH2 = (1.0 - ORTHO_H2_FRACTION) * x_h2
        cell.comp.xHI = float(h_abundance[dvdr_idx])
        cell.comp.xHplus = float(hp_abundance[dvdr_idx])
        cell.comp.xe = float(e_abundance[dvdr_idx])
        cell.comp.xHe = 0.1

        # Match GOW.applyAbundances(): DESPOTIC's bulk composition does not
        # represent H2+, H3+, CHx, OHx, or HCO+, so GOW assigns their small
        # hydrogen remainder to H+ whenever the bulk-H sum fails its tolerance.
        try:
            cell.comp._check_abundance()
        except ValueError:
            cell.comp.xHplus = 1.0 - cell.comp.xHI - 2.0 * cell.comp.xH2

        cell.dust.alphaGD = 3.2e-34
        cell.dust.sigma10 = 2.0e-25
        cell.dust.sigmaPE = 1.0e-21
        cell.dust.sigmaISRF = 3.0e-22
        cell.dust.beta = 2.0
        cell.dust.Zd = 1.0
        cell.Td = 10.0
        cell.rad.TCMB = 2.73
        cell.rad.TradDust = 0.0
        cell.rad.ionRate = 2.0e-17
        cell.rad.chi = 1.0
        cell.addEmitter("CO", float(co_abundance[dvdr_idx]))
        cell.comp.computeDerived(cell.nH)

        output = io.StringIO()
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(output):
                warnings.simplefilter("ignore")
                # Replay the production solver's final dEdt() call. Besides
                # using sphere geometry, dEdt() retries the level solve with
                # progressively smaller damping when needed.
                cell.dEdt()
                transitions = cell.lineLum("CO", escapeProbGeom=LVG_GEOMETRY)

            co10 = _transition_by_levels(transitions, *CO10_TRANSITION)
            co21 = _transition_by_levels(transitions, *CO21_TRANSITION)
            for field in LINE_RESULT_FIELDS:
                line_fields[field][dvdr_idx] = float(co21[field])

            expected = float(expected_co10_lum_per_h[dvdr_idx])
            actual = float(co10["lumPerH"])
            validation_pairs.append((expected, actual))
        except Exception as exc:
            failures.append(
                f"({row_idx},{col_idx},{int(dvdr_idx)}): "
                f"{type(exc).__name__}: {exc}"
            )

    return row_idx, col_idx, line_fields, validation_pairs, failures


def _fill_in_hull(arr: np.ndarray, log_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Fill missing cells by 3D linear interpolation, never extrapolation."""
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("cannot fill a line field with no finite values")
    if finite.all():
        return np.array(arr, copy=True)

    use_log = bool((arr[finite] > 0.0).all())
    work = np.where(arr > 0.0, np.log10(arr), np.nan) if use_log else np.array(arr, copy=True)
    points = np.array(np.meshgrid(*log_axes, indexing="ij")).reshape(3, -1).T
    values = work.ravel()
    valid_values = np.isfinite(values)
    filled = griddata(
        points[valid_values], values[valid_values], points[~valid_values], method="linear",
    )
    out = np.array(values, copy=True)
    out[~valid_values] = filled
    out = out.reshape(arr.shape)
    return 10.0 ** out if use_log else out


def _with_co21(
    table: DespoticTable,
    line_fields: Mapping[str, np.ndarray],
) -> DespoticTable:
    """Return a table with a fixed CO21 pseudo-species line record added."""
    co_abundance = table.require_species("CO").abundance
    line = SpeciesLineGrid(
        freq=np.asarray(line_fields["freq"], dtype=float),
        intIntensity=np.asarray(line_fields["intIntensity"], dtype=float),
        intTB=np.asarray(line_fields["intTB"], dtype=float),
        lumPerH=np.asarray(line_fields["lumPerH"], dtype=float),
        tau=np.asarray(line_fields["tau"], dtype=float),
        tauDust=np.asarray(line_fields["tauDust"], dtype=float),
        abundance=co_abundance,
    )
    species_data = dict(table.species_data)
    species_data[CO21_TABLE_TOKEN] = SpeciesRecord(
        CO21_TABLE_TOKEN, co_abundance, line, True,
    )
    return replace(table, species_data=species_data)


def augment_tables(
    raw_source: Path,
    clean_source: Path,
    raw_output: Path,
    clean_output: Path,
    *,
    workers: int = -1,
) -> tuple[DespoticTable, DespoticTable]:
    """Compute CO(2-1), write raw/clean augmented tables, and return them."""
    raw = load_table(raw_source)
    clean = load_table(clean_source)
    if CO21_TABLE_TOKEN in raw.species_data or CO21_TABLE_TOKEN in clean.species_data:
        raise ValueError("source table already contains CO21")

    for raw_axis, clean_axis, name in (
        (raw.nH_values, clean.nH_values, "nH"),
        (raw.col_density_values, clean.col_density_values, "N_H"),
        (raw.dVdr_values, clean.dVdr_values, "dVdr"),
    ):
        if not np.array_equal(raw_axis, clean_axis):
            raise ValueError(f"raw and clean {name} axes do not match")
    if raw.tg_final.shape != clean.tg_final.shape:
        raise ValueError("raw and clean table shapes do not match")

    co = raw.require_species("CO")
    co_line = co.require_line()
    abundance = raw.abundances
    required = (
        raw.tg_final,
        co.abundance,
        abundance["H"],
        abundance["H2"],
        abundance["H+"],
        abundance["e-"],
        co_line.lumPerH,
    )
    valid = np.ones(raw.tg_final.shape, dtype=bool)
    for values in required:
        valid &= np.isfinite(values)
    valid &= ~_source_invalid_mask(raw)

    shape = raw.tg_final.shape
    line_fields = {
        field: np.full(shape, np.nan, dtype=float)
        for field in LINE_RESULT_FIELDS
    }
    tasks = []
    for row_idx, nH in enumerate(raw.nH_values):
        for col_idx, col_den in enumerate(raw.col_density_values):
            strip_valid = valid[row_idx, col_idx]
            if not strip_valid.any():
                continue
            tasks.append(delayed(_compute_strip)(
                row_idx,
                col_idx,
                float(nH),
                float(col_den),
                np.asarray(raw.dVdr_values),
                np.asarray(raw.tg_final[row_idx, col_idx]),
                np.asarray(co.abundance[row_idx, col_idx]),
                np.asarray(abundance["H"][row_idx, col_idx]),
                np.asarray(abundance["H2"][row_idx, col_idx]),
                np.asarray(abundance["H+"][row_idx, col_idx]),
                np.asarray(abundance["e-"][row_idx, col_idx]),
                np.asarray(co_line.lumPerH[row_idx, col_idx]),
                np.asarray(strip_valid),
            ))

    _configure_despotic_home()
    started = time.perf_counter()
    with tqdm_joblib(tqdm(total=len(tasks), desc="CO(2-1) strips", unit="strip")):
        results = Parallel(n_jobs=workers)(tasks)

    validation_pairs: list[tuple[float, float]] = []
    failures: list[str] = []
    for row_idx, col_idx, strip_fields, strip_pairs, strip_failures in results:
        for field, values in strip_fields.items():
            line_fields[field][row_idx, col_idx] = values
        validation_pairs.extend(strip_pairs)
        failures.extend(strip_failures)

    finite_freq = line_fields["freq"][np.isfinite(line_fields["freq"])]
    if finite_freq.size == 0:
        raise RuntimeError("CO(2-1) produced no finite frequencies")
    frequency = float(np.median(finite_freq))
    if not np.allclose(finite_freq, frequency, rtol=1.0e-12, atol=0.0):
        raise RuntimeError("CO(2-1) frequency is not constant across the table")
    if not np.isclose(frequency, CO21_FREQUENCY_HZ, rtol=1.0e-6):
        raise RuntimeError(
            f"unexpected CO(2-1) frequency {frequency:.9e} Hz; "
            f"expected {CO21_FREQUENCY_HZ:.9e} Hz"
        )
    # Frequency is a transition constant rather than a solved cell property.
    line_fields["freq"] = np.full(shape, frequency, dtype=float)

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(
            f"CO(2-1) failed at {len(failures)} cells that were valid in the source table:\n{preview}"
        )

    pairs = np.asarray(validation_pairs, dtype=float)
    if pairs.size == 0:
        raise RuntimeError("CO(2-1) augmentation found no valid source cells")
    expected_co10 = pairs[:, 0]
    actual_co10 = pairs[:, 1]
    absolute_errors = np.abs(actual_co10 - expected_co10)
    relative_errors = absolute_errors / np.maximum(np.abs(expected_co10), 1.0e-300)
    reference_scale = float(np.max(np.abs(expected_co10)))
    bright_threshold = CO10_REPLAY_BRIGHT_FRACTION * reference_scale
    faint_absolute_tolerance = CO10_REPLAY_FAINT_ATOL_FRACTION * reference_scale
    bright = np.abs(expected_co10) >= bright_threshold
    failing_validation = (
        (bright & (relative_errors > CO10_REPLAY_RTOL))
        | (~bright & (absolute_errors > faint_absolute_tolerance))
    )
    if failing_validation.any():
        normalized_error = np.where(
            bright,
            relative_errors / CO10_REPLAY_RTOL,
            absolute_errors / faint_absolute_tolerance,
        )
        worst = int(np.argmax(normalized_error))
        raise RuntimeError(
            "reconstructed CO(1-0) does not match the source table: "
            f"{int(failing_validation.sum())} cells exceed the scale-aware "
            f"replay tolerance; worst source={expected_co10[worst]:.3e}, "
            f"replay={actual_co10[worst]:.3e}"
        )
    bright_relative_errors = relative_errors[bright]
    print(
        "[CO21] reconstructed CO(1-0) relative error: "
        f"median={np.median(relative_errors):.3e}, "
        f"p99={np.percentile(relative_errors, 99):.3e}, "
        f"max={np.max(relative_errors):.3e}; bright-line max="
        f"{np.max(bright_relative_errors):.3e} "
        f"(threshold={bright_threshold:.3e}, rtol={CO10_REPLAY_RTOL:.1e})"
    )
    print(
        f"[CO21] solved {pairs.shape[0]}/{valid.size} valid cells in "
        f"{time.perf_counter() - started:.1f} s"
    )

    raw_augmented = _with_co21(raw, line_fields)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    save_table(raw_augmented, raw_output)

    log_axes = (
        np.log10(raw.nH_values),
        np.log10(raw.col_density_values),
        np.log10(raw.dVdr_values),
    )
    clean_fields = {
        field: (
            np.full(shape, frequency, dtype=float)
            if field == "freq"
            else _fill_in_hull(values, log_axes)
        )
        for field, values in line_fields.items()
    }
    clean_augmented = _with_co21(clean, clean_fields)
    clean_output.parent.mkdir(parents=True, exist_ok=True)
    save_table(clean_augmented, clean_output)

    remaining = {
        field: int(np.count_nonzero(~np.isfinite(values)))
        for field, values in clean_fields.items()
    }
    print(f"[CO21] clean-table non-finite counts: {remaining}")
    print(f"[CO21] raw   -> {raw_output}")
    print(f"[CO21] clean -> {clean_output}")
    return raw_augmented, clean_augmented


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source", type=Path, default=DEFAULT_RAW_SOURCE)
    parser.add_argument("--clean-source", type=Path, default=DEFAULT_CLEAN_SOURCE)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    raw_source = args.raw_source.expanduser().resolve()
    clean_source = args.clean_source.expanduser().resolve()
    raw_output = args.raw_output.expanduser().resolve()
    clean_output = args.clean_output.expanduser().resolve()
    if raw_output == raw_source or clean_output == clean_source:
        raise SystemExit("Refusing to overwrite either source table in place")
    existing = [path for path in (raw_output, clean_output) if path.exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing output(s):\n{joined}\nPass --force to replace them.")
    augment_tables(
        raw_source,
        clean_source,
        raw_output,
        clean_output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
