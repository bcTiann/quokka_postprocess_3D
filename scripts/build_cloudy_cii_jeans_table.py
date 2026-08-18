"""Build a baseline Jeans-length CII lookup table from CIAOLoop output.

The independent axes are log10(n_H/cm^-3) and log10(T/K).  Jeans length,
its 100 pc cap, and the resulting hydrogen column are stored as derived QA
arrays, not as interpolation coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


N_DENSITY = 10
N_TEMPERATURE = 21
T_MIN_K = 3.6
T_MAX_K = 1.0e9
JEANS_CAP_CM = 3.086e20
ZERO_LIMIT = -90.0
TOLERANCE_DEX = 5.1e-4
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
HDEN_RE = re.compile(r"^#\s*hden\s+(.+?)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(path: Path) -> tuple[float, dict[float, float | None]]:
    log_nh = None
    header = None
    values: dict[float, float | None] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        match = HDEN_RE.match(raw)
        if match:
            log_nh = float(match.group(1))
            continue
        if raw.startswith("#Te"):
            header = tuple(raw.split()[1:])
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split()
        if len(columns) not in (1, 2):
            raise ValueError(f"bad data row at {path}:{line_number}: {raw!r}")
        log_t = float(columns[0])
        if log_t in values:
            raise ValueError(f"duplicate log(T)={log_t}: {path}")
        values[log_t] = float(columns[1]) if len(columns) == 2 else None
    if log_nh is None:
        raise ValueError(f"missing hden metadata: {path}")
    if header != ("C_2_157.636m",):
        raise ValueError(f"unexpected line header {header!r}: {path}")
    if len(values) != N_TEMPERATURE:
        raise ValueError(
            f"expected {N_TEMPERATURE} temperatures in {path}, found {len(values)}"
        )
    return log_nh, values


def _derived_geometry(log_nh: np.ndarray, log_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = 5.0 / 3.0
    pi = np.pi
    k_boltz = 1.3806488e-16
    m_h = 1.67373522381e-24
    gravitational_constant = 6.67384e-8
    constant = pi * np.sqrt(gamma * k_boltz / (gravitational_constant * m_h))
    n_h = np.power(10.0, log_nh)[:, None]
    temperature = np.power(10.0, log_t)[None, :]
    total_density = n_h * m_h / 0.76
    raw_length = constant * np.sqrt(temperature / total_density)
    used_length = np.minimum(raw_length, JEANS_CAP_CM)
    derived_log_column = np.log10(n_h * used_length)
    return used_length, derived_log_column


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=examples / "hm_2012_cii_baseline_jeans_10x21_output",
    )
    parser.add_argument(
        "--parameter-file", type=Path,
        default=examples / "hm_2012_cii_baseline_jeans_10x21.par",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data/cloudy_cii_hm2012_z0_baseline_jeans_10x21_T3p6_to1e9.npz",
    )
    parser.add_argument(
        "--failure-report", type=Path,
        default=root / "data/cloudy_cii_hm2012_z0_baseline_jeans_10x21_T3p6_to1e9_failures.json",
    )
    parser.add_argument(
        "--radiation-field",
        default="HM2012 z=0 shielded",
        help="Human-readable incident-radiation description stored in the table metadata.",
    )
    parser.add_argument(
        "--state-label", default="baseline",
        help="Human-readable physics-state label stored in the table metadata.",
    )
    parser.add_argument(
        "--composition-label",
        default="Cloudy default metals with C/H=1.6e-4",
        help="Human-readable elemental-composition description.",
    )
    parser.add_argument(
        "--cosmic-ray-h0-ionization-rate", type=float, default=0.0,
        help="Cloudy cosmic-rays-rate H0 ionization rate in s^-1.",
    )
    parser.add_argument("--helium-abundance-log10", type=float, default=np.nan)
    parser.add_argument("--carbon-abundance-log10", type=float, default=-3.795880)
    parser.add_argument("--oxygen-abundance-log10", type=float, default=np.nan)
    parser.add_argument("--silicon-abundance-log10", type=float, default=np.nan)
    parser.add_argument("--other-metals-disabled", action="store_true")
    parser.add_argument("--grain-model", default="none")
    parser.add_argument("--grain-scale", type=float, default=0.0)
    parser.add_argument("--grain-quantum-heating-enabled", action="store_true")
    parser.add_argument("--grain-dust-to-gas-mass-ratio", type=float, default=np.nan)
    parser.add_argument("--grain-av-per-nh-external", type=float, default=np.nan)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("input", "parameter_file", "output", "failure_report"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output.exists() or args.failure_report.exists()) and not args.force:
        raise FileExistsError("output exists; pass --force to replace it")
    if list(args.input.glob("*.mach")):
        raise RuntimeError("CIAOLoop jobs are still active")

    files = sorted(args.input.glob("*_run*.dat"))
    if len(files) != N_DENSITY:
        raise ValueError(f"expected {N_DENSITY} run files, found {len(files)}")
    by_id = {}
    for path in files:
        match = RUN_RE.search(path.name)
        if match is None:
            raise ValueError(f"unexpected run filename: {path}")
        by_id[int(match.group(1))] = path
    if set(by_id) != set(range(1, N_DENSITY + 1)):
        raise ValueError("run ids are not exactly 1..10")

    log_t = np.linspace(np.log10(T_MIN_K), np.log10(T_MAX_K), N_TEMPERATURE)
    parsed = [(run_id, *_parse(by_id[run_id])) for run_id in sorted(by_id)]
    log_nh = np.asarray([item[1] for item in parsed], dtype=float)
    if not np.all(np.diff(log_nh) > 0.0):
        raise ValueError("density axis is not strictly increasing in run order")
    raw = np.full((N_DENSITY, N_TEMPERATURE), np.nan)
    short_rows = []
    for density_index, (run_id, _, values) in enumerate(parsed):
        for reported_t, value in values.items():
            temperature_index = int(np.abs(log_t - reported_t).argmin())
            residual = abs(float(log_t[temperature_index]) - reported_t)
            if residual > TOLERANCE_DEX:
                raise ValueError(
                    f"off-grid log(T)={reported_t} by {residual:g} dex: {by_id[run_id]}"
                )
            if value is None:
                short_rows.append({
                    "run_id": run_id,
                    "density_index": density_index,
                    "temperature_index": temperature_index,
                    "log_nH": float(log_nh[density_index]),
                    "log_T": float(log_t[temperature_index]),
                })
            else:
                raw[density_index, temperature_index] = value

    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw <= ZERO_LIMIT)
    coefficient = np.zeros(raw.shape)
    positive = (~failure) & (~zero)
    coefficient[positive] = np.power(10.0, raw[positive])
    jeans_length, derived_log_nh_column = _derived_geometry(log_nh, log_t)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(1, dtype=np.int32),
        table_kind=np.asarray("Cloudy CII Jeans-length lookup"),
        axis_order=np.asarray("log_nH,log_T"),
        log_nH=log_nh, log_T=log_t,
        log_emissivity_per_nH2=raw,
        emissivity_per_nH2=coefficient,
        failure_mask=failure, original_failure_mask=failure.copy(),
        zero_mask=zero, interpolated_mask=np.zeros_like(failure),
        jeans_length_cm=jeans_length,
        derived_log_NH_cm2=derived_log_nh_column,
        jeans_length_cap_cm=np.asarray(JEANS_CAP_CM),
        jeans_length_mu_assumption=np.asarray(1.0),
        jeans_length_hydrogen_mass_fraction=np.asarray(0.76),
        line_key=np.asarray("cii"), line_label=np.asarray("C  2 157.636m"),
        state_label=np.asarray(args.state_label),
        uv_background=np.asarray(args.radiation_field),
        radiation_field=np.asarray(args.radiation_field),
        cloudy_version=np.asarray("17.02"),
        composition_label=np.asarray(args.composition_label),
        helium_abundance_log10=np.asarray(args.helium_abundance_log10),
        carbon_abundance_log10=np.asarray(args.carbon_abundance_log10),
        oxygen_abundance_log10=np.asarray(args.oxygen_abundance_log10),
        silicon_abundance_log10=np.asarray(args.silicon_abundance_log10),
        other_metals_disabled=np.asarray(args.other_metals_disabled),
        cosmic_ray_h0_ionization_rate_s=np.asarray(
            args.cosmic_ray_h0_ionization_rate,
        ),
        grain_model=np.asarray(args.grain_model),
        grain_scale=np.asarray(args.grain_scale),
        grain_quantum_heating_enabled=np.asarray(
            args.grain_quantum_heating_enabled,
        ),
        grain_dust_to_gas_mass_ratio=np.asarray(
            args.grain_dust_to_gas_mass_ratio,
        ),
        grain_av_per_nh_external_mag_cm2=np.asarray(
            args.grain_av_per_nh_external,
        ),
        no_h2_molecule_command=np.asarray(True),
        no_charge_transfer_command=np.asarray(True),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        out_of_bounds_policy=np.asarray("raise"),
        failed_node_policy=np.asarray("unavailable; no numerical fill"),
        parameter_file=np.asarray(str(args.parameter_file)),
        parameter_sha256=np.asarray(_sha256(args.parameter_file)),
        input_directory=np.asarray(str(args.input)),
    )
    failures = [
        {
            "density_index": int(i), "temperature_index": int(j),
            "log_nH": float(log_nh[i]), "log_T": float(log_t[j]),
            "temperature_K": float(10.0 ** log_t[j]),
        }
        for i, j in np.argwhere(failure)
    ]
    report = {
        "table": str(args.output), "shape": list(raw.shape),
        "temperature_bounds_K": [T_MIN_K, T_MAX_K],
        "temperature_spacing_dex": float(log_t[1] - log_t[0]),
        "failure_nodes": int(np.count_nonzero(failure)),
        "true_zero_nodes": int(np.count_nonzero(zero)),
        "positive_nodes": int(np.count_nonzero(positive)),
        "jeans_cap_nodes": int(np.count_nonzero(jeans_length >= JEANS_CAP_CM)),
        "derived_log_NH_range": [
            float(derived_log_nh_column.min()),
            float(derived_log_nh_column.max()),
        ],
        "short_rows": short_rows, "failures": failures,
    }
    args.failure_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
