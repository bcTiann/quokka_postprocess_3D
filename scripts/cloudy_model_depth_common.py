"""Inputs and direct-output checks for the shared-composition depth tables."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from quokka2s.tables.abundances import GOW_ELEMENTAL_ABUNDANCES, METAL_REFERENCE_SCALE
from scripts.build_cloudy_sixline_tables import LINES, SED_DIRECTORY_NAME

PC_IN_CM = 3.0856775809623245e18  # Same yt conversion as the snapshot depth scan.
LOG_DEPTH_PC = np.arange(-0.25, 2.001, 0.25)
LOG_T = np.linspace(np.log10(3.6), 9.0, 21)
ELEMENT_SYMBOLS = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn'.split()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def composition_commands() -> list[str]:
    return [
        'abundances "default.abn"',
        f'element helium abundance {GOW_ELEMENTAL_ABUNDANCES["xHe"]:.17g} linear',
        f'metals {METAL_REFERENCE_SCALE:.17g} linear',
    ]


def common_commands() -> list[str]:
    return [
        'iterate to convergence', 'stop temperature off',
        'cosmic rays rate -16.698970', *composition_commands(), 'CMB redshift 0',
    ]


def direct_input(root: str, *, log_nH: float, log_T: float,
                 log_NH: float, log_L_pc: float) -> str:
    # Match CIAOLoop's actual six-digit temperature serialization.
    temperature = float(f'{10.0 ** log_T:.6e}')
    return '\n'.join([
        'title shared composition fixed model depth', *common_commands(),
        f'hden {log_nH:.17g}',
        f'init "{SED_DIRECTORY_NAME}/logNH{log_NH:g}.out"',
        f'radius 1e30 {PC_IN_CM * 10.0 ** log_L_pc:.17g} linear',
        f'constant temperature {temperature:.6f} K linear',
        f'save last radius outer "{root}.radius"',
        f'save last physical conditions "{root}.physical"',
        f'save last lines emissivity "{root}.lines"', *LINES, 'end of lines',
    ]) + '\n'


def reference_log_abundances(default_abn: Path) -> dict[str, float]:
    rows = [line.split() for line in default_abn.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith(('#', '*'))]
    if len(rows) != 30:
        raise ValueError('Expected the complete H--Zn default abundance pattern')
    values = np.array([float(row[1]) for row in rows])
    values[1] = GOW_ELEMENTAL_ABUNDANCES['xHe']
    values[2:] *= METAL_REFERENCE_SCALE
    return dict(zip(ELEMENT_SYMBOLS, np.log10(values).tolist()))


def inspect_direct_output(root: Path, *, log_nH: float, log_T: float,
                          log_L_pc: float, returncode: int | None,
                          expected_abundances: dict[str, float]) -> dict:
    """Check actual final-zone geometry and input state, not just exit status.

    Abundances in the main output are printed to four decimal places in dex;
    physical conditions and radii have coarser printed precision. These gates
    allow that serialization error, not a change in physical parameters.
    """
    issues = []
    output = root.with_suffix('.out').read_text(errors='replace')
    stops = re.findall(r'Calculation stopped because (.+)', output)
    notices = [line.strip() for line in output.splitlines()
               if re.match(r'^\s*[WC]-', line)]
    summaries = re.findall(r'^\s*Cloudy ends: (.+)$', output, re.M)
    failure_counts = re.findall(
        r'Failures: (\d+) thermal, (\d+) pressure, (\d+) ionization, (\d+) electron density',
        summaries[-1] if summaries else '')
    if not summaries:
        issues.append('missing final Cloudy summary')
    if failure_counts and any(int(value) for value in failure_counts[-1]):
        issues.append('Cloudy reports local convergence failures')
    if returncode != 0 or 'Cloudy exited OK' not in output[-1000:]:
        issues.append('Cloudy did not exit successfully')
    if not stops or not stops[-1].startswith('outer radius reached.'):
        issues.append('final iteration did not stop at the requested outer depth')
    if any(re.search(r'(?:did not converge|convergence fail)', line, re.I)
           for line in notices):
        issues.append('explicit final convergence warning')
    composition = {}
    match = re.search(r'Gas Phase Chemical Composition\s*\n(.*?)(?:\n\s*\n)', output, re.S)
    if match:
        composition = {symbol: float(value) for symbol, value in
                       re.findall(r'([A-Z][a-z]?)\s*:\s*(-?[0-9]+\.[0-9]+)', match.group(1))}
    if set(composition) != set(expected_abundances):
        issues.append('missing or incomplete reported H--Zn composition')
        abundance_error = None
    else:
        abundance_error = max(abs(composition[k] - value) for k, value in expected_abundances.items())
        if abundance_error > 5.1e-5:
            issues.append('reported abundance differs from shared composition')
    result = dict(valid=False, issues=issues, stop_reasons=stops, notices=notices,
                  final_summary=summaries[-1] if summaries else None,
                  local_convergence_failures=[int(v) for v in failure_counts[-1]] if failure_counts else [0,0,0,0],
                  abundance_max_error_dex=abundance_error,
                  reported_log_abundances=composition)
    try:
        radius = np.loadtxt(root.with_suffix('.radius'), ndmin=2)[-1]
        physical = np.loadtxt(root.with_suffix('.physical'), ndmin=2)[-1]
        emission = np.loadtxt(root.with_suffix('.lines'), ndmin=2)[-1]
        if not all(np.isfinite(array).all() for array in (radius, physical, emission)):
            raise ValueError('nonfinite saved final state')
        if physical[0] <= 0 or physical[2] <= 0:
            raise ValueError('nonpositive final depth or density')
        actual_depth = float(radius[2] + 0.5 * radius[3])
        requested_depth = float(PC_IN_CM * 10.0 ** log_L_pc)
        depth_error = abs(actual_depth / requested_depth - 1.0)
        if depth_error > 1e-4:
            issues.append('final outer edge differs from requested thickness')
        if abs(physical[1] / 10.0 ** log_T - 1.0) > 1e-4:
            issues.append('final temperature differs from requested constant temperature')
        if abs(physical[2] / 10.0 ** log_nH - 1.0) > 5.1e-4:
            issues.append('final hydrogen density differs from input density')
        if abs(emission[0] / physical[0] - 1.0) > 1e-4:
            issues.append('line emission and physical state do not refer to the same final zone')
        if abs(radius[2] / physical[0] - 1.0) > 1e-4:
            issues.append('radius and physical state do not refer to the same final zone')
        coefficient = emission[1:] / physical[2] ** 2
        if coefficient.size != len(LINES) or not np.isfinite(coefficient).all() or np.any(coefficient < 0):
            issues.append('missing, negative or nonfinite emission coefficients')
        result.update(actual_depth_cm=actual_depth, requested_depth_cm=requested_depth,
                      depth_relative_error=depth_error, final_temperature_K=float(physical[1]),
                      final_nH_cm3=float(physical[2]), emissivity_per_nH2=coefficient.tolist(),
                      zones=int(radius[0]))
    except (OSError, ValueError, IndexError) as exc:
        issues.append(f'incomplete saved final state: {exc}')
    result['valid'] = not issues
    return result
