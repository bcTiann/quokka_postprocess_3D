"""Streaming LOS spectra using the existing Gaussian channel integrals.

This module accepts already selected, validated volumetric line emissivities.
It does not select a chemistry model, interpolate tables, or apply transfer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import erf as scipy_erf


AMU_G = 1.66053906660e-24
SPEED_OF_LIGHT_KMS = 299792.458
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")
LINE_MASSES_AMU = {
    "cii": 12.01, "halpha": 1.00794, "hi21": 1.00794,
    "ciii_977": 12.01, "ciii_1907": 12.01, "ciii_1909": 12.01,
    "civ_1548": 12.01, "civ_1551": 12.01,
    "co10": 28.009, "co21": 28.009,
}
LINE_TITLES = {
    "cii": r"C II 158 $\mu$m", "halpha": r"H$\alpha$", "hi21": "H I 21 cm",
    "ciii_977": r"C III 977.020 $\AA$",
    "ciii_1907": r"C III] 1906.68 $\AA$",
    "ciii_1909": r"C III] 1908.73 $\AA$",
    "civ_1548": r"C IV 1548.19 $\AA$",
    "civ_1551": r"C IV 1550.78 $\AA$",
    "co10": "CO(1-0)", "co21": "CO(2-1)",
}


def accumulate_velocity_spectra(
    velocity_kms: np.ndarray,
    thermal_width_kms: np.ndarray,
    luminosity_matrix: np.ndarray,
    velocity_edges_kms: np.ndarray,
    *,
    cell_chunk: int = 32768,
    workers: int = 1,
) -> np.ndarray:
    """Accumulate dL/dv with the established analytic erf-edge algorithm.

    This preserves the kernel in
    ``scripts/plot_cloudy_line_physics_ablation_spectra.py`` (the function of
    the same name), without importing that script's simulation dependencies.
Several lines with identical widths share one evaluation of the Gaussian.
Its 1 cm/s minimum width and finite velocity window are retained exactly.
"""
    velocity = np.asarray(velocity_kms, dtype=float).reshape(-1)
    thermal = np.asarray(thermal_width_kms, dtype=float).reshape(-1)
    luminosity = np.asarray(luminosity_matrix, dtype=float)
    velocity_edges_kms = np.asarray(velocity_edges_kms, dtype=float)
    if luminosity.ndim != 2 or luminosity.shape[0] != velocity.size:
        raise ValueError("luminosity_matrix must have shape (n_cells, n_models)")
    if thermal.shape != velocity.shape:
        raise ValueError("thermal width and velocity must have matching shapes")
    if velocity_edges_kms.ndim != 1 or velocity_edges_kms.size < 2:
        raise ValueError("velocity edges must be a one-dimensional array")
    widths = np.diff(velocity_edges_kms)
    if not np.isfinite(velocity_edges_kms).all() or not np.all(widths > 0.0):
        raise ValueError("velocity edges must be finite and strictly increasing")
    if not np.allclose(widths, widths[0], rtol=1.e-12, atol=0.):
        raise ValueError("velocity edges must be uniformly spaced")
    if cell_chunk <= 0 or workers <= 0:
        raise ValueError("cell_chunk and workers must be positive")

    emitting = np.any(luminosity != 0.0, axis=1)
    velocity = velocity[emitting]
    thermal = thermal[emitting]
    luminosity = luminosity[emitting]
    output_shape = (velocity_edges_kms.size - 1, luminosity.shape[1])
    delta_v = float(velocity_edges_kms[1] - velocity_edges_kms[0])
    channel_chunk = 150

    def accumulate_chunk(cell0: int) -> np.ndarray:
        cell1 = min(cell0 + cell_chunk, velocity.size)
        centers = velocity[cell0:cell1][None, :]
        values = luminosity[cell0:cell1]
        sigma = np.maximum(thermal[cell0:cell1], 1.0e-5)[None, :]
        denominator = np.sqrt(2.0) * sigma
        partial = np.zeros(output_shape, dtype=float)
        for channel0 in range(0, output_shape[0], channel_chunk):
            channel1 = min(channel0 + channel_chunk, output_shape[0])
            edges = velocity_edges_kms[channel0:channel1 + 1, None]
            erf_edges = (edges - centers) / denominator
            scipy_erf(erf_edges, out=erf_edges)
            fractions = 0.5 * (erf_edges[1:] - erf_edges[:-1])
            partial[channel0:channel1] = np.einsum(
                "kc,cm->km", fractions, values, optimize=False,
            ) / delta_v
        return partial

    chunk_starts = range(0, velocity.size, cell_chunk)
    output = np.zeros(output_shape, dtype=float)
    if workers <= 1:
        for cell0 in chunk_starts:
            output += accumulate_chunk(cell0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for partial in executor.map(accumulate_chunk, chunk_starts):
                output += partial
    return output


class AdoptedSpectralAccumulator:
    """Accumulate line-by-regime spectra from bounded chunks of valid cells.

``add`` accepts velocity (cells,), temperature and emissivity (lines,cells),
and scalar or per-cell volume. Masks must be boolean (cells,). Omitted
``cold_mask`` places all selected cells in the hot display branch. The caller
must remove unavailable cells before calling; even unselected inputs are
validated. Zero line emissivity is a real zero, never a missing-data marker.
"""

    def __init__(self, line_keys, velocity_edges_kms, boltzmann_erg_K,
                 *, cell_chunk=32768, workers=1):
        self.line_keys = tuple(line_keys)
        if not self.line_keys or len(set(self.line_keys)) != len(self.line_keys):
            raise ValueError("line_keys must be nonempty and unique")
        unknown = set(self.line_keys) - LINE_MASSES_AMU.keys()
        if unknown:
            raise ValueError(f"No adopted thermal mass for line(s): {sorted(unknown)}")
        self.velocity_edges_kms = np.asarray(velocity_edges_kms, dtype=float).copy()
        self.boltzmann_erg_K = float(boltzmann_erg_K)
        if not np.isfinite(self.boltzmann_erg_K) or self.boltzmann_erg_K <= 0.:
            raise ValueError("boltzmann_erg_K must be finite and positive")
        if not isinstance(cell_chunk, (int, np.integer)) or not isinstance(workers, (int, np.integer)):
            raise ValueError("cell_chunk and workers must be positive integers")
        self.cell_chunk, self.workers = int(cell_chunk), int(workers)
        # Validate the channel grid through the same kernel used below.
        accumulate_velocity_spectra([], [], np.zeros((0, 1)), self.velocity_edges_kms,
                                    cell_chunk=self.cell_chunk, workers=self.workers)
        shape = (len(self.line_keys), 2, self.velocity_edges_kms.size - 1)
        self.dL_dv = np.zeros(shape)
        self.input_luminosity = np.zeros(shape[:2])
        self.cell_counts = np.zeros(2, dtype=np.int64)
        self.masses_g = np.array([LINE_MASSES_AMU[key] for key in self.line_keys]) * AMU_G

    @staticmethod
    def _mask(value, size, name, default):
        if value is None:
            return np.full(size, default, dtype=bool)
        result = np.asarray(value)
        if result.dtype != np.bool_ or result.shape != (size,):
            raise ValueError(f"{name} must be boolean with shape (cells,)")
        return result

    def add(self, velocity_kms, temperatures_K, emissivity, cell_volume_cm3,
            *, selected_mask=None, cold_mask=None):
        """Add luminosity epsilon times volume; return self for streaming use."""
        velocity = np.asarray(velocity_kms, dtype=float)
        temperature = np.asarray(temperatures_K, dtype=float)
        epsilon = np.asarray(emissivity, dtype=float)
        if velocity.ndim != 1 or not np.isfinite(velocity).all():
            raise ValueError("velocity_kms must be finite with shape (cells,)")
        if np.any(np.abs(velocity) >= SPEED_OF_LIGHT_KMS):
            raise ValueError("The adopted nonrelativistic profile requires abs(velocity) < c")
        expected = (len(self.line_keys), velocity.size)
        if temperature.shape != expected or epsilon.shape != expected:
            raise ValueError("temperatures_K and emissivity must have shape (lines,cells)")
        if not np.isfinite(temperature).all() or np.any(temperature < 0.):
            raise ValueError("temperatures_K must be finite and nonnegative")
        if not np.isfinite(epsilon).all() or np.any(epsilon < 0.):
            raise ValueError("emissivity must be finite and nonnegative")
        volume = np.asarray(cell_volume_cm3, dtype=float)
        if volume.shape not in ((), velocity.shape) or not np.isfinite(volume).all() or np.any(volume <= 0.):
            raise ValueError("cell_volume_cm3 must be finite positive, scalar or shape (cells,)")
        selected = self._mask(selected_mask, velocity.size, "selected_mask", True)
        cold = self._mask(cold_mask, velocity.size, "cold_mask", False)
        with np.errstate(over="ignore", invalid="ignore"):
            luminosity = epsilon * volume
        if not np.isfinite(luminosity).all():
            raise ValueError("emissivity times volume must remain finite")

        for branch, take in enumerate((selected & cold, selected & ~cold)):
            if not np.any(take):
                continue
            branch_luminosity = luminosity[:, take]
            branch_temperature = temperature[:, take]
            branch_velocity = velocity[take]
            partial = np.zeros_like(self.dL_dv[:, branch])
            remaining = list(np.flatnonzero(np.any(branch_luminosity != 0., axis=1)))
            while remaining:
                first = remaining[0]
                shared = [idx for idx in remaining
                          if self.masses_g[idx] == self.masses_g[first]
                          and np.array_equal(branch_temperature[idx], branch_temperature[first])]
                remaining = [idx for idx in remaining if idx not in shared]
                thermal = np.sqrt(self.boltzmann_erg_K * branch_temperature[first]
                                  / self.masses_g[first]) / 1.e5
                thermal *= 1. - branch_velocity / SPEED_OF_LIGHT_KMS
                partial[shared] = accumulate_velocity_spectra(
                    branch_velocity, thermal, branch_luminosity[shared].T,
                    self.velocity_edges_kms, cell_chunk=self.cell_chunk,
                    workers=self.workers,
                ).T
            self.dL_dv[:, branch] += partial
            self.input_luminosity[:, branch] += branch_luminosity.sum(axis=1)
            self.cell_counts[branch] += np.count_nonzero(take)
        return self

    def finalize(self):
        """Return an NPZ-ready dictionary and a JSON-ready luminosity report.

Spectra and luminosities have a cold/hot regime axis; the payload also stores
the sum over both branches. Values outside the finite velocity window are
reported as missing luminosity and are never redistributed into its edges.
"""
        widths = np.diff(self.velocity_edges_kms)
        captured = np.sum(self.dL_dv * widths, axis=-1)
        outside = np.maximum(0., self.input_luminosity - captured)
        outside_fraction = np.divide(outside, self.input_luminosity,
                                     out=np.zeros_like(outside), where=self.input_luminosity > 0.)
        total_input, total_captured = self.input_luminosity.sum(axis=1), captured.sum(axis=1)
        total_outside = np.maximum(0., total_input - total_captured)
        total_outside_fraction = np.divide(total_outside, total_input,
                                          out=np.zeros_like(total_input), where=total_input > 0.)
        payload = {
            "line_keys": np.asarray(self.line_keys), "regime_keys": np.asarray(REGIME_KEYS),
            "axis_order": np.asarray("line,regime,velocity_channel"),
            "velocity_edges_kms": self.velocity_edges_kms.copy(),
            "velocity_kms": .5 * (self.velocity_edges_kms[:-1] + self.velocity_edges_kms[1:]),
            "dL_dv_erg_s_per_kms": self.dL_dv.copy(),
            "total_dL_dv_erg_s_per_kms": self.dL_dv.sum(axis=1),
            "input_luminosity_erg_s": self.input_luminosity.copy(),
            "captured_luminosity_erg_s": captured,
            "outside_velocity_luminosity_erg_s": outside,
            "outside_velocity_fraction": outside_fraction,
            "total_input_luminosity_erg_s": total_input,
            "total_captured_luminosity_erg_s": total_captured,
            "total_outside_velocity_fraction": total_outside_fraction,
            "cell_counts_by_regime": self.cell_counts.copy(),
            "boltzmann_erg_K": np.asarray(self.boltzmann_erg_K),
            "line_masses_g": self.masses_g.copy(),
            "thermal_width_convention": np.asarray("sqrt(k_B*T/m)/1e5*(1-v/299792.458); floor=1e-5 km/s"),
        }
        report = {
            "line_keys": list(self.line_keys), "regime_keys": list(REGIME_KEYS),
            "axis_order": "line,regime,velocity_channel",
            "velocity_range_kms": self.velocity_edges_kms[[0, -1]].tolist(),
            "velocity_channels": int(widths.size),
            "cell_counts_by_regime": self.cell_counts.tolist(),
            "profile": "analytic Gaussian integral over channel edges; existing Doppler width convention",
            "transfer": "none added; supplied emissivities are summed as epsilon times cell volume",
            "lines": {},
        }
        for idx, key in enumerate(self.line_keys):
            def record(input_value, captured_value, outside_value, fraction):
                return {"input_luminosity_erg_s": float(input_value),
                        "captured_luminosity_erg_s": float(captured_value),
                        "outside_velocity_luminosity_erg_s": float(outside_value),
                        "outside_velocity_fraction": float(fraction)}
            report["lines"][key] = {
                "total": record(total_input[idx], total_captured[idx], total_outside[idx], total_outside_fraction[idx]),
                **{regime: record(self.input_luminosity[idx, branch], captured[idx, branch],
                                  outside[idx, branch], outside_fraction[idx, branch])
                   for branch, regime in enumerate(REGIME_KEYS)},
            }
        return payload, report


def plot_adopted_spectra(payload, output_stem, *, projected_area_cm2=None,
                         title="Adopted line spectra, LOS z"):
    """Write a PNG and PDF of the total and cold/hot spectral components."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = tuple(str(key) for key in payload["line_keys"])
    velocity = np.asarray(payload["velocity_kms"])
    spectra = np.asarray(payload["dL_dv_erg_s_per_kms"])
    if spectra.shape != (len(keys), 2, velocity.size):
        raise ValueError("Plot spectra must have shape (line,regime,velocity_channel)")
    if projected_area_cm2 is not None:
        area = float(projected_area_cm2)
        if not np.isfinite(area) or area <= 0.:
            raise ValueError("projected_area_cm2 must be finite and positive")
        spectra = spectra / area
        ylabel = r"$d\Sigma_L/dv$ [erg s$^{-1}$ cm$^{-2}$ (km s$^{-1}$)$^{-1}$]"
    else:
        ylabel = r"$dL/dv$ [erg s$^{-1}$ (km s$^{-1}$)$^{-1}$]"
    rows = (len(keys) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(12, 2.6 * rows), squeeze=False, sharex=True)
    for idx, (key, axis) in enumerate(zip(keys, axes.flat)):
        total = spectra[idx].sum(axis=0)
        for values, label, color, style in (
            (total, "Total", "#161616", "-"),
            (spectra[idx, 0], r"$T_{\rm QUOKKA}<3000$ K", "#2467A6", "--"),
            (spectra[idx, 1], r"$T_{\rm QUOKKA}\geq3000$ K", "#C55529", ":"),
        ):
            axis.plot(velocity, values, label=label, color=color, ls=style, lw=1.3)
        axis.set_title(LINE_TITLES.get(key, key), fontsize=11)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.set_ylim(bottom=0.)
        axis.grid(alpha=.18)
        axis.set_xlim(float(payload["velocity_edges_kms"][0]), float(payload["velocity_edges_kms"][-1]))
    for axis in list(axes.flat)[len(keys):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel(r"LOS velocity [km s$^{-1}$]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    figure.supylabel(ylabel, fontsize=11)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(.025, 0., 1., .98))
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {extension: stem.with_suffix("." + extension) for extension in ("png", "pdf")}
    for path in paths.values():
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths
