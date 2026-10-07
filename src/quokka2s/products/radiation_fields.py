"""Prepare the historical Cloudy radiation recipes used by the paper figures.

These exports use attenuation labels 0, 19--23 and a fixed filtered ISM field,
or the separate unattenuated HM2012/ISM recipe. They are not the current
Jeans-table SED bundle and do not adopt its interpolation floor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HM12_COLUMN_LABELS = (0, 19, 20, 21, 22, 23)
RADIATION_X_MIN_EV = 7.0
RADIATION_X_MAX_RYD = 1.0e3
RADIATION_Y_DYNAMIC_RANGE_DEX = 8.0
COMPONENT_RADIATION_STEM = "cloudy_HM2012_NH0_19_23_and_attenuated_ISM_NH21_components"
COMBINED_RADIATION_STEM = "cloudy_combined_and_components_HM2012_NH0_19_23_ISM_NH21"
UNATTENUATED_RADIATION_STEM = "cloudy_unattenuated_HM2012_ISM_sum"
UNATTENUATED_CMB_RADIATION_STEM = "cloudy_unattenuated_HM2012_CMB_ISM_sum"


def positive_radiation_values(values):
    """Keep positive intensities and mark other logarithmic-plot samples NaN.

    values is an (N,) incident intensity array [erg cm^-2 s^-1]. For example,
    [0.0, 2e-6, -1e-6] becomes [NaN, 2e-6, NaN]. Raw saved values are retained
    separately, including real zeros; no positive intensity floor is applied.
    """
    return np.where(values > 0.0, values, np.nan)


def calculate_radiation_energy_coordinates():
    """Convert the established 7 eV lower bound and 8 eV guide to saved Ryd values."""
    from quokka2s.constants import EV_PER_RYD

    x_limits_Ryd = np.asarray([RADIATION_X_MIN_EV / EV_PER_RYD, RADIATION_X_MAX_RYD])
    eight_ev_marker_Ryd = np.asarray(8.0 / EV_PER_RYD)
    return x_limits_Ryd, eight_ev_marker_Ryd


def calculate_visible_intensity_limits(energy_Ryd, intensity_curves, x_limits_Ryd, positive_only):
    """Find the shared intensity range using the original recipe's selection.

    energy_Ryd and each intensity_curves entry have shape (N,). Limits span
    eight decades below the next power of ten above the visible maximum.
    positive_only=True preserves the unattenuated recipe's positive-value
    selection; the components recipe originally uses all visible samples.
    """
    x_min, x_max = x_limits_Ryd
    visible = (energy_Ryd >= x_min) & (energy_Ryd <= x_max)
    visible_maxima = []
    for values in intensity_curves:
        selected = visible
        if positive_only:
            selected = visible & (values > 0.0)
            if not np.any(selected):
                raise ValueError("no positive radiation values in requested x range")
        visible_maxima.append(float(values[selected].max()))
    visible_maximum = max(visible_maxima)
    y_max = 10.0 ** np.ceil(np.log10(visible_maximum))
    y_min = y_max / 10.0 ** RADIATION_Y_DYNAMIC_RANGE_DEX
    return np.asarray([y_min, y_max])


def prepare_component_radiation_data(payload):
    """Add logarithmic-plot curves and numerical limits to saved component arrays.

    payload contains energy_Ryd, ism_extinguish_nh21, hm12_nh{label} and
    combined_{label}HM, each shape (N,). Labels are 0, 19, 20, 21, 22 and 23.
    Returns a new dict retaining raw arrays and adding {field}_positive,
    hm12_column_labels, x_limits_Ryd, eight_ev_marker_Ryd and intensity limits.
    This also upgrades existing historical NPZ data without rereading exports.
    """
    prepared = dict(payload)
    energy = prepared["energy_Ryd"]
    x_limits_Ryd, eight_ev_marker_Ryd = calculate_radiation_energy_coordinates()
    combined_curves = tuple(prepared[f"combined_{label}HM"] for label in HM12_COLUMN_LABELS)
    intensity_limits = calculate_visible_intensity_limits(
        energy_Ryd=energy,
        intensity_curves=combined_curves,
        x_limits_Ryd=x_limits_Ryd,
        positive_only=False,
    )
    curve_keys = ["ism_extinguish_nh21"]
    for label in HM12_COLUMN_LABELS:
        curve_keys.append(f"hm12_nh{label}")
        curve_keys.append(f"combined_{label}HM")
    for key in curve_keys:
        prepared[key + "_positive"] = positive_radiation_values(prepared[key])
    prepared["hm12_column_labels"] = np.asarray(HM12_COLUMN_LABELS)
    prepared["x_limits_Ryd"] = x_limits_Ryd
    prepared["eight_ev_marker_Ryd"] = eight_ev_marker_Ryd
    prepared["intensity_limits_erg_cm2_s"] = intensity_limits
    return prepared


def prepare_unattenuated_radiation_data(payload, *, include_cmb):
    """Add ready-to-draw curves and limits to an existing unattenuated bundle.

    payload contains energy_Ryd, hm2012_with_optional_cmb, ism_unattenuated
    and hm2012_plus_ism, each shape (N,). Returns raw arrays plus each curve's
    {field}_positive and numerical x/y limits. The combined field remains the
    original linear sum; only its drawing array has nonpositive values masked.
    include_cmb records the supplied recipe in a saved scalar boolean so the
    renderer labels the actual data even when an explicit --data path is used.
    """
    prepared = dict(payload)
    energy = prepared["energy_Ryd"]
    combined = prepared["hm2012_plus_ism"]
    x_limits_Ryd, eight_ev_marker_Ryd = calculate_radiation_energy_coordinates()
    intensity_limits = calculate_visible_intensity_limits(
        energy_Ryd=energy,
        intensity_curves=(combined,),
        x_limits_Ryd=x_limits_Ryd,
        positive_only=True,
    )
    for key in ("hm2012_with_optional_cmb", "ism_unattenuated", "hm2012_plus_ism"):
        prepared[key + "_positive"] = positive_radiation_values(prepared[key])
    prepared["x_limits_Ryd"] = x_limits_Ryd
    prepared["eight_ev_marker_Ryd"] = eight_ev_marker_Ryd
    prepared["intensity_limits_erg_cm2_s"] = intensity_limits
    prepared["include_cmb"] = np.asarray(include_cmb, dtype=bool)
    return prepared


def radiation_range_report(payload):
    """Describe the prepared numerical axis limits for the saved JSON report."""
    y_min, y_max = payload["intensity_limits_erg_cm2_s"]
    x_min, x_max = payload["x_limits_Ryd"]
    return {
        "x_min_eV": RADIATION_X_MIN_EV,
        "x_min_Ryd": x_min,
        "x_max_Ryd": x_max,
        "y_max": y_max,
        "y_min": y_min,
        "y_dynamic_range_dex": RADIATION_Y_DYNAMIC_RANGE_DEX,
    }


def calculate_component_radiation_data(data_dir):
    """Read historical exports and prepare filtered-ISM plus HM2012 components.

    data_dir : str or Path
        Contains export_ism_filtered.inc, export_hm12_native.inc and
        export_hm12_extinguished_nh{19..23}.inc on the same energy mesh.
    Returns (payload, report): prepared (N,) arrays and source/recipe metadata.
    """
    from quokka2s.cloudy.incident_spectrum import read_incident_spectrum

    data_dir = Path(data_dir)
    ism_path = data_dir / "export_ism_filtered.inc"
    ism_data = read_incident_spectrum(path=ism_path, usecols=(0, 1))
    energy = ism_data[:, 0]
    ism_nh21 = ism_data[:, 1]
    payload = {"energy_Ryd": energy, "ism_extinguish_nh21": ism_nh21}
    hm12_paths = {}
    for label in HM12_COLUMN_LABELS:
        if label == 0:
            filename = "export_hm12_native.inc"
        else:
            filename = f"export_hm12_extinguished_nh{label}.inc"
        path = data_dir / filename
        data = read_incident_spectrum(path=path, usecols=(0, 1))
        if not np.array_equal(energy, data[:, 0]):
            raise ValueError(f"Cloudy energy mesh differs for {path}")
        hm12_paths[str(label)] = str(path)
        payload[f"hm12_nh{label}"] = data[:, 1]
        payload[f"combined_{label}HM"] = ism_nh21 + data[:, 1]
    prepared = prepare_component_radiation_data(payload=payload)
    report = {
        "definition": (
            "combined 0HM = extinguished ISM NH=1e21 + unattenuated HM12; "
            "combined kHM = extinguished ISM NH=1e21 + HM12 after "
            "extinguish column=k leak=0"
        ),
        "hm12_column_labels": list(HM12_COLUMN_LABELS),
        "ism_log_column_density_cm-2": 21,
        "attenuation_method": "Cloudy extinguish command (quick-test prescription)",
        "energy_marker": {"eV": 8.0, "Ryd": float(prepared["eight_ev_marker_Ryd"])},
        "inputs": {"ism_nh21": str(ism_path), "hm12": hm12_paths},
        "plot_range": radiation_range_report(payload=prepared),
    }
    return prepared, report


def calculate_unattenuated_radiation_data(data_dir, include_cmb=False):
    """Read plain ISM and native HM2012 (optionally CMB), then save their sum.

    data_dir : str or Path
        Contains export_ism_plain.inc and export_hm12_native.inc, or
        export_hm12_native_plus_cmb.inc when include_cmb=True.
    Returns (payload, report): prepared (N,) arrays and source/recipe metadata.
    """
    from quokka2s.cloudy.incident_spectrum import read_incident_spectrum

    data_dir = Path(data_dir)
    if include_cmb:
        filename = "export_hm12_native_plus_cmb.inc"
    else:
        filename = "export_hm12_native.inc"
    hm12_path = data_dir / filename
    ism_path = data_dir / "export_ism_plain.inc"
    hm12_data = read_incident_spectrum(path=hm12_path, usecols=(0, 1))
    ism_data = read_incident_spectrum(path=ism_path, usecols=(0, 1))
    if not np.array_equal(hm12_data[:, 0], ism_data[:, 0]):
        raise ValueError("HM2012 and ISM Cloudy energy meshes differ")
    payload = {
        "energy_Ryd": hm12_data[:, 0],
        "hm2012_with_optional_cmb": hm12_data[:, 1],
        "ism_unattenuated": ism_data[:, 1],
        "hm2012_plus_ism": hm12_data[:, 1] + ism_data[:, 1],
    }
    prepared = prepare_unattenuated_radiation_data(
        payload=payload,
        include_cmb=include_cmb,
    )
    report = {
        "definitions": {
            "hm2012": (
                "Cloudy 17.02 table HM12 redshift 0 plus CMB redshift 0, unattenuated"
                if include_cmb else "Cloudy 17.02 table HM12 redshift 0, unattenuated"
            ),
            "ism": "Cloudy 17.02 table ISM (Black 1987), unattenuated",
            "sum": "linear sum of the plotted Cloudy incident continua",
        },
        "inputs": {"hm2012": str(hm12_path), "ism": str(ism_path)},
        "plot_range": radiation_range_report(payload=prepared),
    }
    return prepared, report
