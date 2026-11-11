"""Keep exact cutoff losses separate from approximate lookup flux.

The channel integration changes no cell luminosities, images, gas-phase
products, or complete Gaussian moments. Every saved account has shape
(dust, line, cold/hot), in erg/s. The cutoff applies to histogram-node Gaussians.
"""

from __future__ import annotations

import math

import numpy as np


TAIL_ARRAY = "truncated_inside_velocity_luminosity_erg_s"
RESIDUAL_ARRAY = "signed_lookup_window_residual_erg_s"
FULL_WINDOW_ARRAY = "untruncated_window_luminosity_reference_erg_s"
CLIPPED_WINDOW_ARRAY = "clipped_window_luminosity_reference_erg_s"
EXTRA_ARRAYS = frozenset((TAIL_ARRAY, RESIDUAL_ARRAY, FULL_WINDOW_ARRAY, CLIPPED_WINDOW_ARRAY))
ACCOUNTING_RTOL = 1.0e-10


def relative_to_input(values, input_light):
    """Divide like-shaped accounts; preserve any nonzero value at zero input."""
    fractions = np.divide(
        values,
        input_light,
        out=np.zeros_like(values, dtype=np.float64),
        where=input_light != 0.0,
    )
    fractions[(input_light == 0.0) & (values != 0.0)] = np.inf
    return fractions


def audit_accounting(payload, *, cutoff_sigma=5.0):
    """Validate exact losses and report the independent signed lookup error.

    No equality tolerance is relaxed for lookup errors. Instead, the exact
    window reference closes its own budget, and the lookup residual is tested
    against the actual integrated channel luminosity independently.
    """
    input_light = np.asarray(payload["input_luminosity_erg_s"])
    captured = np.asarray(payload["captured_luminosity_erg_s"])
    outside = np.asarray(payload["outside_velocity_luminosity_erg_s"])
    tail = np.asarray(payload[TAIL_ARRAY])
    residual = np.asarray(payload[RESIDUAL_ARRAY])
    full_window = np.asarray(payload[FULL_WINDOW_ARRAY])
    clipped_window = np.asarray(payload[CLIPPED_WINDOW_ARRAY])
    accounts = (input_light, captured, outside, tail, residual, full_window, clipped_window)
    if input_light.ndim != 3 or input_light.shape[0] != 2 or input_light.shape[2] != 2:
        raise ValueError("Luminosity accounts require shape (2, line_count, 2)")
    for account in accounts:
        if account.shape != input_light.shape or account.dtype != np.dtype(np.float64):
            raise ValueError("Every cutoff luminosity account must be like-shaped float64")
        if not np.isfinite(account).all():
            raise ValueError("Cutoff luminosity accounts must be finite")
    for account in (input_light, captured, outside, tail, full_window, clipped_window):
        if np.any(account < 0.0):
            raise ValueError("Only the signed lookup residual may be negative")

    tolerance = ACCOUNTING_RTOL * input_light
    if np.any(full_window > input_light + tolerance):
        raise AssertionError("Exact full-window luminosity exceeds input light")
    if np.any(clipped_window > full_window + tolerance):
        raise AssertionError("Exact clipped window exceeds the full Gaussian window")
    exact_budget = clipped_window + outside + tail - input_light
    residual_identity = captured - clipped_window - residual
    corrected_budget = captured + outside + tail - residual - input_light
    for description, difference in (
        ("Exact cutoff accounts do not conserve input", exact_budget),
        ("Lookup residual does not equal actual minus exact clipped light", residual_identity),
        ("Lookup-corrected cutoff accounts do not conserve input", corrected_budget),
    ):
        if np.any(np.abs(difference) > tolerance):
            raise AssertionError(description)
    tail_fraction = math.erfc(cutoff_sigma / math.sqrt(2.0))
    if np.any(tail > (tail_fraction + ACCOUNTING_RTOL) * input_light):
        raise AssertionError("Discarded inside-window light exceeds the analytic tail bound")
    return {
        "gaussian_cutoff_sigma": float(cutoff_sigma),
        "cutoff_applies_to": "histogram-node Gaussians",
        "renormalized": False,
        "accounting_relative_tolerance": ACCOUNTING_RTOL,
        "two_sided_gaussian_tail_fraction_bound": tail_fraction,
        "maximum_exact_accounting_relative_error": float(
            np.max(np.abs(relative_to_input(exact_budget, input_light)), initial=0.0)
        ),
        "maximum_corrected_accounting_relative_error": float(
            np.max(np.abs(relative_to_input(corrected_budget, input_light)), initial=0.0)
        ),
        "maximum_discarded_inside_fraction": float(
            np.max(relative_to_input(tail, input_light), initial=0.0)
        ),
        "maximum_absolute_lookup_window_residual_over_input": float(
            np.max(np.abs(relative_to_input(residual, input_light)), initial=0.0)
        ),
        "signed_lookup_window_residual_over_input": relative_to_input(residual, input_light).tolist(),
    }


def add_cutoff_accounts(payload, *, engine, cutoff_sigma=5.0):
    """Attach the selected engine's independent exact endpoint references."""
    input_light = payload["input_luminosity_erg_s"]
    expected_size = input_light.size
    references = []
    for attribute in ("untruncated_window_luminosity", "clipped_window_luminosity"):
        values = np.asarray(getattr(engine, attribute), dtype=np.float64)
        if values.shape != (expected_size,):
            raise ValueError(f"Engine {attribute} requires shape ({expected_size},)")
        references.append(values.reshape(input_light.shape).copy())
    full_window, clipped_window = references
    payload[FULL_WINDOW_ARRAY] = full_window
    payload[CLIPPED_WINDOW_ARRAY] = clipped_window
    payload["outside_velocity_luminosity_erg_s"] = np.maximum(0.0, input_light - full_window)
    payload[TAIL_ARRAY] = np.maximum(0.0, full_window - clipped_window)
    payload[RESIDUAL_ARRAY] = payload["captured_luminosity_erg_s"] - clipped_window
    audit = audit_accounting(payload, cutoff_sigma=cutoff_sigma)
    audit["cutoff_applies_to"] = getattr(engine, "cutoff_applies_to", "histogram-node Gaussians")
    return audit


def update_cutoff_report(report, payload, *, engine, audit):
    """Update all total and branch accounts without replacing histogram metadata."""
    input_light = payload["input_luminosity_erg_s"]
    captured = payload["captured_luminosity_erg_s"]
    outside = payload["outside_velocity_luminosity_erg_s"]
    method = str(engine.lookup_tables.method)
    if method == "standard-normal-prefix":
        node_integration = "standard-normal-prefix-shared-boundaries"
        description = "a standard-normal cumulative table with shared channel boundaries"
    elif method.upper() == "A":
        # Retained A-reference engines use their original report description.
        node_integration = "fractional-mu-sigma-template"
        description = "fractional-mu and sigma integrated-bin templates"
    elif method == "fractional-phase-log-sigma-template":
        node_integration = None
        description = "preintegrated fractional-mean/log-sigma templates"
    else:
        raise ValueError("Unsupported histogram-node integration table")
    for dust_index, dust_key in enumerate(payload["dust_state_keys"].tolist()):
        dust_report = report[dust_key]
        original_cells = getattr(engine, "coordinate_rule", None) == "original-cell"
        if original_cells:
            histogram_description = "Original-cell four-corner interpolation"
        elif getattr(engine, "coordinate_rule", None) == "sigma-scaled-layered":
            histogram_description = "Layered ln(sigma) luminosity histogram with sigma-scaled mean spacing"
        else:
            histogram_description = "Bilinear mu/ln(sigma) luminosity histogram"
        dust_report["profile"] = (
            f"{histogram_description} with {description}; "
            "node Gaussians truncated at +/-5 sigma without renormalization"
        )
        if original_cells:
            dust_report["cell_integration"] = "preintegrated-phase-log-sigma-template-native"
            dust_report.pop("node_integration", None)
        else:
            dust_report["node_integration"] = node_integration
        dust_report["gaussian_template_method"] = method
        dust_report.pop("phi_expansion_statistics", None)
        dust_report["gaussian_cutoff_sigma"] = audit["gaussian_cutoff_sigma"]
        dust_report["cutoff_renormalized"] = False
        dust_report["cutoff_applies_to"] = (
            "Stored template-corner Gaussians" if original_cells else "histogram-node Gaussians"
        )
        dust_report["cutoff_accounting"] = (
            "Input = exact clipped window + physical outside window + truncated inside window; "
            "actual captured = exact clipped window + signed lookup window residual. "
            "Complete moments retain the original cell Gaussians."
        )
        if original_cells:
            dust_report["profile"] = (
                "Original-cell luminosities multiplied by interpolated integrated-bin templates; "
                "fractional mean plus integer-bin shift; logarithmic sigma interpolation"
            )
            dust_report["cutoff_reference_applies_to"] = "Original-cell Gaussians"
            dust_report["cutoff_accounting"] = (
                "Input = exact original-cell clipped window + original-cell physical outside "
                "window + original-cell truncated inside window. Actual captured = exact "
                "original-cell clipped window + signed template residual. The residual includes "
                "template interpolation, shifted corner supports and floating reduction. "
                "Complete moments retain original-cell Gaussians."
            )
        dust_report["cutoff_accounting_audit"] = audit
        dust_report["template_expansion_statistics"] = engine.expansion_statistics
        for line_index, line_key in enumerate(payload["line_keys"].tolist()):
            line_report = dust_report["lines"][line_key]
            selections = [("total", (dust_index, line_index))]
            selections.extend(
                (regime_key, (dust_index, line_index, regime_index))
                for regime_index, regime_key in enumerate(payload["regime_keys"].tolist())
            )
            for entry_key, selection in selections:
                entry = line_report[entry_key]
                input_value = float(np.sum(input_light[selection]))
                outside_value = float(np.sum(outside[selection]))
                entry["captured_luminosity_erg_s"] = float(np.sum(captured[selection]))
                entry["outside_velocity_luminosity_erg_s"] = outside_value
                entry["outside_velocity_fraction"] = outside_value / input_value if input_value > 0.0 else 0.0
                for array_key in EXTRA_ARRAYS:
                    entry[array_key] = float(np.sum(payload[array_key][selection]))
                entry["truncated_inside_velocity_fraction"] = (
                    entry[TAIL_ARRAY] / input_value if input_value > 0.0 else 0.0
                )
                entry["signed_lookup_window_residual_over_input"] = (
                    entry[RESIDUAL_ARRAY] / input_value if input_value > 0.0 else 0.0
                )



def check_histogram_luminosity(payload, expected_luminosity):
    """Check explicit clipping/residual accounts with the original strict check."""
    from ..integrated_spectra import check_spectrum_luminosity
    audit_accounting(payload, cutoff_sigma=5.0)
    checked = dict(payload)
    checked['outside_velocity_luminosity_erg_s'] = (
        payload['outside_velocity_luminosity_erg_s']
        + payload[TAIL_ARRAY]
        - payload[RESIDUAL_ARRAY]
    )
    check_spectrum_luminosity(checked, expected_luminosity)
