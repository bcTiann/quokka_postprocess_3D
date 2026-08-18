"""Unit-safe conversions and labels for integrated emission-line spectra."""
from __future__ import annotations

import numpy as np
from yt.units.physical_constants import speed_of_light
from yt.units.yt_array import YTArray, YTQuantity


# Keep the internal frequency-density unit explicit at the point where the
# numpy channel accumulator hands its result back to the unit-aware pipeline.
DSIGMA_DNU_CGS_UNIT = "erg/s/Hz/cm**2"

# Canonical stored/output units for spectra whose horizontal axis is velocity.
# Keep the surface-brightness and velocity factors separate so plot labels can
# retain the observational form erg s^-1 cm^-2 (km s^-1)^-1.  yt validates and
# converts both factors; no conversion constant or final unit label is typed by
# hand.
SURFACE_BRIGHTNESS_UNIT = "erg/s/cm**2"
SPECTRAL_VELOCITY_UNIT = "km/s"
DSIGMA_DV_UNIT = f"{SURFACE_BRIGHTNESS_UNIT}/({SPECTRAL_VELOCITY_UNIT})"

# The numerical spectrum kernel needs c as a bare cgs float.  Derive it from
# yt's physical constant so the unit conversion is checked before units are
# deliberately stripped at the numpy boundary.
SPEED_OF_LIGHT_CGS = speed_of_light.to("cm/s")


def convert_dsigma_dnu_to_dsigma_dv(
    dsigma_dnu_values: np.ndarray | YTArray,
    rest_frequency_hz: float,
) -> YTArray:
    """Convert dSigma_L/dnu to dSigma_L/dv using a unit-checked Jacobian.

    ``v`` is measured in km/s, so the Jacobian is
    ``|dnu/dv| = nu_0 / c`` with ``c`` explicitly converted to km/s.  The final
    ``.to(DSIGMA_DV_UNIT)`` rejects any dimensional inconsistency.
    """
    if hasattr(dsigma_dnu_values, "units"):
        dsigma_dnu = dsigma_dnu_values.to(DSIGMA_DNU_CGS_UNIT)
    else:
        dsigma_dnu = YTArray(dsigma_dnu_values, DSIGMA_DNU_CGS_UNIT)
    nu_0 = YTQuantity(rest_frequency_hz, "Hz")
    dnu_dv = (nu_0 / speed_of_light.to("km/s")).to("Hz/(km/s)")
    return (dsigma_dnu * dnu_dv).to(DSIGMA_DV_UNIT)


def unit_latex(unit: str | bytes) -> str:
    """Return yt's LaTeX representation after validating ``unit``."""
    if isinstance(unit, bytes):
        unit = unit.decode()
    return YTQuantity(1.0, str(unit)).units.latex_representation()


def dsigma_dv_ylabel(unit: str | bytes = DSIGMA_DV_UNIT) -> str:
    """Axis label containing only unit factors rendered and validated by yt."""
    # Validate the complete spectrum unit even though yt algebraically cancels
    # the two seconds.  Rendering the two validated factors separately keeps
    # the conventional observational notation used in Huang et al. (2025).
    YTQuantity(1.0, str(unit))
    energy = unit_latex("erg")
    time = unit_latex("s")
    length = unit_latex("cm")
    velocity = unit_latex(SPECTRAL_VELOCITY_UNIT)
    return (
        rf"$\left[{energy}\,{time}^{{-1}}\,{length}^{{-2}}\,"
        rf"\left({velocity}\right)^{{-1}}\right]$"
    )
