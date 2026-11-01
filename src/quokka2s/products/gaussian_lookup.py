"""Shared one-dimensional linear erf table for optional spectral integration.

This processing-only module is imported once under Python's module lock. Its
immutable 32769-node table uses odd erf symmetry and the original +/-8 argument
saturation, with no additional tail cutoff or finite-window renormalization.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

LINEAR_ERF_ARGUMENT_STEP = 1.0 / 4096.0
LINEAR_ERF_ARGUMENT_MAXIMUM = 8.0
LINEAR_ERF_NODE_COUNT = 32769
LINEAR_ERF_VALUES = erf(
    np.arange(LINEAR_ERF_NODE_COUNT, dtype=float) * LINEAR_ERF_ARGUMENT_STEP
)
LINEAR_ERF_VALUES.flags.writeable = False
