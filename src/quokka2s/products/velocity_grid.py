"""Shared physical velocity channels for spectra and gas-phase products."""

import numpy as np


VELOCITY_RANGE_KMS = 200.0
VELOCITY_CHANNELS = 400


def velocity_channel_edges_kms():
    """Return fresh float64 boundaries [Nchannel+1] in km/s.

    Gaussian preparation and product accumulation must use this same grid.
    These physical boundaries are independent of the standardized Gaussian
    table, whose dimensionless coordinate is z=(v-mu)/sigma.
    """
    return np.linspace(
        start=-VELOCITY_RANGE_KMS,
        stop=VELOCITY_RANGE_KMS,
        num=VELOCITY_CHANNELS + 1,
        dtype=np.float64,
    )
