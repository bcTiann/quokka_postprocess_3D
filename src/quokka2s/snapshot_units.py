"""Extract ordered scalar unit factors for original snapshot field derivation."""
from __future__ import annotations

import numpy as np


def _conversion_factor(source_unit, target_unit):
    factor, offset = source_unit.get_conversion_factor(target_unit, np.dtype(np.float64))
    if offset:
        raise ValueError('Density and velocity conversions must have zero offsets')
    return float(factor)


def scalar_unit_stages(native_reader):
    """Extract the exact scalar factors used by the original unyt stages.

    Momentum/rho unit simplification may itself multiply the quotient. Keep
    that coefficient distinct from native->gas and gas->cgs conversion factors.
    """
    dataset = native_reader._dataset
    density = dataset.arr([1.0], native_reader._native_units['gasDensity'])
    density_gas = density.to(native_reader._gas_units['density'])
    density_cgs = density_gas.to('g/cm**3')
    density_to_gas = _conversion_factor(density.units, density_gas.units)
    gas_density_to_cgs = _conversion_factor(density_gas.units, density_cgs.units)
    velocity_factors = np.empty((3, 4), dtype=np.float64)
    for axis_index, axis in enumerate(('x', 'y', 'z')):
        momentum = dataset.arr([1.0], native_reader._native_units[f'{axis}-GasMomentum'])
        native_velocity = momentum / density
        gas_velocity = native_velocity.to(native_reader._gas_units[f'velocity_{axis}'])
        cgs_velocity = gas_velocity.to('cm/s')
        kms_velocity = gas_velocity.to('km/s')
        velocity_factors[axis_index, 0] = float(native_velocity.value[0])
        velocity_factors[axis_index, 1] = _conversion_factor(native_velocity.units, gas_velocity.units)
        velocity_factors[axis_index, 2] = _conversion_factor(gas_velocity.units, cgs_velocity.units)
        velocity_factors[axis_index, 3] = _conversion_factor(gas_velocity.units, kms_velocity.units)
    # Unit simplification can leave a factor one ulp away from one even when
    # the printed result already says cm**-2. Reproduce the final .to stage.
    hydrogen_probe = dataset.arr([1.0], 'cm**-3')
    cell_column_probe = hydrogen_probe * native_reader._cell_widths[2].in_cgs()
    inverse_probe = 1.0 / cell_column_probe
    harmonic_probe = 2 / (inverse_probe + inverse_probe)
    canonical_column = harmonic_probe.to('cm**-2')
    shielding_to_cgs = _conversion_factor(harmonic_probe.units, canonical_column.units)
    return density_to_gas, gas_density_to_cgs, velocity_factors, shielding_to_cgs
