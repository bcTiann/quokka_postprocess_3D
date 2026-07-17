"""Build the CHIANTI [C II] 158 um upper-level population lookup table.

Run from the repository root with the project's conda environment:

    conda run -n quokka python scripts/build_cii_chianti_nu_table.py

CHIANTI supplies H and He CIE ion fractions.  The project composition
(X=0.74, Y=0.26) and each table n_H then set explicit electron and
proton densities through charge neutrality.  No fiasco-internal composition
or proton/electron ratio is used for the level-population solve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import fiasco
import numpy as np

from quokka2s.cie_composition import cie_h_he_charge_per_hydrogen
from quokka2s.pipeline.prep.config import X_H, Y_HE


class ExplicitProtonDensityIon(fiasco.Ion):
    """fiasco Ion whose proton/electron ratio is supplied explicitly.

    fiasco 0.6.x otherwise derives this ratio from named abundance and
    ionization-equilibrium datasets.  Overriding the property lets the normal
    CHIANTI rate-matrix solver use simulation-consistent n_p and n_e.
    """

    def set_proton_electron_ratio(self, ratio) -> None:
        ratio = u.Quantity(ratio, u.dimensionless_unscaled)
        if ratio.shape != self.temperature.shape:
            raise ValueError('Explicit n_p/n_e ratio must match the temperature grid')
        self._explicit_proton_electron_ratio = ratio

    @property
    def proton_electron_ratio(self):
        try:
            return self._explicit_proton_electron_ratio
        except AttributeError as exc:
            raise RuntimeError('Explicit n_p/n_e ratio has not been set') from exc


def build_table(output: Path, n_temperature: int, n_density: int) -> None:
    log_temperature = np.linspace(np.log10(1.307e4), 8.5, n_temperature)
    log_hydrogen_density = np.linspace(-12.0, 8.0, n_density)
    temperature = np.power(10.0, log_temperature) * u.K
    hydrogen_density = np.power(10.0, log_hydrogen_density) / u.cm**3

    x_h = np.nan_to_num(np.asarray(
        fiasco.Element('hydrogen', temperature).equilibrium_ionization
    ))
    x_he = np.nan_to_num(np.asarray(
        fiasco.Element('helium', temperature).equilibrium_ionization
    ))
    electron_per_h, proton_per_h = cie_h_he_charge_per_hydrogen(
        x_h,
        x_he,
        hydrogen_mass_fraction=X_H,
        helium_mass_fraction=Y_HE,
    )

    cii = ExplicitProtonDensityIon('C 2', temperature)
    cii.set_proton_electron_ratio(proton_per_h / electron_per_h)

    level_two = np.where(cii.levels.level == 2)[0]
    if level_two.size != 1:
        raise RuntimeError(f'Expected one CHIANTI C II level 2; found {level_two.size}')
    upper_fraction = np.empty((n_temperature, n_density), dtype=float)
    for j, n_h in enumerate(hydrogen_density):
        electron_density = electron_per_h * n_h
        population = cii.level_populations(
            electron_density,
            couple_density_to_temperature=True,
            include_protons=True,
            use_two_ion_model=False,
            include_level_resolved_rate_correction=False,
        )
        upper_fraction[:, j] = np.asarray(
            population[:, 0, int(level_two[0])], dtype=float
        )

    if not np.all(np.isfinite(upper_fraction)):
        raise RuntimeError('CHIANTI returned non-finite C II upper-level populations')
    if np.any((upper_fraction < 0.0) | (upper_fraction > 1.0)):
        raise RuntimeError('CHIANTI returned a C II upper-level population outside [0, 1]')

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        log10_temperature_K=log_temperature,
        log10_hydrogen_density_cm3=log_hydrogen_density,
        log10_upper_fraction=np.log10(np.maximum(upper_fraction, 1e-300)),
        electron_per_hydrogen=electron_per_h,
        proton_per_hydrogen=proton_per_h,
        hydrogen_mass_fraction=np.array(X_H),
        helium_mass_fraction=np.array(Y_HE),
        collider_elements=np.array('H+He'),
        explicit_proton_density=np.array(True),
        include_protons=np.array(True),
        use_two_ion_model=np.array(False),
        include_level_resolved_rate_correction=np.array(False),
        ion=np.array('C 2'),
        upper_level=np.array(2),
        fiasco_version=np.array(fiasco.__version__),
    )
    print(f'Wrote {output} with shape {upper_fraction.shape}')


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=repo_root / 'data' / 'cii_chianti_nu_cie_v3.npz',
    )
    parser.add_argument('--n-temperature', type=int, default=260)
    parser.add_argument('--n-density', type=int, default=201)
    args = parser.parse_args()
    build_table(args.output, args.n_temperature, args.n_density)


if __name__ == '__main__':
    main()
