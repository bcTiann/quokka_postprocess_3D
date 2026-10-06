"""Trilinear lookup for the canonical 3D GOW/LVG table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from quokka2s.despotic.table_data import DespoticTable, LINE_RESULT_FIELDS, SpeciesRecord


@dataclass(frozen=True)
class DespoticTemperatureCO:
    """Temperature and CO luminosities at one shared set of coordinates.

    Attributes
    ----------
    temperature_K : ndarray, shape S
        Gas temperature [K] from the table's tg_final field.
    co10_luminosity_per_H, co21_luminosity_per_H : ndarray, shape S
        CO(1-0) and CO(2-1) luminosities [erg/s/H nucleus].

    DespoticLookup.temperature_and_co() returns these arrays with the broadcast
    query shape S, including shape () for scalar coordinates.
    """

    temperature_K: np.ndarray
    co10_luminosity_per_H: np.ndarray
    co21_luminosity_per_H: np.ndarray


@dataclass(frozen=True)
class DespoticInterpolationCoordinates:
    """Flattened physical coordinates and the shape to restore after querying.

    Attributes
    ----------
    hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : ndarray, shape (Q,)
        Broadcast input values [cm^-3], [cm^-2], [s^-1], in their original order.
    query_shape : tuple of int
        Input broadcast shape S; for a batch of B cells this is (B,).
        Scalar inputs use (), while arrays of shape (2, 3) produce Q=6 queries.
    """

    hydrogen_density_cm3: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    query_shape: tuple[int, ...]

    def logarithmic_points(self, start, stop):
        """Return one chunk's log10 (nH, NH, dV/dr) rows, shape (stop-start, 3)."""
        return np.column_stack(
            (
                np.log10(self.hydrogen_density_cm3[start:stop]),
                np.log10(self.shielding_NH_cm2[start:stop]),
                np.log10(self.velocity_gradient_s[start:stop]),
            )
        )


def prepare_despotic_interpolation_coordinates(
    hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s,
):
    """Broadcast physical coordinates and flatten them for SciPy interpolation.

    Parameters
    ----------
    hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to S
        Physical density [cm^-3], shielding column [cm^-2] and gradient [s^-1].
        DespoticLookup query methods supply these, normally already clipped by
        DespoticCellReader.prepare_query_inputs(); this action does not clip or select cells.

    Returns
    -------
    DespoticInterpolationCoordinates
        Three flattened arrays with Q entries and the original broadcast shape S.
        Example: coordinates.logarithmic_points(0, 2) gives the first two rows
        passed to RegularGridInterpolator, in (log10 nH, log10 NH, log10 dV/dr) order.
    """
    density, column, gradient = np.broadcast_arrays(
        np.asarray(hydrogen_density_cm3, dtype=float),
        np.asarray(shielding_NH_cm2, dtype=float),
        np.asarray(velocity_gradient_s, dtype=float),
    )
    return DespoticInterpolationCoordinates(
        hydrogen_density_cm3=density.ravel(),
        shielding_NH_cm2=column.ravel(),
        velocity_gradient_s=gradient.ravel(),
        query_shape=density.shape,
    )


class DespoticLookup:
    """Interpolate table values linearly in log10 (nH, NH, dV/dr) coordinates.

    Query methods take physical coordinates: hydrogen_density_cm3 [H nuclei/cm^3], shielding_NH_cm2
    [H nuclei/cm^2], and velocity_gradient_s [s^-1]. Arrays broadcast to a common shape S;
    returned scalar-field arrays have shape S. Outside the grid, interpolation
    returns NaN. Coordinate clipping belongs to the caller.

    Parameters
    ----------
    table : DespoticTable
        Materialized table from quokka2s.despotic.table_files.load_table(), including its axes,
        thermal fields and species records. Interpolators are built immediately.

    Examples
    --------
    With a configured DESPOTIC table path::

        lookup = DespoticLookup(load_table(config.despotic_table))
        temperature = lookup.temperature(nH, NH, dvdr)
    """

    # Bound temporary log-coordinate arrays for large diagnostic query lists.
    _EVAL_CHUNK = 4_000_000

    def clip_coordinates(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ):
        """Clip physical query coordinates to the loaded table endpoints.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like
            Density [cm^-3], column [cm^-2] and gradient [s^-1], normally (B,) arrays
            from DespoticCellReader.prepare_query_inputs(). This action does not check coverage.

        Returns
        -------
        tuple of three ndarray
            Clipped coordinates in the same order, shapes and units. Example:
            query_nH, query_NH, query_gradient = lookup.clip_coordinates(nH, NH, dvdr).
            Other query methods do not clip; uncovered coordinates return NaN.
        """
        table = self.table
        query_hydrogen_density_cm3 = np.clip(
            hydrogen_density_cm3,
            table.nH_values.min(),
            table.nH_values.max(),
        )
        query_shielding_NH_cm2 = np.clip(
            shielding_NH_cm2,
            table.col_density_values.min(),
            table.col_density_values.max(),
        )
        query_velocity_gradient_s = np.clip(
            velocity_gradient_s,
            table.dVdr_values.min(),
            table.dVdr_values.max(),
        )
        return (
            query_hydrogen_density_cm3,
            query_shielding_NH_cm2,
            query_velocity_gradient_s,
        )

    def __init__(self, table: DespoticTable):
        """Keep the loaded table and build its linear field interpolators.

        Parameters
        ----------
        table : DespoticTable
            From load_table(); grids have shape (N_nH, N_NH, N_gradient).
            The same logarithmic coordinate axes serve every stored field.
        """
        self.table = table
        self._axes = (
            np.log10(table.nH_values),
            np.log10(table.col_density_values),
            np.log10(table.dVdr_values),
        )
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_records: dict[str, SpeciesRecord] = dict(table.species_data)

        self._register_thermal_fields()
        self._register_species_fields()
        self._register_energy_fields()
        self._build_temperature_and_co_interpolator()

    def _register_thermal_fields(self):
        """Make temperature, mu, heat capacity and internal energy queryable."""
        for token, values in (
            ("tg_final", self.table.tg_final),
            ("mu", self.table.mu_values),
            ("cv", self.table.cv_values),
            ("Eint", self.table.Eint_values),
        ):
            self._register_field(token=token, values=values)

    def _register_species_fields(self):
        """Make each species' abundance and stored line quantities queryable."""
        for name, record in self._species_records.items():
            self._register_field(
                token=f"species:{name}:abundance",
                values=record.abundance,
            )
            if record.line is not None:
                for field in LINE_RESULT_FIELDS:
                    self._register_field(
                        token=f"species:{name}:line:{field}",
                        values=getattr(record.line, field),
                    )
                self._register_field(
                    token=f"species:{name}:lumPerH",
                    values=record.line.lumPerH,
                )

    def _register_energy_fields(self):
        """Make any saved heating and cooling terms queryable."""
        if self.table.energy_terms:
            for name, values in self.table.energy_terms.items():
                self._register_field(
                    token=f"energy:{name}",
                    values=values,
                )

    def _build_temperature_and_co_interpolator(self):
        """Bundle T, CO(1-0) and CO(2-1) into one three-output query."""
        self._temperature_and_co_interpolator = None
        co10_record = self._species_records.get("CO")
        co21_record = self._species_records.get("CO21")
        co_records = (co10_record, co21_record)
        if not all(record is not None and record.line is not None for record in co_records):
            return

        # Last axis: 0 = temperature [K], 1/2 = CO(1-0)/(2-1) [erg/s/H].
        temperature_and_co = np.stack(
            (
                self.table.tg_final,
                co10_record.line.lumPerH,
                co21_record.line.lumPerH,
            ),
            axis=-1,
        )
        self._temperature_and_co_interpolator = RegularGridInterpolator(
            points=self._axes,
            values=temperature_and_co,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def _register_field(self, token: str, values: np.ndarray) -> None:
        """Store a linear interpolator for one 3D field under its string token."""
        self._interpolators[token] = RegularGridInterpolator(
            points=self._axes,
            values=np.asarray(values, dtype=float),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def _interpolate_query_values(self, interpolator, coordinates, output_shape=()):
        """Evaluate flattened queries in chunks, then restore their input shape.

        coordinates comes from prepare_despotic_interpolation_coordinates().
        output_shape is () for one field or (3,) for the T/CO bundle; returned
        values have shape (*coordinates.query_shape, *output_shape). No physical
        coordinate is clipped here, and uncovered queries remain NaN.
        """
        query_count = coordinates.hydrogen_density_cm3.size
        values = np.empty((query_count, *output_shape))
        for start in range(0, query_count, self._EVAL_CHUNK):
            stop = min(start + self._EVAL_CHUNK, query_count)
            points = coordinates.logarithmic_points(start=start, stop=stop)
            values[start:stop] = interpolator(points)
        return values.reshape((*coordinates.query_shape, *output_shape))

    def _interpolate_field(
        self, token, hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s,
    ):
        """Read one registered field at physical coordinates, retaining their shape."""
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' is not registered")
        coordinates = prepare_despotic_interpolation_coordinates(
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )
        return self._interpolate_query_values(
            interpolator=self._interpolators[token],
            coordinates=coordinates,
        )

    def mu(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate the dimensionless mean molecular weight.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            Values from table.mu_values, in hydrogen-mass units.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            mu = lookup.mu(nH, NH, dvdr)
        """
        return self._interpolate_field(
            token="mu",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def cv(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate the dimensionless heat capacity per H nucleus.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            table.cv_values; multiplying by k_B gives heat capacity
            [erg/K/H nucleus], following DESPOTIC composition.computeCv().

        Examples
        --------
        With lookup and cell coordinates already loaded::

            cv = lookup.cv(nH, NH, dvdr)
        """
        return self._interpolate_field(
            token="cv",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def Eint(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate the dimensionless internal-energy coefficient per H nucleus.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            table.Eint_values, stored in units of k_B*T per H nucleus by
            DESPOTIC composition.computeEint(); these are not energies in erg.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            eint_coefficient = lookup.Eint(nH, NH, dvdr)
        """
        return self._interpolate_field(
            token="Eint",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def temperature(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate the DESPOTIC gas temperature.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            Temperature [K] from table.tg_final, preserving query order.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            temperature_K = lookup.temperature(nH, NH, dvdr)
        """
        return self._interpolate_field(
            token="tg_final",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def temperature_and_co(
        self,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> DespoticTemperatureCO:
        """Sample temperature and both CO lines in one interpolation pass.

        Parameters
        ----------
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].
            CellEmissionCalculator.calculate() supplies clipped cell coordinates.

        Returns
        -------
        DespoticTemperatureCO
            Three arrays of shape S: temperature [K], CO(1-0) and CO(2-1)
            luminosities [erg/s/H nucleus], at identical coordinates.
            Both CO and CO21 line records must be present in the table.
            Out-of-bounds coordinates return NaN, as in the scalar methods.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            fields = lookup.temperature_and_co(nH, NH, dvdr)
            temperature_K = fields.temperature_K
        """
        interpolator = self._temperature_and_co_interpolator
        if interpolator is None:
            missing = []
            for name in ("CO", "CO21"):
                record = self._species_records.get(name)
                if record is None or record.line is None:
                    missing.append(name)
            raise ValueError(f"Temperature/CO lookup requires line data for {', '.join(missing)}")

        coordinates = prepare_despotic_interpolation_coordinates(
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )
        values = self._interpolate_query_values(
            interpolator=interpolator,
            coordinates=coordinates,
            output_shape=(3,),
        )
        return DespoticTemperatureCO(
            temperature_K=np.asarray(values[..., 0]),
            co10_luminosity_per_H=np.asarray(values[..., 1]),
            co21_luminosity_per_H=np.asarray(values[..., 2]),
        )

    def abundance(
        self,
        species: str,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate one species' abundance relative to hydrogen nuclei.

        Parameters
        ----------
        species : str
            SpeciesRecord key, for example 'H', 'H+', 'e-' or 'CO'.
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            Dimensionless number abundance from table.species_data[species].

        Examples
        --------
        With lookup and cell coordinates already loaded::

            electron_fraction = lookup.abundance('e-', nH, NH, dvdr)
        """
        return self._interpolate_field(
            token=f"species:{species}:abundance",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def field(
        self,
        token: str,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate a field by its registered token.

        Parameters
        ----------
        token : str
            Field key such as 'tg_final', 'species:CO:line:lumPerH', or
            'energy:<name>'; available keys depend on the loaded table.
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            Interpolated values in the selected field's original units.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            temperature_K = lookup.field('tg_final', nH, NH, dvdr)
        """
        return self._interpolate_field(
            token=token,
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def number_densities(
        self,
        species: Sequence[str],
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> dict[str, np.ndarray]:
        """Convert selected species abundances to number densities.

        Parameters
        ----------
        species : sequence of str
            SpeciesRecord keys, in the desired dictionary insertion order.
        hydrogen_density_cm3 : float or ndarray, broadcastable to shape S
            Hydrogen-nuclei density [cm^-3], multiplied by each abundance.
        shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen column [cm^-2] and LVG gradient [s^-1].

        Returns
        -------
        dict of str to ndarray, shape S
            One species number density [cm^-3] for each requested key.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            number = lookup.number_densities(('e-', 'H+', 'H'), nH, NH, dvdr)
        """
        number_densities = {}
        for name in species:
            abundance = self.abundance(
                species=name,
                hydrogen_density_cm3=hydrogen_density_cm3,
                shielding_NH_cm2=shielding_NH_cm2,
                velocity_gradient_s=velocity_gradient_s,
            )
            number_densities[name] = hydrogen_density_cm3 * abundance
        return number_densities

    def line_field(
        self,
        species: str,
        field_name: str,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> np.ndarray:
        """Interpolate one stored emission-line quantity for a species.

        Parameters
        ----------
        species : str
            Emitter key; the emission pipeline uses 'C+', 'CO' and 'CO21'.
        field_name : str
            One of LINE_RESULT_FIELDS: freq, intIntensity, intTB, lumPerH,
            tau or tauDust.
        hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : array-like, broadcastable to shape S
            Hydrogen density [cm^-3], hydrogen column [cm^-2], and gradient [s^-1].

        Returns
        -------
        ndarray, shape S
            Requested values: freq [Hz], intIntensity [erg/cm^2/s/sr],
            intTB [K km/s], lumPerH [erg/s/H nucleus], or dimensionless tau/tauDust.

        Examples
        --------
        With lookup and cell coordinates already loaded::

            co_per_H = lookup.line_field('CO', 'lumPerH', nH, NH, dvdr)
        """
        record = self._species_records.get(species)
        if record is None or record.line is None:
            raise ValueError(f"Species '{species}' has no line data")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'; expected one of {LINE_RESULT_FIELDS}")
        return self._interpolate_field(
            token=f"species:{species}:line:{field_name}",
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
            velocity_gradient_s=velocity_gradient_s,
        )

    def species_record(self, species: str) -> SpeciesRecord:
        """Return one stored species record without interpolating it.

        Parameters
        ----------
        species : str
            Key in table.species_data, such as 'CO'.

        Returns
        -------
        SpeciesRecord
            Original record with abundance and optional line-field grids of
            shape (N_nH, N_NH, N_gradient).

        Examples
        --------
        With a loaded lookup::

            co_grid = lookup.species_record('CO')
        """
        try:
            return self._species_records[species]
        except KeyError as exc:
            raise ValueError(f"Species '{species}' not found; available: {', '.join(self._species_records)}") from exc
