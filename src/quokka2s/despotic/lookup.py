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

    DespoticLookup.temperature_and_co(queries=...) returns arrays with the
    prepared query shape S, including shape () for scalar coordinates.
    """

    temperature_K: np.ndarray
    co10_luminosity_per_H: np.ndarray
    co21_luminosity_per_H: np.ndarray


@dataclass(frozen=True)
class DespoticQueries:
    """Batch-local physical coordinates and reusable log10 interpolation rows.

    density/column/gradient have Q flattened entries [cm^-3], [cm^-2], [s^-1].
    query_shape restores the input broadcast shape S. Small query lists cache
    points (Q, 3); larger diagnostic lists form only one bounded chunk at a time.
    No lookup or reader retains this object after its caller releases it.
    """

    hydrogen_density_cm3: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    query_shape: tuple[int, ...]
    _logarithmic_points: np.ndarray | None

    def logarithmic_points(self, start: int, stop: int) -> np.ndarray:
        """Return log10 nH/NH/dVdr rows (stop-start, 3) for one query chunk."""
        if self._logarithmic_points is not None:
            return self._logarithmic_points[start:stop]
        return np.column_stack(
            (
                np.log10(self.hydrogen_density_cm3[start:stop]),
                np.log10(self.shielding_NH_cm2[start:stop]),
                np.log10(self.velocity_gradient_s[start:stop]),
            )
        )

    def select(self, selected_cells: np.ndarray) -> DespoticQueries:
        """Select Q queries with an original-shape Boolean mask, keeping order.

        For selected_cells=[True, False, True], output rows describe cells 0
        and 2, with query_shape=(2,). Cached logarithms are selected, not recomputed.
        """
        selected = selected_cells.ravel()
        density = self.hydrogen_density_cm3[selected]
        points = None
        if self._logarithmic_points is not None:
            points = self._logarithmic_points[selected]
        return DespoticQueries(
            hydrogen_density_cm3=density,
            shielding_NH_cm2=self.shielding_NH_cm2[selected],
            velocity_gradient_s=self.velocity_gradient_s[selected],
            query_shape=density.shape,
            _logarithmic_points=points,
        )


class DespoticLookup:
    """Interpolate table values linearly in log10 (nH, NH, dV/dr) coordinates.

    prepare_queries() broadcasts physical nH/NH/dVdr into shape S. Every field
    method takes that same batch-local query object and returns shape S. Each
    field keeps its own SciPy interpolator and NaN support. Outside the table,
    queries return NaN; coordinate clipping belongs to the caller.

    Parameters
    ----------
    table : DespoticTable
        Materialized table from quokka2s.despotic.table_files.load_table(), including its axes,
        thermal fields and species records. Interpolators are built immediately.

    Examples
    --------
    With a configured DESPOTIC table path::

        lookup = DespoticLookup(load_table(config.despotic_table))
        queries = lookup.prepare_queries(
            hydrogen_density_cm3=nH,
            shielding_NH_cm2=NH,
            velocity_gradient_s=dvdr,
        )
        temperature = lookup.temperature(queries=queries)
    """

    # Bound temporary log-coordinate arrays for large diagnostic query lists.
    _EVAL_CHUNK = 4_000_000

    def prepare_queries(
        self,
        *,
        hydrogen_density_cm3,
        shielding_NH_cm2,
        velocity_gradient_s,
    ) -> DespoticQueries:
        """Broadcast physical coordinates and prepare shared interpolation rows.

        Inputs are broadcastable to S, in cm^-3, cm^-2 and s^-1. Cell readers
        supply checked, clipped coordinates; diagnostic callers may query
        uncovered values. This method does not clip or reject those values.

        Return batch-local DespoticQueries. Lists up to _EVAL_CHUNK cache their
        log10 rows once; larger lists retain physical views and create bounded
        log chunks during evaluation. Example: scalar NH/dVdr and three nH
        values produce query_shape=(3,), reused by temperature and species queries.
        """
        density, column, gradient = np.broadcast_arrays(
            np.asarray(hydrogen_density_cm3, dtype=float),
            np.asarray(shielding_NH_cm2, dtype=float),
            np.asarray(velocity_gradient_s, dtype=float),
        )
        query_shape = density.shape
        density = density.ravel()
        column = column.ravel()
        gradient = gradient.ravel()
        points = None
        if density.size <= self._EVAL_CHUNK:
            points = np.column_stack(
                (
                    np.log10(density),
                    np.log10(column),
                    np.log10(gradient),
                )
            )
        return DespoticQueries(
            hydrogen_density_cm3=density,
            shielding_NH_cm2=column,
            velocity_gradient_s=gradient,
            query_shape=query_shape,
            _logarithmic_points=points,
        )

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

        coordinates comes from prepare_queries().
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

    def _interpolate_field(self, *, token: str, queries: DespoticQueries) -> np.ndarray:
        """Read one registered scalar field with the prepared broadcast shape S."""
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' is not registered")
        return self._interpolate_query_values(
            interpolator=self._interpolators[token],
            coordinates=queries,
        )

    def mu(self, *, queries: DespoticQueries) -> np.ndarray:
        """Return mean molecular weight with shape S, in hydrogen-mass units.

        queries comes from prepare_queries(). Example: lookup.mu(queries=queries).
        """
        return self._interpolate_field(
            token="mu",
            queries=queries,
        )

    def cv(self, *, queries: DespoticQueries) -> np.ndarray:
        """Return heat capacity with shape S, in k_B per H nucleus.

        Multiplying by k_B gives erg/K/H, following composition.computeCv().
        """
        return self._interpolate_field(
            token="cv",
            queries=queries,
        )

    def Eint(self, *, queries: DespoticQueries) -> np.ndarray:
        """Return internal-energy coefficient with shape S, in k_B*T per H."""
        return self._interpolate_field(
            token="Eint",
            queries=queries,
        )

    def temperature(self, *, queries: DespoticQueries) -> np.ndarray:
        """Return gas temperature with shape S [K] from the table's tg_final."""
        return self._interpolate_field(
            token="tg_final",
            queries=queries,
        )

    def temperature_and_co(self, *, queries: DespoticQueries) -> DespoticTemperatureCO:
        """Read T [K] and CO(1-0)/(2-1) [erg/s/H] at the same prepared queries.

        Returns DespoticTemperatureCO, with each array retaining shape S.
        Uses the existing three-output RGI; missing support remains per field.
        Example: fields = lookup.temperature_and_co(queries=queries).
        """
        interpolator = self._temperature_and_co_interpolator
        if interpolator is None:
            missing = []
            for name in ("CO", "CO21"):
                record = self._species_records.get(name)
                if record is None or record.line is None:
                    missing.append(name)
            raise ValueError(f"Temperature/CO lookup requires line data for {', '.join(missing)}")

        values = self._interpolate_query_values(
            interpolator=interpolator,
            coordinates=queries,
            output_shape=(3,),
        )
        return DespoticTemperatureCO(
            temperature_K=np.asarray(values[..., 0]),
            co10_luminosity_per_H=np.asarray(values[..., 1]),
            co21_luminosity_per_H=np.asarray(values[..., 2]),
        )

    def abundance(self, *, species: str, queries: DespoticQueries) -> np.ndarray:
        """Return one species' dimensionless abundance relative to H, shape S."""
        return self._interpolate_field(
            token=f"species:{species}:abundance",
            queries=queries,
        )

    def field(self, *, token: str, queries: DespoticQueries) -> np.ndarray:
        """Read a registered field with shape S in its stored units.

        token can be 'tg_final', 'species:CO:line:lumPerH' or 'energy:<name>'.
        queries comes from prepare_queries(); no coordinates are prepared again.
        """
        return self._interpolate_field(
            token=token,
            queries=queries,
        )

    def number_densities(
        self,
        *,
        species: Sequence[str],
        queries: DespoticQueries,
    ) -> dict[str, np.ndarray]:
        """Return each species' number density with shape S [cm^-3].

        Multiply each independently interpolated abundance by the prepared nH.
        Example: species=("e-", "H+", "H") returns those three named arrays.
        """
        hydrogen_density_cm3 = queries.hydrogen_density_cm3.reshape(queries.query_shape)
        number_densities = {}
        for name in species:
            abundance = self.abundance(
                species=name,
                queries=queries,
            )
            number_densities[name] = hydrogen_density_cm3 * abundance
        return number_densities

    def line_field(
        self,
        *,
        species: str,
        field_name: str,
        queries: DespoticQueries,
    ) -> np.ndarray:
        """Read one species' line quantity with shape S at the prepared queries.

        field_name selects freq [Hz], intIntensity [erg/cm^2/s/sr], intTB
        [K km/s], lumPerH [erg/s/H], or dimensionless tau/tauDust.
        Example: species="C+", field_name="lumPerH" returns CII luminosity per H.
        """
        record = self._species_records.get(species)
        if record is None or record.line is None:
            raise ValueError(f"Species '{species}' has no line data")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'; expected one of {LINE_RESULT_FIELDS}")
        return self._interpolate_field(
            token=f"species:{species}:line:{field_name}",
            queries=queries,
        )
