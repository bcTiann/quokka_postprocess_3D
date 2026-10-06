"""Query the fixed Cloudy table in shielding column, density and temperature."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from quokka2s.physics.composition import reject_superseded_composition


TOUCH_EPS = 1.0e-12
EXPECTED_AXIS_ORDER = "line,log_NH_attenuation,log_nH,log_T"


class CloudyFailureTouchError(RuntimeError):
    """Raised when a query gives a failed Cloudy node positive weight."""


@dataclass(frozen=True)
class CloudyQueryResult:
    """Interpolated Cloudy values in the order supplied to the lookup.

    Attributes
    ----------
    emissivity_per_nH2 : ndarray, shape (L, *S)
        Values [erg cm^3/s] in lookup.line_keys order. S is the input broadcast
        shape: selected Q-cell queries normally use S=(Q,). Multiplication by
        physical nH**2 is performed later when
        CellEmissionCalculator.calculate_intrinsic_lines() constructs each line.
    attenuation_column_below_table, attenuation_column_above_table : ndarray of bool, shape S
        True where the supplied shielding NH lies below/above the lookup axis.
        Only the table coordinate is clipped; input cell columns are unchanged.
    failed_queries : ndarray of bool, shape S
        True where at least one line depends on failed table nodes. Every line
        at that query is NaN. These flags do not yet describe original batch cells.

    Example: failed_queries=[False, True] for selected batch cells 1 and 3 means
    that only cell 3's query failed. CloudyCellReader.restore_batch_positions()
    maps it back to the original batch.
    """

    emissivity_per_nH2: np.ndarray
    attenuation_column_below_table: np.ndarray
    attenuation_column_above_table: np.ndarray
    failed_queries: np.ndarray


@dataclass(frozen=True)
class CloudyDiagnostics:
    """Failure-touch and column-clipping diagnostics returned by diagnose().

    Attributes
    ----------
    failure_touched : ndarray of bool, shape (L, *S)
        Failed-node interpolation weight exceeds TOUCH_EPS for each line and
        query. Lines follow lookup.line_keys; S is the broadcast query shape.
    maximum_failure_weight : float
        Largest failed-node weight among touched entries, or zero if none.
    attenuation_column_below_table, attenuation_column_above_table : ndarray of bool, shape S
        Queries whose supplied column was clipped below/above the table bounds.
    """

    failure_touched: np.ndarray
    maximum_failure_weight: float
    attenuation_column_below_table: np.ndarray
    attenuation_column_above_table: np.ndarray


@dataclass(frozen=True)
class CloudyPhysicalQueries:
    """Broadcast physical inputs, each array retaining the query shape S.

    temperature_K [K], hydrogen_density_cm3 [cm^-3], shielding_NH_cm2 [cm^-2]
    come from a lookup entry point. No coordinate has been clipped yet.
    """

    temperature_K: np.ndarray
    hydrogen_density_cm3: np.ndarray
    shielding_NH_cm2: np.ndarray


@dataclass(frozen=True)
class CloudyLogarithmicQueries:
    """Flattened log10 coordinates after checking coverage and clipping NH.

    points: (Q, 3), ordered log NH, log nH, log T.
    query_shape: input broadcast shape S to restore after interpolation.
    below_column, above_column: (Q,) flags from the original log NH values.
    """

    points: np.ndarray
    query_shape: tuple[int, ...]
    below_column: np.ndarray
    above_column: np.ndarray


@dataclass(frozen=True)
class InterpolationCoordinates:
    """Prepared table coordinates, internal to CloudyLookup.

    points: (Q, 3) log10 coordinates, ordered NH, nH, T.
    brackets: one (lower index, upper index, fraction) tuple per axis.
    shape: original query shape before flattening Q queries.
    below_column, above_column: (Q,) NH clipping flags.
    """
    points: np.ndarray
    brackets: tuple
    shape: tuple
    below_column: np.ndarray
    above_column: np.ndarray


@dataclass(frozen=True)
class InterpolationNodes:
    """Per-line corner information, shape (L, Q), before interpolation.

    zero_support marks contributing physical zeros. failure_weight is the
    summed weight of failed nodes; only totals > TOUCH_EPS exclude a query.
    """
    zero_support: np.ndarray
    failure_weight: np.ndarray

    @property
    def failure_touched(self):
        """Return (L, Q) flags using the established node-weight tolerance."""
        return self.failure_weight > TOUCH_EPS


def _validate_axis(name: str, axis: np.ndarray) -> None:
    """Require a finite, increasing 1D axis with at least two nodes.

    name labels a malformed axis in the ValueError; valid axes return None.
    """
    if (
        axis.ndim != 1
        or axis.size < 2
        or not np.isfinite(axis).all()
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(f"{name} must be a finite, strictly increasing 1D axis")


def _brackets(
    axis: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracket coordinates on an increasing 1D axis in the same log10 units.

    Return lower indices, upper indices and upper-node fractions, each with
    coordinate.shape. The caller supplies coordinates inside the axis bounds.
    """
    upper = np.searchsorted(axis, coordinate, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


class CloudyLookup:
    """Trilinear lookup in shielding NH, hydrogen density and temperature.

    The attenuation-column query coordinate is clipped to the table bounds;
    physical cell columns are unchanged. Density and temperature must be covered.
    Positive support is interpolated in log10 emissivity_per_nH2; support containing
    a physical zero uses the linear value. Failed nodes remain unavailable.
    Model depth was applied during table building and is not a lookup coordinate.
    """

    def __init__(self, path: str | Path, *, allow_superseded_composition: bool = False):
        """Load Cloudy arrays and immediately build both emissivity_per_nH2 interpolators.

        Parameters
        ----------
        path : str or pathlib.Path
            Cloudy NPZ path, normally config.cloudy_table. Stored emissivity_per_nH2 grids
            have shape (L, N_NH, N_nH, N_T); line_keys stores their
            line order. The current processing table contains eight atomic lines.
        allow_superseded_composition : bool, optional
            Allow an older composition label; False keeps the adopted check.

        Examples
        --------
        With processing arguments already loaded::

            lookup = CloudyLookup(config.cloudy_table)
            line_keys = lookup.line_keys
        """
        self.path = Path(path)
        self._load_table_arrays(
            allow_superseded_composition=allow_superseded_composition,
        )
        self._validate_table_arrays()
        self._build_interpolators()

    def _load_table_arrays(self, *, allow_superseded_composition: bool) -> None:
        """Read NPZ axes, emissivity_per_nH2, masks and metadata into this lookup.

        self.path comes from __init__(). Values [erg cm^3/s] retain shape
        (L, N_NH, N_nH, N_T); axes store log10 physical coordinates.
        Example: the processing table has eight atomic lines (L=8).
        The NPZ closes after its arrays are loaded; validation follows next.
        """
        with np.load(self.path, allow_pickle=False) as source:
            if not allow_superseded_composition:
                if "composition_label" in source.files:
                    reject_superseded_composition(str(source["composition_label"].item()))
                if "provenance_json" in source.files:
                    provenance = json.loads(str(source["provenance_json"].item()))
                    reject_superseded_composition(provenance.get("abundance", {}).get("setup"))
            required = {
                "axis_order",
                "line_keys",
                "log_NH_attenuation",
                "log_nH",
                "log_T",
                "log_emissivity_per_nH2",
                "emissivity_per_nH2",
                "failure_mask",
                "zero_mask",
            }
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"Cloudy table is missing fields: {missing}")
            axis_order = str(np.asarray(source["axis_order"]).item())
            if axis_order != EXPECTED_AXIS_ORDER:
                raise ValueError(f"unexpected Cloudy axis order: {axis_order!r}")
            self.axis_order = axis_order
            self.line_keys = tuple(
                str(value) for value in np.asarray(source["line_keys"]).tolist()
            )
            self.log_NH_attenuation = np.asarray(
                source["log_NH_attenuation"], dtype=float
            )
            self.log_nH = np.asarray(source["log_nH"], dtype=float)
            self.log_T = np.asarray(source["log_T"], dtype=float)
            self.log_emissivity_per_nH2 = np.asarray(
                source["log_emissivity_per_nH2"], dtype=float
            )
            self.emissivity_per_nH2 = np.asarray(
                source["emissivity_per_nH2"], dtype=float
            )
            self.failed_table_nodes = np.asarray(source["failure_mask"], dtype=bool)
            self.zero_emission_table_nodes = np.asarray(source["zero_mask"], dtype=bool)
            self.metadata = {
                name: np.asarray(source[name])
                for name in source.files
                if name not in required
            }

    def _validate_table_arrays(self) -> None:
        """Check loaded axes, emissivity_per_nH2 shapes and zero/failure consistency.

        Uses arrays from _load_table_arrays(); returns None or raises ValueError.
        For example, a failed node must have a zero linear placeholder, while
        zero_mask marks only successful nodes whose emissivity_per_nH2 is exactly zero.
        """
        _validate_axis("log_NH_attenuation", self.log_NH_attenuation)
        _validate_axis("log_nH", self.log_nH)
        _validate_axis("log_T", self.log_T)
        expected = (
            len(self.line_keys),
            self.log_NH_attenuation.size,
            self.log_nH.size,
            self.log_T.size,
        )
        for name, array in (
            ("log_emissivity_per_nH2", self.log_emissivity_per_nH2),
            ("emissivity_per_nH2", self.emissivity_per_nH2),
            ("failure_mask", self.failed_table_nodes),
            ("zero_mask", self.zero_emission_table_nodes),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} has shape {array.shape}, expected {expected}")
        if np.any(self.emissivity_per_nH2 < 0.0):
            raise ValueError("Cloudy emissivity_per_nH2 must be non-negative")
        if not np.isfinite(self.emissivity_per_nH2).all():
            raise ValueError("Cloudy linear emissivity_per_nH2 must be finite")
        expected_zero = (self.emissivity_per_nH2 == 0.0) & ~self.failed_table_nodes
        if not np.array_equal(self.zero_emission_table_nodes, expected_zero):
            raise ValueError("zero_mask disagrees with linear emissivity_per_nH2")
        if np.any(self.emissivity_per_nH2[self.failed_table_nodes] != 0.0):
            raise ValueError("failed nodes must have zero placeholder emissivity_per_nH2")

    def _build_interpolators(self) -> None:
        """Build shared linear/log interpolators after table validation.

        Move the line axis to the final dimension as a view, so each query
        evaluates all L emissivity_per_nH2 [erg cm^3/s] together. The finite-log
        array preserves historical sentinels, e.g. -99 for a zero emissivity_per_nH2.
        Stores two interpolators on self and returns None.
        """
        axes = [self.log_NH_attenuation, self.log_nH, self.log_T]
        # SciPy treats trailing value dimensions as independent outputs, so a
        # single interpolator evaluates every line at the same coordinates.
        self._linear_interpolator = RegularGridInterpolator(
            points=tuple(axes),
            values=np.moveaxis(self.emissivity_per_nH2, 0, -1),
            method="linear",
            bounds_error=True,
        )
        # Retain finite historical zero sentinels (e.g. -99) exactly as the
        # previous sampler did. Only non-finite log placeholders become zero;
        # the explicit support checks below select the physical output branch.
        finite_log = np.where(
            np.isfinite(self.log_emissivity_per_nH2),
            self.log_emissivity_per_nH2,
            0.0,
        )
        self._log_interpolator = RegularGridInterpolator(
            points=tuple(axes),
            values=np.moveaxis(finite_log, 0, -1),
            method="linear",
            bounds_error=True,
        )

    @property
    def attenuation_column_bounds_cm2(self) -> tuple[float, float]:
        """Return the attenuation-column bounds in physical units.

        Returns
        -------
        tuple of two float
            Minimum and maximum hydrogen-nuclei column [cm^-2], converted
            from the endpoints of self.log_NH_attenuation.
        """
        return (
            float(10.0 ** self.log_NH_attenuation[0]),
            float(10.0 ** self.log_NH_attenuation[-1]),
        )

    def prepare_interpolation_coordinates(
        self,
        temperature_K,
        hydrogen_density_cm3,
        shielding_NH_cm2,
    ) -> InterpolationCoordinates:
        """Prepare physical inputs, log coordinates and interpolation brackets.

        Parameters
        ----------
        temperature_K, hydrogen_density_cm3, shielding_NH_cm2 : array-like, broadcastable to S
            Physical temperature [K], hydrogen-nuclei density [cm^-3] and
            shielding column [cm^-2], supplied by sample(), interpolate_available()
            or diagnose(). Main processing supplies selected arrays with S=(Q,).

        Returns
        -------
        InterpolationCoordinates
            Flattened points (Q,3), their bracketing table indices/fractions, and
            input shape S. Example: points[7] is the eighth query's log NH/nH/T row;
            it need not be the eighth original batch cell.
        """
        physical_queries = self.prepare_physical_query_coordinates(
            temperature_K=temperature_K,
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
        )
        logarithmic_queries = self.prepare_logarithmic_query_coordinates(
            physical_queries=physical_queries,
        )
        return self.locate_query_interpolation_brackets(
            logarithmic_queries=logarithmic_queries,
        )

    def prepare_physical_query_coordinates(
        self, temperature_K, hydrogen_density_cm3, shielding_NH_cm2,
    ) -> CloudyPhysicalQueries:
        """Broadcast the physical inputs and require finite, positive coordinates.

        Inputs have the units documented by prepare_interpolation_coordinates().
        Return CloudyPhysicalQueries with shape S, without selecting or clipping
        any cell. For example, scalar T and three density/column values give S=(3,).
        """
        inputs = [
            np.asarray(temperature_K, dtype=float),
            np.asarray(hydrogen_density_cm3, dtype=float),
            np.asarray(shielding_NH_cm2, dtype=float),
        ]
        inputs = np.broadcast_arrays(*inputs)
        if not all(np.isfinite(value).all() for value in inputs):
            raise ValueError("Cloudy lookup inputs must all be finite")
        if any(np.any(value <= 0.0) for value in inputs):
            raise ValueError("Cloudy lookup inputs must all be positive")

        return CloudyPhysicalQueries(
            temperature_K=inputs[0],
            hydrogen_density_cm3=inputs[1],
            shielding_NH_cm2=inputs[2],
        )

    def prepare_logarithmic_query_coordinates(self, physical_queries) -> CloudyLogarithmicQueries:
        """Convert to log10 table units, clip NH and check all other axes.

        physical_queries comes from prepare_physical_query_coordinates(). Only
        shielding NH permits physical clipping. Density/temperature accept the
        existing log-coordinate roundoff tolerance. Physical input arrays remain unchanged.
        """
        log_column = np.log10(physical_queries.shielding_NH_cm2).ravel()
        below = log_column < self.log_NH_attenuation[0]
        above = log_column > self.log_NH_attenuation[-1]
        log_density = np.log10(physical_queries.hydrogen_density_cm3).ravel()
        log_temperature = np.log10(physical_queries.temperature_K).ravel()

        for name, axis, coordinate in (
            ("log_nH", self.log_nH, log_density),
            ("log_T", self.log_T, log_temperature),
        ):
            tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
            if np.any(coordinate < axis[0] - tolerance) or np.any(
                coordinate > axis[-1] + tolerance
            ):
                raise ValueError(
                    f"{name} is outside [{axis[0]:.8g}, {axis[-1]:.8g}]"
                )

        coordinates = [
            np.clip(log_column, self.log_NH_attenuation[0], self.log_NH_attenuation[-1]),
            np.clip(log_density, self.log_nH[0], self.log_nH[-1]),
            np.clip(log_temperature, self.log_T[0], self.log_T[-1]),
        ]
        return CloudyLogarithmicQueries(
            points=np.stack(coordinates, axis=-1),
            query_shape=physical_queries.temperature_K.shape,
            below_column=below,
            above_column=above,
        )

    def locate_query_interpolation_brackets(self, logarithmic_queries) -> InterpolationCoordinates:
        """Locate lower/upper nodes and fractions for every log-coordinate row.

        logarithmic_queries comes from prepare_logarithmic_query_coordinates().
        Each axis has Q lower indices, upper indices and interpolation fractions.
        Table axes stay ordered NH, nH, T, as in the saved table.
        """
        axes = [self.log_NH_attenuation, self.log_nH, self.log_T]

        brackets = []
        for axis_index, axis in enumerate(axes):
            bracket = _brackets(
                axis=axis,
                coordinate=logarithmic_queries.points[:, axis_index],
            )
            brackets.append(bracket)
        return InterpolationCoordinates(
            points=logarithmic_queries.points,
            brackets=tuple(brackets),
            shape=logarithmic_queries.query_shape,
            below_column=logarithmic_queries.below_column,
            above_column=logarithmic_queries.above_column,
        )

    def interpolate_available(
        self, temperature_K, hydrogen_density_cm3, shielding_NH_cm2,
    ) -> CloudyQueryResult:
        """Read all lines once, returning unavailable queries as NaN.

        Parameters
        ----------
        temperature_K, hydrogen_density_cm3, shielding_NH_cm2 : array-like, broadcastable to S
            Physical T [K], nH [cm^-3] and shielding NH [cm^-2] from the selected
            CloudyQueryInputs. Main processing supplies arrays of shape (Q,).

        Returns
        -------
        CloudyQueryResult
            emissivity_per_nH2 has shape (L, *S) [erg cm^3/s]. If any line touches
            failed nodes, every line at that query is NaN; physical zeros stay zero.
            Example: result.failed_queries[7] describes the eighth query, before
            CloudyCellReader.restore_batch_positions() maps results to the
            original batch.
        """
        coordinates = self.prepare_interpolation_coordinates(
            temperature_K=temperature_K,
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
        )
        nodes = self.inspect_interpolation_nodes(coordinates=coordinates)
        return self.interpolate_available_emissivity_per_nH2(
            coordinates=coordinates,
            nodes=nodes,
        )

    def sample(
        self, temperature_K, hydrogen_density_cm3, shielding_NH_cm2,
    ) -> CloudyQueryResult:
        """Read all lines, requiring every supplied query to be available.

        Parameters
        ----------
        temperature_K, hydrogen_density_cm3, shielding_NH_cm2 : array-like, broadcastable to S
            Physical T [K], nH [cm^-3] and shielding NH [cm^-2], normally from a
            diagnostic's selected cell arrays. Same coordinates as interpolate_available().

        Returns
        -------
        CloudyQueryResult
            Values [erg cm^3/s], shape (L, *S). Failed-node support raises
            CloudyFailureTouchError. Example: result.emissivity_per_nH2[0, 7]
            is the eighth query's first line value before multiplying by nH**2.
        """
        coordinates = self.prepare_interpolation_coordinates(
            temperature_K=temperature_K,
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
        )
        nodes = self.inspect_interpolation_nodes(coordinates=coordinates)
        touched = nodes.failure_touched
        if np.any(touched):
            counts = {}
            for index, key in enumerate(self.line_keys):
                if np.any(touched[index]):
                    counts[key] = int(np.count_nonzero(touched[index]))
            raise CloudyFailureTouchError(
                "simulation touches unavailable Cloudy nodes: "
                f"counts={counts}, maximum_weight={nodes.failure_weight[touched].max():.6g}"
            )
        return self.interpolate_available_emissivity_per_nH2(
            coordinates=coordinates,
            nodes=nodes,
        )

    def inspect_interpolation_nodes(self, coordinates) -> InterpolationNodes:
        """Inspect the eight 3D corners once for all requested lines.

        coordinates comes from prepare_interpolation_coordinates(). Keep the
        historical corner order and multiplication order for reproducible sums.
        A zero corner matters individually above TOUCH_EPS; failed weights are
        summed first, then compared with that threshold.
        """
        query_count = len(coordinates.points)
        shape = (len(self.line_keys), query_count)
        zero_support = np.zeros(shape, dtype=bool)
        failure_weight = np.zeros(shape)
        brackets = coordinates.brackets

        # 0 selects the lower node; 1 selects the upper node on each axis.
        for corner in product((0, 1), repeat=3):
            indices = []
            weight = np.ones(query_count)
            for side, (lower, upper, fraction) in zip(corner, brackets):
                if side == 0:
                    indices.append(lower)
                    weight = weight * (1.0 - fraction)
                else:
                    indices.append(upper)
                    weight = weight * fraction
            table_index = (slice(None), *indices)
            line_weights = weight[None, :]
            zero_support |= self.zero_emission_table_nodes[table_index] & (line_weights > TOUCH_EPS)
            failure_weight += self.failed_table_nodes[table_index] * line_weights

        return InterpolationNodes(
            zero_support=zero_support,
            failure_weight=failure_weight,
        )

    def interpolate_available_emissivity_per_nH2(self, coordinates, nodes) -> CloudyQueryResult:
        """Interpolate retained points and restore their original query shape.

        coordinates/nodes come from the two preparation steps above. All lines
        are excluded together if one line has failed support; no failed-node
        placeholder is interpolated as physical emission.
        """
        failed_queries = nodes.failure_touched.any(axis=0)
        retained_queries = ~failed_queries
        emissivity_per_nH2 = np.full(nodes.zero_support.shape, np.nan)
        if np.any(retained_queries):
            emissivity_per_nH2[:, retained_queries] = self._interpolate_emissivity_per_nH2(
                points=coordinates.points[retained_queries],
                zero_support=nodes.zero_support[:, retained_queries],
            )
        return CloudyQueryResult(
            emissivity_per_nH2=emissivity_per_nH2.reshape(
                (len(self.line_keys), *coordinates.shape),
            ),
            attenuation_column_below_table=coordinates.below_column.reshape(coordinates.shape),
            attenuation_column_above_table=coordinates.above_column.reshape(coordinates.shape),
            failed_queries=failed_queries.reshape(coordinates.shape),
        )

    def _interpolate_emissivity_per_nH2(self, points, zero_support) -> np.ndarray:
        """Evaluate the required linear/log branches without repeating queries.

        points contains log10 NH, nH, T, shape (Q, 3).
        zero_support has shape (L, Q). Returns emissivity_per_nH2 of
        shape (L, Q) [erg cm^3/s]. For example, one zero-supported line uses
        the linear result while positive lines at the same point use log values.
        """
        # Each RGI evaluates all lines together; a point needs a branch only
        # when at least one of its lines uses that branch. Mixed-line points
        # still evaluate both and retain the same per-line support selection.
        need_linear = zero_support.any(axis=0)
        need_log = (~zero_support).any(axis=0)
        emissivity_per_nH2 = np.full(zero_support.shape, np.nan)
        if need_log.any():
            emissivity_per_nH2[:, need_log] = np.power(
                10.0, self._log_interpolator(points[need_log]).T
            )
        if need_linear.any():
            linear_emissivity_per_nH2 = self._linear_interpolator(points[need_linear]).T
            emissivity_per_nH2[:, need_linear] = np.where(
                zero_support[:, need_linear],
                linear_emissivity_per_nH2,
                emissivity_per_nH2[:, need_linear],
            )
        return emissivity_per_nH2

    def diagnose(
        self,
        temperature_K,
        hydrogen_density_cm3,
        shielding_NH_cm2,
    ) -> CloudyDiagnostics:
        """Report unavailable interpolation support for every line and query.

        Parameters
        ----------
        temperature_K, hydrogen_density_cm3, shielding_NH_cm2 : array-like, broadcastable to shape S
            Same coordinates as sample(): temperature [K], hydrogen-nuclei
            density [cm^-3], and shielding column [cm^-2].

        Returns
        -------
        CloudyDiagnostics
            Failed-node mask of shape (len(self.line_keys), *S), in stored
            line order; the maximum failed-node weight; and two column-clipping
            masks of shape S. Unlike sample(), failed-node touches are returned
            without raising CloudyFailureTouchError.

        Examples
        --------
        With a three-dimensional lookup and selected hot-cell arrays already loaded::

            diagnostic = lookup.diagnose(temperature_K, nH, shielding_NH)
            omitted = diagnostic.failure_touched.any(axis=0)
        """
        coordinates = self.prepare_interpolation_coordinates(
            temperature_K=temperature_K,
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=shielding_NH_cm2,
        )
        nodes = self.inspect_interpolation_nodes(coordinates=coordinates)
        touched = nodes.failure_touched
        maximum_weight = 0.0
        if np.any(touched):
            maximum_weight = float(nodes.failure_weight[touched].max())
        return CloudyDiagnostics(
            failure_touched=touched.reshape((len(self.line_keys), *coordinates.shape)),
            maximum_failure_weight=maximum_weight,
            attenuation_column_below_table=coordinates.below_column.reshape(coordinates.shape),
            attenuation_column_above_table=coordinates.above_column.reshape(coordinates.shape),
        )
