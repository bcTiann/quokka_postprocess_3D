from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.despotic.table_data import DespoticTable
from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.snapshot_reader import SlabArrays


SCRIPT = Path(__file__).resolve().parents[1] / "tools/despotic/check_snapshot_coverage.py"
SPEC = importlib.util.spec_from_file_location("snapshot_coverage_diagnostic", SCRIPT)
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


def make_table(size=2):
    axes = np.logspace(0, size - 1, size)
    values = np.ones((size, size, size))
    return DespoticTable(
        species_data={}, tg_final=values * 50, nH_values=axes,
        col_density_values=axes.copy(), dVdr_values=axes.copy(),
        mu_values=values * 2, cv_values=values, Eint_values=values,
        failure_mask=np.zeros(values.shape, dtype=bool))


def clip_inputs(lookup, *values):
    return lookup.clip_coordinates(*values)


def classify(table, points):
    return coverage.classify_queries(DespoticLookup(table), *np.asarray(points).T,
                                     clip_inputs=clip_inputs)


class SnapshotCoverageTests(unittest.TestCase):
    def test_nan_at_zero_weight_corner_is_reported_by_actual_lookup(self):
        table = make_table()
        table.failure_mask[1, 1, 1] = True
        table.tg_final[1, 1, 1] = np.nan
        table.mu_values[1, 1, 1] = np.nan
        flags = classify(table, [(1., 1., 1.), (np.sqrt(10),) * 3, (10., 10., 10.)])
        np.testing.assert_array_equal(flags["positive_weight_failed_support"], [False, True, True])
        np.testing.assert_array_equal(flags["any_failed_among_eight_corners"], True)
        np.testing.assert_array_equal(flags["temperature_or_mu_query_nonfinite"], True)
        np.testing.assert_array_equal(flags["query_nonfinite_without_positive_failed_support"],
                                      [True, False, False])
        self.assertFalse(flags["all_eight_corners_failed"].any())

    def test_exact_interior_node_uses_right_interval_as_rgi(self):
        table = make_table(size=3)
        table.failure_mask[0, 0, 0] = True
        table.tg_final[0, 0, 0] = np.nan
        table.mu_values[0, 0, 0] = np.nan
        flags = classify(table, [(10., 10., 10.)])
        self.assertFalse(flags["any_failed_among_eight_corners"][0])
        self.assertFalse(flags["temperature_or_mu_query_nonfinite"][0])

    def test_invalid_inputs_are_not_clipped_into_queries(self):
        table = make_table()
        calls = []
        def recording_clip(lookup, *values):
            calls.append(tuple(v.copy() for v in values))
            return clip_inputs(lookup, *values)
        nH = np.array([0., -1., np.nan, np.inf, .1, 100., 2.])
        original = nH.copy()
        flags = coverage.classify_queries(DespoticLookup(table), nH, 2., 2.,
                                          clip_inputs=recording_clip)
        np.testing.assert_array_equal(flags["invalid_input"], [True] * 4 + [False] * 3)
        np.testing.assert_array_equal(flags["raw_below_nH"], [False] * 4 + [True, False, False])
        np.testing.assert_array_equal(flags["raw_above_nH"], [False] * 5 + [True, False])
        self.assertFalse(flags["temperature_or_mu_query_nonfinite"].any())
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], [.1, 100., 2.])
        np.testing.assert_array_equal(nH, original)

    def test_unflagged_nonfinite_mu_and_nonpositive_temperature_remain_visible(self):
        table = make_table()
        table.mu_values[:] = np.nan
        table.tg_final[:] = 0.
        flags = classify(table, [(3., 3., 3.)])
        self.assertFalse(flags["positive_weight_failed_support"][0])
        self.assertTrue(flags["mu_query_nonfinite"][0])
        self.assertFalse(flags["temperature_query_nonfinite"][0])
        self.assertTrue(flags["temperature_or_mu_query_nonpositive"][0])

    def test_all_eight_failures_and_clipped_raw_outside_are_distinct(self):
        table = make_table()
        table.failure_mask[:] = True
        table.tg_final[:] = np.nan
        table.mu_values[:] = np.nan
        flags = classify(table, [(.1, 100., 3.), (3., 3., 3.)])
        np.testing.assert_array_equal(flags["all_eight_corners_failed"], True)
        np.testing.assert_array_equal(flags["positive_weight_failed_support"], True)
        np.testing.assert_array_equal(flags["raw_out_of_bounds"], [True, False])

    def test_counts_cold_boundary_and_mass_denominators(self):
        totals = coverage.CoverageTotals()
        totals.add({"bad": np.array([True, True, False, True]),
                    "valid_input": np.ones(4, dtype=bool)},
                   np.array([100., 3000., 2999., np.nan]),
                   np.array([1., 9., 3., np.nan]))
        result = totals.result()
        all_cells, cold = result["all_cells"], result["T_QUOKKA_lt_3000_K"]
        self.assertEqual(all_cells["cell_count"], 4)
        self.assertEqual(all_cells["flags"]["bad"]["cell_fraction"], .75)
        self.assertAlmostEqual(all_cells["flags"]["bad"]["mass_fraction"], 10 / 13)
        self.assertEqual(all_cells["flags"]["invalid_mass"]["cell_count"], 1)
        self.assertEqual(all_cells["flags"]["invalid_temperature_quokka"]["cell_count"], 1)
        self.assertEqual(cold["cell_count"], 2)
        self.assertEqual(cold["flags"]["bad"]["cell_fraction"], .5)
        self.assertEqual(cold["flags"]["bad"]["mass_fraction"], .25)
        json.dumps(result, allow_nan=False)

    def test_chunked_accumulation_matches_one_pass_and_empty_cold_is_null(self):
        table = make_table()
        points = np.array([(2., 2., 2.), (0., 2., 2.), (20., 2., 2.), (3., 3., 3.)])
        full, chunked = coverage.CoverageTotals(), coverage.CoverageTotals()
        full.add(classify(table, points), np.full(4, 4000.), np.arange(1., 5.))
        for i in range(4):
            chunked.add(classify(table, points[i:i+1]), np.array([4000.]), np.array([i+1.]))
        self.assertEqual(full.result(), chunked.result())
        self.assertIsNone(full.result()["T_QUOKKA_lt_3000_K"]["flags"]["valid_input"]["cell_fraction"])

    def test_slab_windows_cover_every_cell_once(self):
        # Neighbour reads now belong to Snapshot.read_slab(); this diagnostic
        # receives only the non-overlapping x bounds it needs to scan.
        for slab_nx in (1, 2, 3, 20):
            counts = np.zeros(7, dtype=int)
            for x_start, x_stop in coverage.slab_windows(7, slab_nx):
                counts[x_start:x_stop] += 1
            np.testing.assert_array_equal(counts, 1)

    def test_snapshot_scan_batches_preserve_cell_selection_and_coverage_counts(self):
        nH = np.array([1., 2., 20., 0., 3., 4., 5., 6., 7., 8., 9., 10.])
        rho = nH * HYDROGEN_MASS_G / coverage.settings.X_H
        TQ = np.array([100., 3000., 4000., np.nan] * 3)

        class Snapshot:
            shape = (3, 2, 2)
            cell_count = 12

            def read_slab(self, x_start, x_stop):
                selection = slice(x_start * 4, x_stop * 4)
                count = (x_stop - x_start) * 4
                return SlabArrays(
                    density_g_cm3=rho[selection],
                    foreground_NH_cm2=np.ones(count),
                    temperature_QUOKKA_K=TQ[selection],
                    shielding_NH_cm2=np.full(count, 2.),
                    velocity_gradient_s=np.full(count, 3.),
                    velocity_z_kms=np.zeros(count),
                    x_start=x_start,
                    shape=(x_stop - x_start, 2, 2),
                    cell_volume_cm3=4.,
                )

        lookup = DespoticLookup(make_table())
        with redirect_stdout(StringIO()):
            totals, scanned, extrema = coverage.scan_snapshot_coverage(
                snapshot=Snapshot(),
                lookup=lookup,
                slab_nx=2,
                query_chunk=3,
            )
        result = totals.result()
        self.assertEqual(scanned, 12)
        self.assertEqual(result["all_cells"]["cell_count"], 12)
        self.assertEqual(result["T_QUOKKA_lt_3000_K"]["cell_count"], 3)
        self.assertEqual(result["all_cells"]["flags"]["invalid_input"]["cell_count"], 1)
        self.assertEqual(result["all_cells"]["flags"]["raw_above_nH"]["cell_count"], 1)
        self.assertAlmostEqual(extrema["nH"]["minimum"], 1.)
        self.assertAlmostEqual(extrema["nH"]["maximum"], 20.)
        self.assertEqual(extrema["NH"], dict(minimum=2., maximum=2.))
        self.assertEqual(extrema["dVdr"], dict(minimum=3., maximum=3.))

    def test_missing_failure_provenance_or_bad_axes_rejected(self):
        with self.assertRaisesRegex(ValueError, "failure_mask"):
            classify(replace(make_table(), failure_mask=None), [(2., 2., 2.)])
        with self.assertRaisesRegex(ValueError, "increasing positive"):
            coverage._table_axes(replace(make_table(), nH_values=np.array([0., 1.])))

    def test_missing_candidate_fails_before_loading_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "pending.npz"
            with self.assertRaisesRegex(FileNotFoundError, "not complete or is missing"):
                coverage.main(["--table", str(missing), "--output", str(Path(directory)/"report.json")])

    def test_provenance_checks_physical_setup_and_table_endpoints_without_hash_dependency(self):
        table = make_table()
        domain = dict(
            selection="all simulation cells",
            dataset="/old/machine/snapshot",
            shape=[2, 2, 2],
            total_cells=8,
            X_H=coverage.settings.X_H,
            column_mean="harmonic",
            column_directions="z",
            physics_source_sha256="old-source-hash-before-refactor",
            axes={name: dict(minimum=1., maximum=10.) for name in coverage.AXIS_NAMES},
        )
        table = replace(table, build_metadata={"snapshot_domain": domain})
        # A source-code refactor or relocating the files does not alter physics.
        self.assertEqual(coverage._validate_scan_provenance(table, (2, 2, 2)), domain)
        with self.assertRaisesRegex(ValueError, "shape"):
            coverage._validate_scan_provenance(table, (2, 2, 3))
        original_XH = domain["X_H"]
        domain["X_H"] = .5
        with self.assertRaisesRegex(ValueError, "X_H"):
            coverage._validate_scan_provenance(table, (2, 2, 2))
        domain["X_H"] = original_XH
        domain["axes"]["NH"]["maximum"] = 11.
        with self.assertRaisesRegex(ValueError, "bounds differ"):
            coverage._validate_scan_provenance(table, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
