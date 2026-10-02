"""Numerical-product and relocated-input checks for the process command."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.cloudy_cell_queries import CloudyCellQueries
from quokka2s.cloudy_sixline_lookup import CloudyFailureTouchError
from quokka2s.emission_processing import (
    _compute_with_cloudy_failure_exclusions,
    combine_spectral_variants, slab_windows,
)
from quokka2s.legacy_emission_validation import _accepted_file


class EmissionProcessingTests(unittest.TestCase):
    @staticmethod
    def cloudy_queries():
        return CloudyCellQueries(
            n_H_cm3=np.array([1., 2., 3., 4.]),
            column_density_H_cm2=np.full(4, 1e19),
            model_depth_pc=np.full(4, 100.),
            state=SimpleNamespace(
                cold_mask=np.array([False, True, False, False]),
                temperature_K=np.array([1e4, 50., 2e4, 1e4])),
            excluded=np.array([False, False, False, True]),
        )

    def test_cloudy_failure_in_one_line_excludes_whole_cell(self):
        queries = self.cloudy_queries()
        calls = []

        class Lookup:
            def diagnose(self, temperature, n_h, column):
                np.testing.assert_array_equal(temperature, [1e4, 2e4])
                np.testing.assert_array_equal(n_h, [1., 3.])
                return SimpleNamespace(failure_touched=np.array([
                    [False, False], [False, True], [False, False],
                ]))

        def compute(current, dvdr, despotic, cloudy, *, allow_capped_legacy_jeans):
            self.assertTrue(allow_capped_legacy_jeans)
            calls.append(current.excluded.copy())
            if not current.excluded[2]:
                raise CloudyFailureTouchError('one line touches a failed node')
            emissivity = np.where(current.excluded[None, :], np.nan,
                                  np.ones((3, current.excluded.size)))
            return SimpleNamespace(valid=~current.excluded,
                                   emissivity_erg_s_cm3=emissivity)

        with patch('quokka2s.emission_processing.compute_adopted_cell_emission',
                   side_effect=compute):
            selected, emission, excluded_cloudy = _compute_with_cloudy_failure_exclusions(
                queries, np.ones(4), object(), Lookup())
        np.testing.assert_array_equal(calls, [[False, False, False, True],
                                              [False, False, True, True]])
        np.testing.assert_array_equal(excluded_cloudy, [False, False, True, False])
        np.testing.assert_array_equal(selected.excluded, [False, False, True, True])
        np.testing.assert_array_equal(emission.valid, [True, True, False, False])
        self.assertTrue(np.isnan(emission.emissivity_erg_s_cm3[:, 2]).all())
        self.assertEqual(int(excluded_cloudy.sum()), 1)
        self.assertEqual(int(selected.excluded.sum()), 2)

    def test_unexplained_cloudy_failure_still_raises(self):
        class Lookup:
            def diagnose(self, temperature, n_h, column):
                return SimpleNamespace(failure_touched=np.zeros((3, temperature.size), dtype=bool))

        def compute(*args, **kwargs):
            raise CloudyFailureTouchError('not explained by a failed node')

        with self.assertRaisesRegex(CloudyFailureTouchError, 'not explained'):
            with patch('quokka2s.emission_processing.compute_adopted_cell_emission',
                       side_effect=compute):
                _compute_with_cloudy_failure_exclusions(
                    self.cloudy_queries(), np.ones(4), object(), Lookup())

    def test_non_cloudy_failure_is_not_swallowed(self):
        def compute(*args, **kwargs):
            raise ValueError('unexpected emissivity failure')

        with self.assertRaisesRegex(ValueError, 'unexpected emissivity failure'):
            with patch('quokka2s.emission_processing.compute_adopted_cell_emission',
                       side_effect=compute):
                _compute_with_cloudy_failure_exclusions(
                    self.cloudy_queries(), np.ones(4), object(), object())

    def test_two_variant_moments_use_full_box_channel_luminosity(self):
        edges = np.array([-2., 0., 2.])
        base = {
            'line_keys': np.array(['cii']),
            'regime_keys': np.array(['cold', 'hot']),
            'velocity_edges_kms': edges,
            'velocity_kms': np.array([-1., 1.]),
            'cell_counts_by_regime': np.array([2, 3]),
            'input_luminosity_erg_s': np.array([[2., 2.]]),
            'captured_luminosity_erg_s': np.array([[2., 2.]]),
            'outside_velocity_luminosity_erg_s': np.zeros((1, 2)),
        }
        intrinsic = dict(base, dL_dv_erg_s_per_kms=np.array([[[1., 0.], [0., 1.]]]))
        attenuated = dict(base, dL_dv_erg_s_per_kms=np.array([[[.5, 0.], [0., .5]]]),
                          input_luminosity_erg_s=np.array([[1., 1.]]),
                          captured_luminosity_erg_s=np.array([[1., 1.]]))
        result = combine_spectral_variants(
            {'intrinsic': intrinsic, 'attenuated': attenuated}, 10.)
        self.assertEqual(result['dL_dv_erg_s_per_kms'].shape, (2, 1, 2, 2))
        np.testing.assert_allclose(result['line_centroid_kms'][:, 0], 0.)
        np.testing.assert_allclose(result['line_sigma_kms'][:, 0], 1.)
        np.testing.assert_allclose(result['total_dL_dv_erg_s_per_kms'][:, 0],
                                   [[1., 1.], [.5, .5]])

    def test_relocated_manifest_finds_adjacent_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = directory / 'accepted_table.json'
            manifest.write_text('{}')
            copied = directory / 'table.npz'
            copied.write_bytes(b'table')
            chosen = _accepted_file(manifest, '/old/mac/path/table.npz')
            self.assertEqual(chosen, copied.resolve())
            other = directory / 'override.npz'
            other.write_bytes(b'override')
            self.assertEqual(_accepted_file(manifest, '/old/mac/path/table.npz', other),
                             other.resolve())

    def test_slab_windows_cover_native_x_without_overlap(self):
        windows = list(slab_windows(10, 4))
        self.assertEqual([(start, end) for start, end, *_ in windows],
                         [(0, 4), (4, 8), (8, 10)])
        self.assertEqual([(lo, hi) for _, _, lo, hi, _ in windows],
                         [(0, 5), (3, 9), (7, 10)])


if __name__ == '__main__':
    unittest.main()
