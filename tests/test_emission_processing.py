"""Numerical-product and relocated-input checks for the process command."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.emission_processing import (
    _accepted_file, combine_spectral_variants, slab_windows,
)


class EmissionProcessingTests(unittest.TestCase):
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
