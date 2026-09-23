import numpy as np

from quokka2s.adopted_phase_overlay import accepted_display_profiles, LINE_ORDER
from quokka2s.adopted_spectral_products import REGIME_KEYS


def _payload():
    keys = tuple(reversed(LINE_ORDER))
    edges = np.array([-2., 0., 2.])
    spectra = np.arange(1., 41.).reshape(10, 2, 2)
    for key in keys:
        if key.startswith(('ciii_', 'civ_')):
            spectra[keys.index(key), 0] = 0.
    return dict(line_keys=np.asarray(keys), regime_keys=np.asarray(REGIME_KEYS),
        velocity_edges_kms=edges, velocity_kms=np.array([-1., 1.]),
        dL_dv_erg_s_per_kms=spectra, total_dL_dv_erg_s_per_kms=spectra.sum(axis=1))


def test_profiles_match_adopted_display_without_mutating_source():
    payload = _payload()
    before = {key: value.copy() for key, value in payload.items()}
    profiles, report = accepted_display_profiles(payload)
    keys = list(payload['line_keys'])
    for key in LINE_ORDER:
        source = payload['dL_dv_erg_s_per_kms'][keys.index(key)]
        if key.startswith('co'):
            expected, branch = source[0], 'cold'
        elif key.startswith(('ciii_', 'civ_')):
            expected, branch = source[1], 'hot'
        else:
            expected, branch = source.sum(axis=0), 'total'
        np.testing.assert_array_equal(profiles[key], expected)
        assert report[key]['display_branch'] == branch
        assert report[key]['captured_luminosity_erg_s'] == 2*expected.sum()
    profiles['co10'][0] = -99.
    for key in before:
        np.testing.assert_array_equal(payload[key], before[key])


def test_line_sigma_is_own_centroid_within_saved_window():
    payload = _payload()
    index = list(payload['line_keys']).index('co10')
    payload['dL_dv_erg_s_per_kms'][index, 0] = [1., 3.]
    payload['total_dL_dv_erg_s_per_kms'] = payload['dL_dv_erg_s_per_kms'].sum(axis=1)
    _, report = accepted_display_profiles(payload)
    assert report['co10']['mean_velocity_kms'] == .5
    np.testing.assert_allclose(report['co10']['sigma_line_window_kms'], np.sqrt(.75))


def test_invalid_or_mismatched_source_is_rejected():
    import unittest
    for kind in ('cold_ciii', 'negative', 'bad_total', 'wrong_regime', 'bad_velocity'):
        payload = _payload()
        if kind == 'cold_ciii':
            i = list(payload['line_keys']).index('ciii_977')
            payload['dL_dv_erg_s_per_kms'][i, 0, 0] = 1.
            payload['total_dL_dv_erg_s_per_kms'] = payload['dL_dv_erg_s_per_kms'].sum(axis=1)
        elif kind == 'negative': payload['dL_dv_erg_s_per_kms'][0, 0, 0] = -1.
        elif kind == 'bad_total': payload['total_dL_dv_erg_s_per_kms'][0, 0] += 1.
        elif kind == 'wrong_regime': payload['regime_keys'] = np.array(['wrong', 'order'])
        elif kind == 'bad_velocity': payload['velocity_kms'][0] = 0.
        with unittest.TestCase().assertRaises(ValueError, msg=kind):
            accepted_display_profiles(payload)


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
