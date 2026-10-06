import numpy as np

from quokka2s.figures.gas_phase_spectra import (
    select_display_profiles,
    read_display_profile_statistics,
    check_phase_comparison_options,
    LineSpectraForComparison,
    LINE_ORDER,
)


def _payload():
    keys = tuple(reversed(LINE_ORDER))
    spectra = np.arange(1., 41.).reshape(10, 2, 2)
    for key in keys:
        if key.startswith(('ciii_', 'civ_')):
            spectra[keys.index(key), 0] = 0.
    return LineSpectraForComparison(
        line_keys=keys,
        velocity_kms=np.array([-1., 1.]),
        dL_dv_erg_s_per_kms=spectra,
        line_centroid_window_kms=np.arange(10.),
        line_sigma_full_kms=np.arange(10.) + 100.,
    )


def test_profiles_match_saved_components_without_mutating_source():
    payload = _payload()
    before = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in vars(payload).items()
    }
    profiles = select_display_profiles(payload)
    report = read_display_profile_statistics(payload)
    keys = list(payload.line_keys)
    for key in LINE_ORDER:
        index = keys.index(key)
        source = payload.dL_dv_erg_s_per_kms[index]
        if key.startswith(('ciii_', 'civ_')):
            expected = source[1]
            branch = 'hot'
        else:
            expected = source.sum(axis=0)
            branch = 'total'
        np.testing.assert_array_equal(profiles[key], expected)
        assert report[key]['display_branch'] == branch
        assert report[key]['sigma_line_full_kms'] == payload.line_sigma_full_kms[index]
    profiles['co10'][0] = -99.
    for key in before:
        np.testing.assert_array_equal(getattr(payload, key), before[key])


def test_plot_reads_saved_full_sigma_instead_of_recalculating_from_channels():
    payload = _payload()
    index = list(payload.line_keys).index('co10')
    payload.dL_dv_erg_s_per_kms[index, 0] = [1., 3.]
    payload.line_centroid_window_kms[index] = 17.
    payload.line_sigma_full_kms[index] = 9.
    report = read_display_profile_statistics(payload)
    assert report['co10']['mean_velocity_kms'] == 17.
    assert report['co10']['sigma_line_full_kms'] == 9.


def test_invalid_display_options_are_rejected():
    import unittest
    options = (
        ((), 'latex'),
        (('co10', 'co10'), 'latex'),
        (('unknown',), 'latex'),
        (('co10',), 'unknown'),
    )
    for line_keys, figure_style in options:
        with unittest.TestCase().assertRaises(ValueError):
            check_phase_comparison_options(line_keys, figure_style)


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items())
        if name.startswith('test_')
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
