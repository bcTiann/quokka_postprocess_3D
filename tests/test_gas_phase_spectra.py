import numpy as np

from quokka2s.emission_results import GasPhaseResults, SpectralResults
from quokka2s.figures.gas_phase_spectra import (
    select_display_profiles,
    read_display_profile_statistics,
    check_phase_comparison_options,
    draw_gas_mass_profile_curves,
    draw_line_profile_curve,
    format_phase_cut_footer,
    LINE_ORDER,
)


def _payload():
    keys = tuple(reversed(LINE_ORDER))
    cold_and_hot = np.arange(1., 41.).reshape(10, 2, 2)
    for key in keys:
        if key.startswith(('ciii_', 'civ_')):
            cold_and_hot[keys.index(key), 0] = 0.
    # All named axes deliberately differ from the production order.
    profiles = np.stack((cold_and_hot[:, ::-1], 7. * cold_and_hot[:, ::-1]))
    total = profiles.sum(axis=2)
    moments = np.stack((np.arange(10.), np.arange(10.) + 20.))
    return SpectralResults(
        line_keys=keys,
        dust_state_keys=('attenuated', 'intrinsic'),
        regime_keys=('T_QUOKKA_ge_3000K', 'T_QUOKKA_lt_3000K'),
        axis_order='dust_state,line,regime,velocity_channel',
        velocity_edges_kms=np.array([-2., 0., 2.]),
        velocity_kms=np.array([-1., 1.]),
        dL_dv_erg_s_per_kms=profiles,
        total_dL_dv_erg_s_per_kms=total,
        projected_area_cm2=1.,
        line_centroid_window_kms=moments,
        line_sigma_window_kms=moments + 10.,
        line_centroid_full_kms=moments + 30.,
        line_sigma_full_kms=moments + 100.,
        line_centroid_window_by_regime_kms=np.repeat(moments[:, :, None], 2, axis=2),
        line_sigma_window_by_regime_kms=np.repeat(moments[:, :, None] + 40., 2, axis=2),
    )


def test_profiles_match_saved_named_components_without_mutating_source():
    payload = _payload()
    before = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in vars(payload).items()
    }
    profiles = select_display_profiles(payload)
    report = read_display_profile_statistics(payload)
    for key in LINE_ORDER:
        index = payload.line_keys.index(key)
        source = payload.dL_dv_erg_s_per_kms[0, index]
        if key.startswith(('ciii_', 'civ_')):
            expected = source[0]  # Saved regime order is hot, cold.
            branch = 'hot'
        else:
            expected = payload.total_dL_dv_erg_s_per_kms[0, index]
            branch = 'total'
        np.testing.assert_array_equal(profiles[key], expected)
        assert report[key]['display_branch'] == branch
        assert report[key]['sigma_line_full_kms'] == payload.line_sigma_full_kms[0, index]
    profiles['co10'][0] = -99.
    for key in before:
        np.testing.assert_array_equal(getattr(payload, key), before[key])


def test_total_and_regime_selections_keep_their_saved_moment_meanings():
    payload = _payload()
    index = payload.line_keys.index('co10')
    payload.total_dL_dv_erg_s_per_kms[0, index] = [1., 3.]
    payload.line_centroid_window_kms[0, index] = 17.
    payload.line_sigma_full_kms[0, index] = 9.
    total = payload.for_line('co10', 'attenuated', 'total')
    hot = payload.for_line('co10', 'attenuated', 'T_QUOKKA_ge_3000K')
    report = read_display_profile_statistics(payload)
    np.testing.assert_array_equal(total.dL_dv_erg_s_per_kms, [1., 3.])
    assert report['co10']['mean_velocity_kms'] == 17.
    assert report['co10']['sigma_line_full_kms'] == 9.
    assert hot.sigma_window_kms == payload.line_sigma_window_by_regime_kms[0, index, 0]
    assert hot.centroid_full_kms is None
    assert hot.sigma_full_kms is None

    from unittest.mock import Mock
    axis = Mock()
    draw_line_profile_curve(axis, total.velocity_kms, total.dL_dv_erg_s_per_kms,
                            total.sigma_full_kms)
    np.testing.assert_array_equal(axis.plot.call_args.args[0], [-1., 1.])


def test_gas_phase_curves_use_named_rows_and_their_own_velocity_grid():
    from unittest.mock import Mock
    phases = GasPhaseResults(
        phase_keys=('total', 'HIM', 'WIM', 'WNM', 'UNM', 'CNM'),
        phase_boundaries_K=np.array([200., 3000., 1.e4, 10.**5.5]),
        velocity_edges_kms=np.array([10., 20., 30.]),
        histogram_mass_g=np.array([[4., 2.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 3.]]),
        sigma_about_global_mean_kms=np.array([99., 0., 0., 0., 0., 7.]),
        sigma_internal_kms=np.array([8., 0., 0., 0., 0., 2.]),
    )
    axis = Mock()
    _, labels = draw_gas_mass_profile_curves(axis, 'co10', phases)
    assert len(axis.plot.call_args_list) == 2
    np.testing.assert_array_equal(axis.plot.call_args_list[0].args[0], [15., 25.])
    np.testing.assert_array_equal(axis.plot.call_args_list[0].args[1], [1. / 3., 1.])
    np.testing.assert_array_equal(axis.plot.call_args_list[1].args[0], [15., 25.])
    assert '7.0' in labels[0]
    assert '8.0' in labels[1]


def test_phase_cut_footer_uses_saved_boundaries_without_changing_existing_text():
    from dataclasses import replace
    phases = GasPhaseResults(
        phase_keys=('CNM', 'UNM', 'WNM', 'WIM', 'HIM', 'total'),
        phase_boundaries_K=np.array([200., 3000., 1.e4, 10.**5.5]),
        velocity_edges_kms=np.array([-1., 1.]),
        histogram_mass_g=np.zeros((6, 1)),
        sigma_about_global_mean_kms=np.zeros(6),
        sigma_internal_kms=np.zeros(6),
    )
    expected = (
        r'Phase cuts [K]: CNM $<200$; UNM $200$-$3000$; WNM $3000$-$10^4$; '
        r'WIM $10^4$-$10^{5.5}$; HIM $\geq10^{5.5}$.'
    )
    assert format_phase_cut_footer(phases) == expected
    changed = replace(phases, phase_boundaries_K=np.array([100., 2000., 1.e4, 1.e5]))
    assert format_phase_cut_footer(changed) == (
        r'Phase cuts [K]: CNM $<100$; UNM $100$-$2000$; WNM $2000$-$10^4$; '
        r'WIM $10^4$-$10^5$; HIM $\geq10^5$.'
    )


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
