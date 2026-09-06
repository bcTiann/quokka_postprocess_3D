import numpy as np

from quokka2s.pipeline.tasks.adopted_phase_hist import (
    PANELS, DexHistogram, select_emissivities,
)


def test_nine_panels_and_temperature_policy():
    assert [row[0] for row in PANELS] == [
        'mass_T_QK', 'mass_T_DSP', 'mass_T_2R', 'NH_rho',
        'halpha', 'hi21', 'cii', 'co10', 'co21',
    ]
    temperatures = {key: temp for key, temp, _ in PANELS}
    assert temperatures['cii'] == 'mixed'
    assert temperatures['co10'] == temperatures['co21'] == 'DESPOTIC'


def test_model_selection_at_3000_and_co_all_temperatures():
    tq = np.array([10., 2999., 3000., 1e7])
    cold = {key: np.full(4, 2.) for key in ('cii', 'halpha', 'hi21', 'co10', 'co21')}
    hot = {key: np.full(4, 7.) for key in ('cii', 'halpha', 'hi21')}
    values = select_emissivities(tq, cold, hot)
    for key in hot:
        np.testing.assert_array_equal(values[key], [2, 2, 7, 7])
    for key in ('co10', 'co21'):
        np.testing.assert_array_equal(values[key], [2, 2, 2, 2])


def test_streamed_histogram_matches_one_shot_and_conserves_weight():
    rng = np.random.default_rng(832)
    x, y, w = rng.normal(size=500), rng.normal(size=500), rng.uniform(size=500)
    streamed = DexHistogram()
    for start in range(0, x.size, 17):
        sl = slice(start, start + 17)
        streamed.add(x[sl], y[sl], w[sl])
    result = streamed.result()
    expected, _, _ = np.histogram2d(x, y, bins=(result['x_edges'], result['y_edges']), weights=w)
    np.testing.assert_allclose(result['H'], expected, rtol=1e-14)
    np.testing.assert_allclose(result['H'].sum(), w.sum(), rtol=1e-14)
    assert streamed.count == 500


def test_exact_bin_edges_remain_inside_histogram():
    h = DexHistogram(.2)
    h.add(np.array([0, 1, -1]), np.array([0, 1, -1]), np.ones(3))
    assert h.H.sum() == 3
    assert h.result()['x_edges'][-1] > 1


def test_no_silent_invalid_emission():
    import unittest
    cold = {key: np.array([np.nan]) for key in ('cii', 'halpha', 'hi21', 'co10', 'co21')}
    hot = {key: np.array([1.]) for key in ('cii', 'halpha', 'hi21')}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Invalid'):
        select_emissivities(np.array([100.]), cold, hot)


def test_lookup_inputs_and_emissivity_formulas():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from build_adopted_phase_histograms import emissivities
    from quokka2s.pipeline.prep.physics_fields import (
        _HI_emissivity_from_number_density, effective_halpha_recombination_coefficient,
        h, c, lambda_Halpha,
    )

    class Despotic:
        table = SimpleNamespace(nH_values=np.array([.1, 100.]),
                                col_density_values=np.array([1e18, 1e23]),
                                dVdr_values=np.array([1e-22, 1e-10]))

        def line_field(self, species, field, nh, column, dvdr):
            assert field == 'lumPerH'
            return np.full_like(nh, {'CO': 2., 'CO21': 3., 'C+': 5.}[species])

        def number_densities(self, species, nh, column, dvdr):
            return {'e-': nh * .1, 'H+': nh * .2, 'H': nh * .8}

    class Cloudy:
        line_keys = ('cii', 'halpha', 'hi21')

        def sample(self, temperature, nh, column):
            np.testing.assert_array_equal(temperature, [3000.])
            np.testing.assert_array_equal(nh, [4.])
            return SimpleNamespace(emissivity_per_nH2=np.array([[7.], [11.], [13.]]))

    values = emissivities(np.array([2999., 3000.]), np.array([1000., 100.]),
                          np.array([2., 4.]), np.array([1e20, 1e20]),
                          np.array([1e-15, 1e-15]), Despotic(), Cloudy())
    np.testing.assert_allclose(values['cii'], [10., 7. * 16])
    np.testing.assert_allclose(values['co10'], [4., 8.])
    np.testing.assert_allclose(values['co21'], [6., 12.])
    photon = float(((h * c) / lambda_Halpha).in_cgs().value)
    np.testing.assert_allclose(values['halpha'],
                              [photon * effective_halpha_recombination_coefficient(1000.) * .2 * .4,
                               11. * 16], rtol=1e-14, atol=0)
    np.testing.assert_allclose(values['hi21'],
                              [_HI_emissivity_from_number_density(1.6), 13. * 16],
                              rtol=1e-14, atol=0)


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
