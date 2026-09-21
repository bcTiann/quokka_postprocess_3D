import numpy as np

from quokka2s.pipeline.tasks.adopted_phase_hist import (
    PANELS, DexHistogram, select_emissivities, add_adopted_phase_chunk,
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


def _accepted_phase_chunk():
    from types import SimpleNamespace
    rho = np.array([1e-25, 2e-25, 3e-25, 4e-25])
    tq = np.array([100., 2999., 3000., 1e6])
    td = np.array([21., 5000., 33., np.nan])
    column = np.array([1e18, 1e19, 1e20, 1e21])
    # Deliberately different order from PANELS, and an unplotted line.
    keys = ('co21', 'hi21', 'ciii_977', 'cii', 'halpha', 'co10')
    epsilon = np.arange(1., 25.).reshape(6, 4)
    epsilon[:, -1] = np.nan
    temperature = np.tile([21., 5000., 3000., np.nan], (6, 1))
    for key in ('co10', 'co21'):
        temperature[keys.index(key)] = td
    emission = SimpleNamespace(line_keys=keys, emissivity_erg_s_cm3=epsilon,
        thermal_temperature_K=temperature, valid=np.array([True, True, True, False]),
        excluded=np.array([False, False, False, True]))
    return rho, tq, td, column, emission


def _phase_histograms():
    return {key: DexHistogram(.2) for key, _, _ in PANELS}


def _slice_phase_chunk(chunk, selected):
    from types import SimpleNamespace
    rho, tq, td, column, emission = chunk
    return tuple(value[selected] for value in (rho, tq, td, column)) + (
        SimpleNamespace(line_keys=emission.line_keys,
            emissivity_erg_s_cm3=emission.emissivity_erg_s_cm3[:, selected],
            thermal_temperature_K=emission.thermal_temperature_K[:, selected],
            valid=emission.valid[selected], excluded=emission.excluded[selected]),)


def _assert_panel_points(histogram, x, y, weights):
    panel = histogram.result()
    expected, _, _ = np.histogram2d(np.log10(x), np.log10(y),
        bins=(panel['x_edges'], panel['y_edges']), weights=weights)
    np.testing.assert_allclose(panel['H'], expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(histogram.total, np.sum(weights), rtol=1e-14, atol=0)
    assert histogram.count == len(weights)


def test_adopted_chunk_uses_supplied_emission_and_per_line_temperatures():
    chunk = _accepted_phase_chunk()
    rho, tq, td, column, emission = chunk
    histograms = _phase_histograms()
    volume = np.array([2., 3., 4., 5.])
    add_adopted_phase_chunk(histograms, *chunk, volume)
    valid = emission.valid
    mass = rho[valid] * volume[valid]
    _assert_panel_points(histograms['mass_T_QK'], rho[valid], tq[valid], mass)
    _assert_panel_points(histograms['mass_T_DSP'], rho[valid], td[valid], mass)
    # A cold QK cell may have a hotter DESPOTIC temperature; the 3000 K cell
    # still uses QK T. This is the canonical emission's chosen temperature.
    _assert_panel_points(histograms['mass_T_2R'], rho[valid], [21., 5000., 3000.], mass)
    _assert_panel_points(histograms['NH_rho'], column[valid], rho[valid], mass)
    for key in ('halpha', 'hi21', 'cii', 'co10', 'co21'):
        index = emission.line_keys.index(key)
        temperature = td[valid] if key.startswith('co') else [21., 5000., 3000.]
        _assert_panel_points(histograms[key], rho[valid], temperature,
                            emission.emissivity_erg_s_cm3[index, valid] * volume[valid])


def test_adopted_chunk_raw_mass_policy_keeps_only_raw_panels_all_cell():
    chunk = _accepted_phase_chunk()
    rho, tq, _, column, _ = chunk
    histograms = _phase_histograms()
    add_adopted_phase_chunk(histograms, *chunk, 2., raw_mass_all_cells=True)
    _assert_panel_points(histograms['mass_T_QK'], rho, tq, rho * 2.)
    _assert_panel_points(histograms['NH_rho'], column, rho, rho * 2.)
    for key, _, _ in PANELS:
        assert histograms[key].count == (4 if key in ('mass_T_QK', 'NH_rho') else 3)


def test_adopted_chunk_streaming_and_empty_retained_chunks():
    chunk = _accepted_phase_chunk()
    for raw_all in (False, True):
        one_shot, streamed = _phase_histograms(), _phase_histograms()
        add_adopted_phase_chunk(one_shot, *chunk, 2., raw_mass_all_cells=raw_all)
        # Start with a fully excluded chunk (all emission/TD are NaN).
        for selected in (slice(3, 4), slice(0, 2), slice(2, 3), slice(0, 0)):
            add_adopted_phase_chunk(streamed, *_slice_phase_chunk(chunk, selected),
                                    2., raw_mass_all_cells=raw_all)
        for key, _, _ in PANELS:
            np.testing.assert_allclose(streamed[key].H, one_shot[key].H, rtol=1e-14, atol=0)
            np.testing.assert_array_equal(streamed[key].origin, one_shot[key].origin)
            assert streamed[key].count == one_shot[key].count
            np.testing.assert_allclose(streamed[key].total, one_shot[key].total, rtol=1e-14)


def test_adopted_chunk_rejects_invalid_inputs_without_partial_accumulation():
    import unittest
    def invalid(case):
        rho, tq, td, column, emission = _accepted_phase_chunk()
        volume = 2.
        if case == 'rho': rho[0] = 0.
        elif case == 'column': column[0] = np.nan
        elif case == 'tq': tq[-1] = np.nan  # Raw simulation values must remain valid.
        elif case == 'td': td[0] = np.nan
        elif case == 'mask': emission.excluded[0] = True
        elif case == 'mask_type': emission.valid = emission.valid.astype(int)
        elif case == 'coordinate_shape': td = td[:-1]
        elif case == 'emission_shape': emission.emissivity_erg_s_cm3 = emission.emissivity_erg_s_cm3[:, :-1]
        elif case == 'thermal': emission.thermal_temperature_K[0, 0] = 0.
        elif case == 'epsilon': emission.emissivity_erg_s_cm3[0, 0] = np.nan
        elif case == 'negative_epsilon': emission.emissivity_erg_s_cm3[0, 0] = -1.
        elif case == 'missing_line': emission.line_keys = ('unknown', *emission.line_keys[1:])
        elif case == 'duplicate_line': emission.line_keys = ('co10', *emission.line_keys[1:])
        elif case == 'volume': volume = -2.
        elif case == 'volume_nan': volume = np.nan
        elif case == 'volume_shape': volume = np.ones(3)
        return rho, tq, td, column, emission, volume

    for case in ('rho', 'column', 'tq', 'td', 'mask', 'mask_type', 'coordinate_shape',
                 'emission_shape', 'thermal', 'epsilon', 'negative_epsilon', 'missing_line',
                 'duplicate_line', 'volume', 'volume_nan', 'volume_shape'):
        histograms = _phase_histograms()
        with unittest.TestCase().assertRaises(ValueError, msg=case):
            add_adopted_phase_chunk(histograms, *invalid(case))
        assert all(histogram.H is None and histogram.count == 0
                   for histogram in histograms.values()), case


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
