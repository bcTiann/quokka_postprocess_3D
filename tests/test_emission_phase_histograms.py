import numpy as np

from quokka2s.products.emission_phase_histograms import PANELS, DexHistogram, accumulate_emission_phase_histograms


HIGH_ION_LINES = ('ciii_977', 'ciii_1907', 'ciii_1909', 'civ_1548', 'civ_1551')


def test_fourteen_panels_and_temperature_policy():
    assert [row[0] for row in PANELS] == [
        'mass_T_QK', 'mass_T_DSP', 'mass_T_2R', 'NH_rho',
        'halpha', 'hi21', 'cii', 'co10', 'co21',
        *HIGH_ION_LINES,
    ]
    temperatures = {key: temp for key, temp, _ in PANELS}
    assert temperatures['cii'] == 'mixed'
    assert temperatures['co10'] == temperatures['co21'] == 'DESPOTIC'
    assert all(temperatures[key] == 'QUOKKA' for key in HIGH_ION_LINES)
    # Each transition retains its own luminosity scale, rather than summing
    # the C III multiplet or C IV doublet into a combined panel.
    groups = {key: group for key, _, group in PANELS}
    assert all(groups[key] == key for key in HIGH_ION_LINES)


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


def _accepted_phase_chunk():
    from types import SimpleNamespace
    from quokka2s.physics.cell_emission import LineEmission

    rho = np.array([1e-25, 2e-25, 3e-25, 4e-25])
    tq = np.array([100., 2999., 3000., 1e6])
    td = np.array([21., 5000., 33., np.nan])
    column = np.array([1e18, 1e19, 1e20, 1e21])
    # Deliberately different order from PANELS; every transition is distinct.
    keys = ('civ_1551', 'co21', 'ciii_1907', 'hi21', 'ciii_977',
            'cii', 'civ_1548', 'halpha', 'ciii_1909', 'co10')
    epsilon = np.arange(1., 41.).reshape(10, 4)
    temperature = np.tile([21., 5000., 3000., 1e6], (10, 1))
    for key in ('co10', 'co21'):
        index = keys.index(key)
        epsilon[index, -1] = np.nan
        temperature[index] = td
    for key in HIGH_ION_LINES:
        epsilon[keys.index(key), :2] = 0.
    lines = {}
    for index, key in enumerate(keys):
        lines[key] = LineEmission(
            intrinsic_emissivity_erg_s_cm3=epsilon[index],
            attenuated_emissivity_erg_s_cm3=.5 * epsilon[index],
            temperature_K=temperature[index],
        )
    emission = SimpleNamespace(
        lines=lines,
        cold_cells=tq < 3000.,
        despotic_temperature_K=td,
    )
    return rho, tq, td, column, emission


def _phase_histograms():
    return {key: DexHistogram(.2) for key, _, _ in PANELS}


def _slice_phase_chunk(chunk, selected):
    from types import SimpleNamespace
    from quokka2s.physics.cell_emission import LineEmission

    rho, tq, td, column, emission = chunk
    lines = {}
    for key, line_emission in emission.lines.items():
        lines[key] = LineEmission(
            intrinsic_emissivity_erg_s_cm3=line_emission.intrinsic_emissivity_erg_s_cm3[selected],
            attenuated_emissivity_erg_s_cm3=line_emission.attenuated_emissivity_erg_s_cm3[selected],
            temperature_K=line_emission.temperature_K[selected],
        )
    return tuple(value[selected] for value in (rho, tq, td, column)) + (
        SimpleNamespace(
            lines=lines,
            cold_cells=emission.cold_cells[selected],
            despotic_temperature_K=td[selected],
        ),
    )


def _assert_panel_points(histogram, x, y, weights):
    panel = histogram.result()
    expected, _, _ = np.histogram2d(np.log10(x), np.log10(y),
        bins=(panel['x_edges'], panel['y_edges']), weights=weights)
    np.testing.assert_allclose(panel['H'], expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(histogram.total, np.sum(weights), rtol=1e-14, atol=0)
    assert histogram.count == len(weights)


def test_mass_and_line_panels_select_only_their_own_required_quantities():
    chunk = _accepted_phase_chunk()
    rho, tq, td, column, emission = chunk
    histograms = _phase_histograms()
    volume = np.array([2., 3., 4., 5.])
    accumulate_emission_phase_histograms(histograms, *chunk, volume)
    mass = rho * volume
    _assert_panel_points(histograms['mass_T_QK'], rho, tq, mass)
    _assert_panel_points(histograms['mass_T_DSP'], rho[:3], td[:3], mass[:3])
    _assert_panel_points(histograms['mass_T_2R'], rho, [21., 5000., 3000., 1e6], mass)
    _assert_panel_points(histograms['NH_rho'], column, rho, mass)
    for key, line_emission in emission.lines.items():
        if key in ('co10', 'co21'):
            indices = slice(0, 3)
            temperature = td[:3]
        else:
            indices = slice(None)
            temperature = [21., 5000., 3000., 1e6]
        _assert_panel_points(
            histograms[key],
            rho[indices],
            temperature,
            line_emission.intrinsic_emissivity_erg_s_cm3[indices] * volume[indices],
        )


def test_high_ions_keep_separate_hot_luminosities_and_zero_cold_contribution():
    chunk = _accepted_phase_chunk()
    rho, tq, td, _, emission = chunk
    histograms = _phase_histograms()
    volume = np.array([2., 3., 4., 5.])
    accumulate_emission_phase_histograms(histograms, *chunk, volume)
    # A cold TQ cell with TD > 3000 K still contributes no high-ion light.
    # The TQ=3000 K cell and the hot cell with missing TD both contribute.
    assert tq[1] < 3000. < td[1]
    assert td[2] < 3000. == tq[2]
    totals = []
    for key in HIGH_ION_LINES:
        panel = histograms[key].result()
        epsilon = emission.lines[key].intrinsic_emissivity_erg_s_cm3
        _assert_panel_points(histograms[key], rho, [21., 5000., 3000., 1e6], epsilon * volume)
        assert np.count_nonzero(panel['H']) == 2
        assert histograms[key].count == 4
        totals.append(histograms[key].total)
    assert len(set(totals)) == len(HIGH_ION_LINES)


def test_missing_one_line_does_not_remove_other_lines_or_gas_mass():
    chunk = _accepted_phase_chunk()
    rho, tq, td, column, emission = chunk
    emission.lines['halpha'].intrinsic_emissivity_erg_s_cm3[0] = np.nan
    emission.lines['hi21'].intrinsic_emissivity_erg_s_cm3[1] = np.inf
    # Also omit one CII temperature without changing its available emissivity.
    emission.lines['cii'].temperature_K[1] = np.nan
    histograms = _phase_histograms()
    volume = np.array([2., 3., 4., 5.])
    accumulate_emission_phase_histograms(histograms, *chunk, volume)
    _assert_panel_points(histograms['mass_T_QK'], rho, tq, rho * volume)
    _assert_panel_points(histograms['mass_T_2R'], rho, [21., 5000., 3000., 1e6], rho * volume)
    _assert_panel_points(histograms['halpha'], rho[1:], [5000., 3000., 1e6],
                         emission.lines['halpha'].intrinsic_emissivity_erg_s_cm3[1:] * volume[1:])
    chosen = np.array([0, 2, 3])
    _assert_panel_points(histograms['cii'], rho[chosen], [21., 3000., 1e6],
                         emission.lines['cii'].intrinsic_emissivity_erg_s_cm3[chosen] * volume[chosen])
    _assert_panel_points(histograms['hi21'], rho[chosen], [21., 3000., 1e6],
                         emission.lines['hi21'].intrinsic_emissivity_erg_s_cm3[chosen] * volume[chosen])
    assert histograms['civ_1548'].count == 4


def test_missing_cold_temperature_removes_only_temperature_dependent_mass():
    chunk = _accepted_phase_chunk()
    rho, tq, td, column, emission = chunk
    td[1] = np.nan
    histograms = _phase_histograms()
    accumulate_emission_phase_histograms(histograms, *chunk, 2.)
    _assert_panel_points(histograms['mass_T_QK'], rho, tq, rho * 2.)
    _assert_panel_points(histograms['NH_rho'], column, rho, rho * 2.)
    _assert_panel_points(histograms['mass_T_DSP'], rho[[0, 2]], td[[0, 2]], rho[[0, 2]] * 2.)
    _assert_panel_points(histograms['mass_T_2R'], rho[[0, 2, 3]], [21., 3000., 1e6],
                         rho[[0, 2, 3]] * 2.)


def test_float32_temperature_keeps_physical_dex_bin_and_luminosity():
    from quokka2s.physics.cell_emission import LineEmission

    chunk = _accepted_phase_chunk()
    rho, tq, td, column, emission = chunk
    # float32 log10 rounds this value to 2.2, although the stored physical
    # temperature is below 10**2.2 K. Coordinates must be evaluated in float64.
    boundary_temperature = np.float32(10. ** 2.2)
    assert np.log10(float(boundary_temperature)) < 2.2
    tq[:] = 100.
    td[:] = float(boundary_temperature)
    td[-1] = np.nan
    emission.cold_cells[:] = True
    for key, line in list(emission.lines.items()):
        thermal_temperature = np.full(4, boundary_temperature, dtype=np.float32)
        thermal_temperature[-1] = np.nan
        emission.lines[key] = LineEmission(
            intrinsic_emissivity_erg_s_cm3=line.intrinsic_emissivity_erg_s_cm3,
            attenuated_emissivity_erg_s_cm3=line.attenuated_emissivity_erg_s_cm3,
            temperature_K=thermal_temperature,
        )
    histograms = _phase_histograms()
    volume = np.array([2., 3., 4., 5.])
    accumulate_emission_phase_histograms(histograms, *chunk, volume)
    temperature = np.full(3, float(boundary_temperature))
    _assert_panel_points(histograms['mass_T_2R'], rho[:3], temperature, rho[:3] * volume[:3])
    for key, line in emission.lines.items():
        _assert_panel_points(histograms[key], rho[:3], temperature,
                            line.intrinsic_emissivity_erg_s_cm3[:3] * volume[:3])


def test_independent_selections_match_streaming_and_empty_chunks():
    chunk = _accepted_phase_chunk()
    one_shot, streamed = _phase_histograms(), _phase_histograms()
    accumulate_emission_phase_histograms(one_shot, *chunk, 2.)
    # Begin with a hot cell whose TD/CO are missing; other panels still need it.
    for selected in (slice(3, 4), slice(0, 2), slice(2, 3), slice(0, 0)):
        accumulate_emission_phase_histograms(streamed, *_slice_phase_chunk(chunk, selected), 2.)
    for key, _, _ in PANELS:
        np.testing.assert_allclose(streamed[key].H, one_shot[key].H, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(streamed[key].origin, one_shot[key].origin)
        assert streamed[key].count == one_shot[key].count
        np.testing.assert_allclose(streamed[key].total, one_shot[key].total, rtol=1e-14)


def test_invalid_inputs_leave_all_histograms_unchanged():
    import unittest
    from quokka2s.physics.cell_emission import LineEmission

    def invalid(case):
        rho, tq, td, column, emission = _accepted_phase_chunk()
        volume = 2.
        if case == 'rho': rho[0] = 0.
        elif case == 'column': column[0] = np.nan
        elif case == 'tq': tq[-1] = np.nan
        elif case == 'mask': emission.cold_cells = emission.cold_cells[:-1]
        elif case == 'mask_type': emission.cold_cells = emission.cold_cells.astype(int)
        elif case == 'coordinate_shape': td = td[:-1]
        elif case == 'emission_shape':
            line = emission.lines['halpha']
            emission.lines['halpha'] = LineEmission(
                intrinsic_emissivity_erg_s_cm3=np.ones(3),
                attenuated_emissivity_erg_s_cm3=np.ones(3),
                temperature_K=line.temperature_K,
            )
        elif case == 'negative_epsilon': emission.lines['halpha'].intrinsic_emissivity_erg_s_cm3[0] = -1.
        elif case == 'missing_line': del emission.lines['halpha']
        elif case == 'volume': volume = -2.
        elif case == 'volume_nan': volume = np.nan
        elif case == 'volume_shape': volume = np.ones(3)
        return rho, tq, td, column, emission, volume

    for case in ('rho', 'column', 'tq', 'mask', 'mask_type', 'coordinate_shape',
                 'emission_shape', 'negative_epsilon', 'missing_line',
                 'volume', 'volume_nan', 'volume_shape'):
        histograms = _phase_histograms()
        with unittest.TestCase().assertRaises(ValueError, msg=case):
            accumulate_emission_phase_histograms(histograms, *invalid(case))
        assert all(histogram.H is None and histogram.count == 0
                   for histogram in histograms.values()), case


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
