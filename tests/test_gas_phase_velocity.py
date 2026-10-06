import json
import unittest

import numpy as np

from quokka2s.products.gas_phase_velocity import (
    GasPhaseVelocityAccumulator,
    PHASE_ORDER,
    PHASE_BOUNDS_K,
    check_phase_accounting,
)


def _direct_moments(v, mass, global_mean):
    mean = np.average(v, weights=mass)
    return mean, np.sqrt(np.average((v - mean)**2, weights=mass)), np.sqrt(
        np.average((v - global_mean)**2, weights=mass))


def _assert_moments(result, velocity, mass, global_mean):
    assert result["count"] == len(velocity)
    np.testing.assert_allclose(result["mass_g"], mass.sum(), rtol=2e-14)
    expected = _direct_moments(velocity, mass, global_mean)
    actual = [result["mean_velocity_kms"], result["sigma_internal_kms"],
              result["sigma_about_global_mean_kms"]]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-12)


def test_exact_phase_boundaries_match_existing_classification():
    t = np.array([1., *[value for cut in PHASE_BOUNDS_K
                        for value in (np.nextafter(cut, 0.), cut)], 1e9])
    acc = GasPhaseVelocityAccumulator([-1., 0., 1.])
    acc.add(np.zeros(t.size), t, np.ones(t.size), 1.)
    report = acc.report()
    for key in PHASE_ORDER:
        assert report["groups"][key]["count"] == 2
    assert report["groups"]["total"]["count"] == t.size


def test_streamed_matches_direct_mass_weighted_moments_and_histograms():
    rng = np.random.default_rng(691)
    v = rng.normal(14., 170., 997)
    t = 10.**rng.uniform(1., 7., v.size)
    rho = 10.**rng.uniform(-28., -21., v.size)
    volume = 10.**rng.uniform(49., 56., v.size)
    mass = rho * volume
    edges = np.linspace(-200., 200., 301)
    acc, once = (GasPhaseVelocityAccumulator(edges) for _ in range(2))
    once.add(v, t, rho, volume)
    for start in range(0, v.size, 37):
        sl = slice(start, start + 37)
        acc.add(v[sl], t[sl], rho[sl], volume[sl])
    report, hist = acc.report(), acc.histogram_mass_g
    global_mean = np.average(v, weights=mass)
    phase_index = np.searchsorted(PHASE_BOUNDS_K, t, side="right")
    for index, key in enumerate((*PHASE_ORDER, "total")):
        selected = phase_index == index if key != "total" else np.ones(v.size, bool)
        vg, mg = v[selected], mass[selected]
        _assert_moments(report["groups"][key], vg, mg, global_mean)
        inside = (vg >= edges[0]) & (vg <= edges[-1])
        _assert_moments(report["groups"][key]["in_window"], vg[inside], mg[inside], global_mean)
        # Sum each bin directly. A weighted cumulative histogram subtracts
        # large nearby totals and is a poor reference for this mass range.
        direct_hist = np.array([mg[(vg >= lo) & ((vg < hi) if i < edges.size - 2
                               else (vg <= hi))].sum()
                               for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))])
        np.testing.assert_allclose(hist[key], direct_hist, rtol=2e-14, atol=0.)
        np.testing.assert_allclose(hist[key], once.histogram_mass_g[key],
                                   rtol=2e-14, atol=0.)
        np.testing.assert_allclose(report["groups"][key]["sigma_internal_kms"],
            once.report()["groups"][key]["sigma_internal_kms"], rtol=2e-14)
        group = report["groups"][key]
        assert (group["in_window"]["count"] + group["below_window"]["count"]
                + group["above_window"]["count"]) == group["count"]
        np.testing.assert_allclose(group["histogram_mass_g"]
            + group["below_window"]["mass_g"] + group["above_window"]["mass_g"],
            group["mass_g"], rtol=2e-14)
    np.testing.assert_allclose(sum(hist[key] for key in PHASE_ORDER), hist["total"],
                               rtol=2e-14, atol=0.)
    assert sum(report["groups"][key]["count"] for key in PHASE_ORDER) == v.size
    np.testing.assert_allclose(sum(report["groups"][key]["mass_g"] for key in PHASE_ORDER), mass.sum())
    # Phase second moments about the shared global mean partition the total.
    variance = sum(report["groups"][key]["mass_fraction"] *
                   report["groups"][key]["sigma_about_global_mean_kms"]**2
                   for key in PHASE_ORDER)
    np.testing.assert_allclose(variance, report["groups"]["total"]["sigma_internal_kms"]**2)


def test_outside_window_never_clamps_and_still_affects_full_range_width():
    v = np.array([-900., -200., -10., 0., 10., 200., 700.])
    mass = np.array([1., 2., 3., 4., 5., 6., 7.])
    acc = GasPhaseVelocityAccumulator([-200., 0., 200.]).add(
        v, np.full(v.size, 100.), mass, 1.)
    report = acc.report()["groups"]["CNM"]
    np.testing.assert_array_equal(acc.histogram_mass_g["CNM"], [5., 15.])
    assert report["below_window"] == {"count": 1, "mass_g": 1.}
    assert report["above_window"] == {"count": 1, "mass_g": 7.}
    assert report["in_window"]["count"] == 5
    assert report["mass_g"] == (report["histogram_mass_g"]
        + report["below_window"]["mass_g"] + report["above_window"]["mass_g"])
    assert report["sigma_internal_kms"] > report["in_window"]["sigma_internal_kms"]
    _assert_moments(report, v, mass, np.average(v, weights=mass))
    _assert_moments(report["in_window"], v[1:-1], mass[1:-1], np.average(v, weights=mass))


def test_realistic_mass_contrast_keeps_small_velocity_tail_bins():
    rng = np.random.default_rng(151)
    n = 100000
    velocity = rng.normal(0., 50., n)
    temperature = 10.**rng.uniform(1., 8., n)
    density = 10.**rng.uniform(-28., -20., n)
    volume = 1.638467e57
    edges = np.linspace(-200., 200., 301)
    acc = GasPhaseVelocityAccumulator(edges)
    for start in range(0, n, 4096):
        sl = slice(start, start + 4096)
        acc.add(velocity[sl], temperature[sl], density[sl], volume)
    hist = acc.histogram_mass_g
    # The independent all-gas and per-phase sums must agree even in small
    # tail bins. A generic weighted cumulative histogram fails this check.
    np.testing.assert_allclose(sum(hist[key] for key in PHASE_ORDER), hist["total"],
                               rtol=1e-12, atol=0.)
    mass = density * volume
    direct = np.array([mass[(velocity >= lo) & ((velocity < hi) if i < 299
                       else (velocity <= hi))].sum()
                       for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))])
    np.testing.assert_allclose(hist["total"], direct, rtol=1e-12, atol=0.)
    assert np.count_nonzero(hist["total"][:30]) > 0
    assert np.count_nonzero(hist["total"][-30:]) > 0


def test_phase_dispersion_about_global_mean_differs_from_internal_width():
    acc = GasPhaseVelocityAccumulator([-200., 200.]).add(
        [-30., -10., 90., 110.], [100., 100., 1e6, 1e6], [1., 1., 1., 1.], 1.)
    report = acc.report()
    assert report["global_mean_velocity_kms"] == 40.
    for key in ("CNM", "HIM"):
        assert report["groups"][key]["sigma_internal_kms"] == 10.
        np.testing.assert_allclose(report["groups"][key]["sigma_about_global_mean_kms"],
                                   np.sqrt(3700.))


def test_volume_changes_mass_weights_and_scalar_matches_constant_array():
    v, t, rho = np.array([0., 10.]), np.array([100., 100.]), np.array([1., 2.])
    variable = GasPhaseVelocityAccumulator([-200., 200.]).add(v, t, rho, [9., 0.5])
    assert variable.report()["groups"]["total"]["mean_velocity_kms"] == 1.
    scalar = GasPhaseVelocityAccumulator([-200., 200.]).add(v, t, rho, 3.)
    array = GasPhaseVelocityAccumulator([-200., 200.]).add(v, t, rho, [3., 3.])
    assert scalar.report() == array.report()


def test_large_bulk_velocity_preserves_small_dispersion():
    # A raw sum(v^2)/sum(m) - mean^2 computation cancels to zero here.
    v = 1e9 + np.array([-0.5, -0.25, 0.25, 0.5])
    acc = GasPhaseVelocityAccumulator([1e9 - 1., 1e9 + 1.])
    for sl in (slice(0, 2), slice(2, 4)):
        acc.add(v[sl], np.full(2, 100.), np.ones(2), 1.)
    np.testing.assert_allclose(acc.report()["groups"]["total"]["sigma_internal_kms"],
                               np.sqrt(0.15625), rtol=1e-14)


def test_empty_chunks_phases_and_windows_have_json_safe_none_moments():
    acc = GasPhaseVelocityAccumulator([-200., 200.])
    original = acc.report()
    acc.add([], [], [], 1.)
    assert acc.report() == original
    json.dumps(original, allow_nan=False)
    acc.add([900.], [100.], [2.], 3.)
    result = acc.report()
    assert result["groups"]["CNM"]["sigma_internal_kms"] == 0.
    assert result["groups"]["CNM"]["in_window"]["mean_velocity_kms"] is None
    assert result["groups"]["HIM"]["sigma_internal_kms"] is None
    assert result["groups"]["HIM"]["mass_fraction"] == 0.
    json.dumps(result, allow_nan=False)


def test_invalid_input_leaves_existing_state_unchanged():
    acc = GasPhaseVelocityAccumulator([-200., 200.]).add([10.], [100.], [2.], 3.)
    before, histogram = acc.report(), acc.histogram_mass_g
    for field, value in [
        (0, [np.nan]), (0, [np.inf]), (0, [[1.]]),
        (1, [0.]), (1, [-1.]), (1, [np.nan]), (1, []),
        (2, [0.]), (2, [-1.]), (2, [np.inf]), (2, []),
        (3, 0.), (3, -1.), (3, np.nan), (3, [1., 2.]), (3, [[1.]]),
    ]:
        args = [[10.], [100.], [2.], 3.]
        args[field] = value
        with unittest.TestCase().assertRaises(ValueError):
            acc.add(*args)
        assert before == acc.report()
        for key in histogram:
            np.testing.assert_array_equal(histogram[key], acc.histogram_mass_g[key])


def test_mass_overflow_and_underflow_rejected_without_partial_update():
    acc = GasPhaseVelocityAccumulator([-200., 200.])
    before = acc.report()
    for density, volume in ((1e308, 1e308), (1e-300, 1e-300)):
        with unittest.TestCase().assertRaisesRegex(ValueError, "Cell mass"):
            acc.add([0.], [100.], [density], volume)
        assert before == acc.report()


def test_batch_gas_uses_its_temperature_even_when_all_line_emissivities_are_missing():
    from types import SimpleNamespace

    cells = SimpleNamespace(
        temperature_QUOKKA_K=np.array([100., 2999., 3000., 1e6]),
        velocity_z_kms=np.array([-30., -10., 20., 80.]),
        density_g_cm3=np.array([1., 2., 3., 4.]),
        cell_volume_cm3=2.,
    )
    emission = SimpleNamespace(
        cold_cells=np.array([True, True, False, False]),
        despotic_temperature_K=np.array([100., np.nan, np.nan, np.nan]),
        lines={},
    )
    accumulated = GasPhaseVelocityAccumulator([-200., 0., 200.])
    accumulated.add_batch(cells=cells, emission=emission)
    # Only the cold cell with missing TD is unavailable. Both hot cells use TQ,
    # regardless of their missing TD or any emission-line lookup failures.
    expected = GasPhaseVelocityAccumulator([-200., 0., 200.])
    selected = np.array([0, 2, 3])
    expected.add(
        velocity_kms=cells.velocity_z_kms[selected],
        phase_temperature_K=np.array([100., 3000., 1e6]),
        density_g_cm3=cells.density_g_cm3[selected],
        cell_volume_cm3=2.,
    )
    assert accumulated.report() == expected.report()
    for group in (*PHASE_ORDER, "total"):
        np.testing.assert_array_equal(
            accumulated.histogram_mass_g[group],
            expected.histogram_mass_g[group],
        )
    assert accumulated.report()['groups']['total']['count'] == 3
    assert accumulated.report()['groups']['total']['mass_g'] == 16.
    payload, report = accumulated.build_output()
    check_phase_accounting(
        groups=report['groups'],
        mass_by_bin=payload['histogram_mass_g'],
        gas_cell_count=3,
        gas_mass_g=16.,
    )


def test_batch_gas_keeps_cold_mass_when_temperature_is_available_without_emission():
    from types import SimpleNamespace

    cells = SimpleNamespace(
        temperature_QUOKKA_K=np.array([100., 100.]),
        velocity_z_kms=np.array([-10., 10.]),
        density_g_cm3=np.array([1., 3.]),
        cell_volume_cm3=1.,
    )
    emission = SimpleNamespace(
        cold_cells=np.ones(2, dtype=bool),
        despotic_temperature_K=np.array([50., 5000.]),
        lines={},
    )
    accumulated = GasPhaseVelocityAccumulator([-200., 0., 200.])
    accumulated.add_batch(cells=cells, emission=emission)
    report = accumulated.report()
    assert report['groups']['CNM']['mass_g'] == 1.
    assert report['groups']['WNM']['mass_g'] == 3.
    assert report['groups']['total']['count'] == 2


def test_batch_with_only_missing_cold_temperatures_leaves_gas_state_empty():
    from types import SimpleNamespace

    cells = SimpleNamespace(
        temperature_QUOKKA_K=np.array([100., 200.]),
        velocity_z_kms=np.array([-10., 10.]),
        density_g_cm3=np.ones(2),
        cell_volume_cm3=1.,
    )
    emission = SimpleNamespace(
        cold_cells=np.ones(2, dtype=bool),
        despotic_temperature_K=np.full(2, np.nan),
    )
    accumulated = GasPhaseVelocityAccumulator([-200., 0., 200.])
    before = accumulated.report()
    accumulated.add_batch(cells=cells, emission=emission)
    assert accumulated.report() == before


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
