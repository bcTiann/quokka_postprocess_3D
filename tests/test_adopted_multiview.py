import json
import unittest

import numpy as np

from quokka2s.adopted_multiview import MultiviewAccumulator


def _cube():
    x, y, z = np.indices((2, 3, 4))
    rho = 1. + 13. * x + 3. * y + z
    vz = -70. + 31. * x - 11. * y + 5. * z
    tq = 900. + 2300. * x + 130. * y + 51. * z
    mixed = np.where(tq < 3000., 40. + 23. * y + 3. * z, tq)
    valid = np.ones(rho.shape, dtype=bool)
    valid[:, 0, 0] = False  # Empty edge sightline.
    valid[0, 1, :] = False  # Empty face sightline, including its central slice.
    valid[1, 2, 2] = False  # A hole on both central slices.
    return rho, vz, tq, mixed, valid


def test_asymmetric_grid_projections_slices_and_independent_volume_totals():
    rho, vz, tq, mixed, valid = _cube()
    widths = np.array([2., 3., 5.])
    acc = MultiviewAccumulator(rho.shape, widths)
    # Each x slab is streamed independently, not materialised internally.
    for ix in range(2):
        acc.add(ix, *(array[ix:ix + 1] for array in (rho, vz, tq, mixed, valid)))
    arrays, report = acc.payload(), acc.report()
    json.dumps(report, allow_nan=False)
    for selection, selected in (("all", np.ones(rho.shape, bool)), ("valid", valid)):
        selected_rho = np.where(selected, rho, 0.)
        mass = selected_rho * np.prod(widths)
        totals = report["selections"][selection]
        assert totals["cell_count"] == selected.sum()
        np.testing.assert_allclose(totals["mass_g"], mass.sum())
        for field, array, total_key in (
            ("vz_kms", vz, "momentum_z_g_kms"),
            ("T_quokka_K", tq, "T_quokka_mass_g_K"),
            ("T_mixed_K", mixed, "T_mixed_mass_g_K"),
        ):
            np.testing.assert_allclose(totals[total_key], (mass * array).sum())
        for view, axis, dl, area in (("edge", 0, widths[0], widths[1] * widths[2]),
                                     ("face", 2, widths[2], widths[0] * widths[1])):
            prefix = f"{selection}_{view}_"
            sigma = arrays[prefix + "sigma_g_cm2"]
            np.testing.assert_allclose(sigma, selected_rho.sum(axis=axis) * dl)
            np.testing.assert_allclose(sigma.sum() * area, mass.sum())
            np.testing.assert_array_equal(arrays[prefix + "cell_count"], selected.sum(axis=axis))
            for field, array, total_key in (
                ("vz_kms", vz, "momentum_z_g_kms"),
                ("T_quokka_K", tq, "T_quokka_mass_g_K"),
                ("T_mixed_K", mixed, "T_mixed_mass_g_K"),
            ):
                # Independent ray-by-ray weighted averages distinguish axes.
                coordinates = np.ndindex(sigma.shape)
                expected = np.full(sigma.shape, np.nan)
                for pos in coordinates:
                    index = (slice(None), *pos) if axis == 0 else (*pos, slice(None))
                    ray_mask = selected[index]
                    if ray_mask.any():
                        expected[pos] = np.average(array[index][ray_mask], weights=rho[index][ray_mask])
                actual = arrays[prefix + field]
                np.testing.assert_allclose(actual, expected, equal_nan=True)
                # Integrating each weighted map recovers the independent 3D total.
                np.testing.assert_allclose(np.sum(np.where(sigma > 0, actual, 0.) * sigma) * area,
                                           totals[total_key], rtol=2e-14)
            index = (1, slice(None), slice(None)) if axis == 0 else (slice(None), slice(None), 2)
            expected_slice = np.where(selected[index], rho[index], np.nan)
            np.testing.assert_array_equal(arrays[prefix + "rho_g_cm3"], expected_slice)
    assert arrays["all_edge_sigma_g_cm2"].shape == (3, 4)
    assert arrays["all_face_sigma_g_cm2"].shape == (2, 3)
    assert np.isnan(arrays["valid_edge_vz_kms"][0, 0])
    assert np.isnan(arrays["valid_face_vz_kms"][0, 1])


def test_streamed_matches_one_slab_and_payload_does_not_expose_state():
    values = _cube()
    streamed = MultiviewAccumulator((2, 3, 4), [2., 3., 5.])
    full = MultiviewAccumulator((2, 3, 4), [2., 3., 5.]).add(0, *values)
    for ix in range(2):
        streamed.add(ix, *(array[ix:ix + 1] for array in values))
    expected = full.payload()
    for name, value in streamed.payload().items():
        np.testing.assert_allclose(value, expected[name], equal_nan=True, rtol=2e-14)
    for selection in ("all", "valid"):
        for key, value in streamed.report()["selections"][selection].items():
            np.testing.assert_allclose(value, full.report()["selections"][selection][key], rtol=2e-14)
    expected["all_edge_sigma_g_cm2"][:] = -1.
    assert np.all(full.payload()["all_edge_sigma_g_cm2"] > 0.)


def test_invalid_grid():
    for shape, widths in [
        ((2, 3), [1., 1., 1.]), ((2, 0, 4), [1., 1., 1.]),
        ((2., 3, 4), [1., 1., 1.]), ((2, 3, 4), [1., 1.]),
        ((2, 3, 4), [1., 0., 1.]), ((2, 3, 4), [1., np.nan, 1.]),
        ((2, 3, 4), [1e300, 1e300, 1e300]),
    ]:
        with unittest.TestCase().assertRaises(ValueError):
            MultiviewAccumulator(shape, widths)


def test_incomplete_order_bad_inputs_and_failed_add_leave_state_unchanged():
    values = _cube()
    acc = MultiviewAccumulator((2, 3, 4), [2., 3., 5.])
    for method in (acc.payload, acc.report):
        with unittest.TestCase().assertRaisesRegex(ValueError, "All x slabs"):
            method()
    with unittest.TestCase().assertRaisesRegex(ValueError, "ordered"):
        acc.add(1, *(array[:1] for array in values))
    acc.add(0, *(array[:1] for array in values))
    for field, bad in ((0, np.nan), (0, -1.), (1, np.inf),
                       (2, 0.), (3, np.nan)):
        invalid = [array[1:].copy() for array in values]
        invalid[field][0, 0, 0] = bad  # Invalid even though valid[1,0,0] is False.
        with unittest.TestCase().assertRaises(ValueError):
            acc.add(1, *invalid)
        assert acc.next_ix == 1
    with unittest.TestCase().assertRaisesRegex(ValueError, "boolean"):
        acc.add(1, *(array[1:] for array in values[:-1]), values[-1][1:].astype(int))
    with unittest.TestCase().assertRaisesRegex(ValueError, "overflowed"):
        acc.add(1, values[0][1:] * 1e300, values[1][1:] * 1e100,
                values[2][1:], values[3][1:], values[4][1:])
    acc.add(1, *(array[1:] for array in values))
    expected = MultiviewAccumulator((2, 3, 4), [2., 3., 5.]).add(0, *values)
    assert acc.report() == expected.report()
    for name, array in acc.payload().items():
        np.testing.assert_allclose(array, expected.payload()[name], equal_nan=True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "ordered"):
        acc.add(2, *(array[:1] for array in values))


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in list(globals().items())
                              if name.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
