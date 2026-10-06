"""Check complete Gaussian moments independently of the saved velocity window."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import quad

from quokka2s.constants import (
    ATOMIC_MASS_UNIT_G,
    BOLTZMANN_ERG_K,
    SPEED_OF_LIGHT_KMS,
)
from quokka2s.physics.cell_emission import LineEmission
from quokka2s.line_definitions import LINE_DEFINITIONS
from quokka2s.products.integrated_spectra import IntegratedSpectra
from quokka2s.products.line_velocity_moments import LineVelocityMoments


def direct_cell_moments(velocity, width, luminosity):
    """Independent two-pass reference, using long double for the reductions."""
    emitting = luminosity > 0.0
    velocity = np.asarray(velocity[emitting], dtype=np.longdouble)
    width = np.asarray(np.maximum(width[emitting], 1.e-5), dtype=np.longdouble)
    luminosity = np.asarray(luminosity[emitting], dtype=np.longdouble)
    total = luminosity.sum()
    if total == 0.0:
        return 0.0, np.nan, np.nan, np.nan
    offset = velocity[0]
    relative_mean = np.sum(luminosity * (velocity - offset)) / total
    mean = offset + relative_mean
    variance = np.sum(luminosity * ((velocity - offset - relative_mean)**2 + width**2)) / total
    return float(total), float(mean), float(np.sqrt(variance)), float(variance + mean**2)


class FullGaussianMomentTests(unittest.TestCase):
    def test_complete_moments_agree_with_numeric_gaussian_integration(self):
        velocity = np.array([-8.0, 14.0, 500.0])
        width = np.array([2.0, 6.0, 11.0])
        luminosity = np.array([3.0, 7.0, 2.0])
        result = LineVelocityMoments.from_cells(
            velocity_kms=velocity,
            thermal_width_kms=width,
            luminosity_erg_s=luminosity,
        )

        def profile(speed):
            gaussian = np.exp(-0.5 * ((speed - velocity) / width)**2)
            return float(np.sum(luminosity * gaussian / (np.sqrt(2.0 * np.pi) * width)))

        # Split the integral around every peak so quadrature cannot miss a
        # narrow Gaussian far from the main emission near zero velocity.
        breakpoints = np.sort(np.concatenate((velocity - 8.0 * width, velocity, velocity + 8.0 * width)))
        bounds = dict(a=-100.0, b=650.0, points=breakpoints, epsabs=1.e-9, epsrel=1.e-11)
        total = quad(profile, **bounds)[0]
        centroid = quad(lambda speed: speed * profile(speed), **bounds)[0] / total
        variance = quad(lambda speed: (speed - centroid)**2 * profile(speed), **bounds)[0] / total
        self.assertAlmostEqual(result.luminosity_erg_s, total, places=10)
        self.assertAlmostEqual(result.centroid_kms, centroid, places=9)
        self.assertAlmostEqual(result.sigma_kms, np.sqrt(variance), places=8)

    def test_two_cells_include_emission_beyond_200_kms(self):
        result = LineVelocityMoments.from_cells(
            velocity_kms=np.array([0.0, 500.0]),
            thermal_width_kms=np.array([3.0, 3.0]),
            luminosity_erg_s=np.array([1.0, 1.0]),
        )
        self.assertEqual(result.centroid_kms, 250.0)
        self.assertAlmostEqual(result.sigma_kms, np.sqrt(250.0**2 + 3.0**2))
        self.assertAlmostEqual(result.raw_second_moment_kms2, 125009.0)

    def test_thermal_floor_matches_the_channel_kernel(self):
        result = LineVelocityMoments.from_cells(
            velocity_kms=np.array([25.0]),
            thermal_width_kms=np.array([1.e-12]),
            luminosity_erg_s=np.array([4.0]),
        )
        self.assertEqual(result.centroid_kms, 25.0)
        self.assertEqual(result.sigma_kms, 1.e-5)

    def test_empty_emission_has_no_centroid_or_dispersion(self):
        result = LineVelocityMoments.from_cells(
            velocity_kms=np.array([-300.0, 500.0]),
            thermal_width_kms=np.array([10.0, 20.0]),
            luminosity_erg_s=np.zeros(2),
        )
        self.assertEqual(result.luminosity_erg_s, 0.0)
        self.assertTrue(np.isnan(result.sigma_kms))
        self.assertTrue(np.isnan(result.raw_second_moment_kms2))
        nonempty = LineVelocityMoments.from_cells(
            velocity_kms=np.array([5.0]),
            thermal_width_kms=np.array([2.0]),
            luminosity_erg_s=np.array([3.0]),
        )
        self.assertEqual(result.merged(nonempty), nonempty)
        self.assertEqual(nonempty.merged(result), nonempty)

    def test_large_velocity_offset_does_not_erase_a_small_width(self):
        velocities = np.array([-3.0, 1.0, 4.0]) + 1.e12
        widths = np.array([0.1, 0.3, 0.2])
        light = np.array([1.0, 2.0, 3.0])
        result = LineVelocityMoments.from_cells(
            velocity_kms=velocities,
            thermal_width_kms=widths,
            luminosity_erg_s=light,
        )
        expected = direct_cell_moments(velocities, widths, light)
        self.assertEqual(result.centroid_kms, expected[1])
        np.testing.assert_allclose(result.sigma_kms, expected[2], rtol=2.e-9, atol=0.0)

    def test_independent_worker_moments_merge_to_the_serial_result(self):
        velocity = np.array([-180.0, -20.0, 0.0, 13.0, 500.0])
        width = np.array([2.0, 1.e-10, 3.0, 6.0, 20.0])
        light = np.array([1.0, 0.0, 3.0, 7.0, 2.0])
        expected = direct_cell_moments(velocity, width, light)

        def calculate(indices):
            return LineVelocityMoments.from_cells(
                velocity_kms=velocity[indices],
                thermal_width_kms=width[indices],
                luminosity_erg_s=light[indices],
            )

        for worker_count in (1, 3):
            with self.subTest(workers=worker_count):
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    pieces = list(executor.map(calculate, np.array_split(np.arange(5), 4)))
                merged = LineVelocityMoments()
                for piece in pieces:
                    merged = merged.merged(piece)
                self.assertEqual(merged.luminosity_erg_s, expected[0])
                np.testing.assert_allclose(merged.centroid_kms, expected[1], rtol=1.e-14)
                np.testing.assert_allclose(merged.sigma_kms, expected[2], rtol=1.e-14)


class FullIntegratedLineMomentTests(unittest.TestCase):
    def test_single_accumulator_combines_both_regimes_and_merges_worker_products(self):
        line_keys = ("halpha", "co10")
        edges = np.linspace(-200.0, 200.0, 401)
        velocity = np.array([-12.0, 500.0, 8.0, -260.0, 4.0, 35.0])
        volume = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])
        cold = np.array([True, False, True, False, True, False])
        temperature = np.array([
            [100.0, 10000.0, 80.0, 20000.0, 10.0, 5000.0],
            [25.0, 150.0, 80.0, 500.0, 10.0, 100.0],
        ])
        epsilon = np.array([[2.0, 4.0, 1.0, 3.0, 0.0, 2.0], [1.0, 2.0, 3.0, 1.0, 0.0, 5.0]])

        def accumulate(indices):
            result = IntegratedSpectra(line_keys, edges, cell_chunk=2)
            cells = SimpleNamespace(velocity_z_kms=velocity[indices], cell_volume_cm3=1.)
            emission = SimpleNamespace(
                cold_cells=cold[indices],
                lines={
                    key: LineEmission(
                        intrinsic_emissivity_erg_s_cm3=epsilon[row, indices] * volume[indices],
                        attenuated_emissivity_erg_s_cm3=epsilon[row, indices] * volume[indices],
                        temperature_K=temperature[row, indices],
                    )
                    for row, key in enumerate(line_keys)
                },
            )
            result.add_batch(cells=cells, emission=emission)
            return result

        serial = accumulate(np.arange(velocity.size))
        with ThreadPoolExecutor(max_workers=3) as executor:
            partials = list(executor.map(accumulate, np.array_split(np.arange(velocity.size), 3)))
        merged = IntegratedSpectra(line_keys, edges, cell_chunk=2)
        for partial in partials:
            merged.merge(other=partial)
        serial_payload, _ = serial.build_output(projected_area_cm2=10.)
        merged_payload, _ = merged.build_output(projected_area_cm2=10.)
        for row, key in enumerate(line_keys):
            mass_g = LINE_DEFINITIONS[key].emitter_mass_amu * ATOMIC_MASS_UNIT_G
            width = np.sqrt(BOLTZMANN_ERG_K * temperature[row] / mass_g) / 1.e5
            width *= 1.0 - velocity / SPEED_OF_LIGHT_KMS
            expected = direct_cell_moments(velocity, width, epsilon[row] * volume)
            for field, value in zip((
                "full_line_luminosity_erg_s", "line_centroid_full_kms",
                "line_sigma_full_kms", "line_second_raw_moment_full_kms2",
            ), expected):
                np.testing.assert_allclose(serial_payload[field][0, row], value, rtol=2.e-14)
                np.testing.assert_allclose(merged_payload[field][0, row], value, rtol=2.e-14)
            for label, selected in (("cold", cold), ("hot", ~cold)):
                branch_expected = direct_cell_moments(
                    velocity[selected], width[selected], (epsilon[row] * volume)[selected],
                )
                branch_actual = merged.full_line_moments["intrinsic"][key][label]
                np.testing.assert_allclose(
                    [branch_actual.luminosity_erg_s, branch_actual.centroid_kms, branch_actual.sigma_kms],
                    branch_expected[:3], rtol=2.e-14,
                )
        np.testing.assert_allclose(
            serial_payload["dL_dv_erg_s_per_kms"], merged_payload["dL_dv_erg_s_per_kms"],
            rtol=2.e-14, atol=1.e-250,
        )

    def test_each_line_and_dust_state_use_its_own_cells_and_complete_light(self):
        line_keys = ("halpha", "co10", "ciii_977")
        velocity = np.array([-25.0, 500.0, 35.0, 8.0])
        volumes = np.array([1.0, 2.0, 3.0, 4.0])
        cold = np.array([True, False, False, True])
        lines = {
            "halpha": LineEmission(
                np.array([np.nan, 5.0, 2.0, 1.0]),
                np.array([np.nan, 0.1, 1.0, 0.5]),
                np.array([np.nan, 10000.0, 5000.0, 80.0]),
            ),
            "co10": LineEmission(
                np.array([2.0, np.nan, 3.0, 1.0]),
                np.array([0.5, np.nan, 0.75, 0.25]),
                np.array([30.0, np.nan, 150.0, 80.0]),
            ),
            "ciii_977": LineEmission(
                np.array([0.0, 4.0, np.nan, 0.0]),
                np.array([0.0, 0.2, np.nan, 0.0]),
                np.array([100.0, 20000.0, np.nan, 100.0]),
            ),
        }
        expected = {}
        for key, line in lines.items():
            available = ~line.emissivity_is_missing
            mass_g = LINE_DEFINITIONS[key].emitter_mass_amu * ATOMIC_MASS_UNIT_G
            width = np.sqrt(BOLTZMANN_ERG_K * line.temperature_K[available] / mass_g) / 1.e5
            width *= 1.0 - velocity[available] / SPEED_OF_LIGHT_KMS
            expected[key] = []
            for epsilon in (line.intrinsic_emissivity_erg_s_cm3, line.attenuated_emissivity_erg_s_cm3):
                expected[key].append(direct_cell_moments(
                    velocity[available], width, epsilon[available] * volumes[available],
                ))

        # The production grid has one scalar volume. Express diagnostic
        # volume weights in epsilon while retaining the independent references.
        cells = SimpleNamespace(velocity_z_kms=velocity, cell_volume_cm3=1.)
        emission = SimpleNamespace(
            cold_cells=cold,
            lines={
                key: LineEmission(
                    intrinsic_emissivity_erg_s_cm3=line.intrinsic_emissivity_erg_s_cm3 * volumes,
                    attenuated_emissivity_erg_s_cm3=line.attenuated_emissivity_erg_s_cm3 * volumes,
                    temperature_K=line.temperature_K,
                )
                for key, line in lines.items()
            },
        )
        payloads = []
        for workers in (1, 2):
            spectra = IntegratedSpectra(line_keys, np.linspace(-200.0, 200.0, 401), workers=workers, cell_chunk=2)
            spectra.add_batch(cells=cells, emission=emission)
            payload, _ = spectra.build_output(projected_area_cm2=10.0)
            payloads.append(payload)
            for line_index, key in enumerate(line_keys):
                for dust_index in (0, 1):
                    reference = expected[key][dust_index]
                    for field, value in zip((
                        "full_line_luminosity_erg_s", "line_centroid_full_kms",
                        "line_sigma_full_kms", "line_second_raw_moment_full_kms2",
                    ), reference):
                        np.testing.assert_allclose(payload[field][dust_index, line_index], value, rtol=2.e-14)
            # CIII emits only at 500 km/s. Its full centroid is defined even
            # though the saved window contains almost no light from that cell.
            self.assertEqual(payload["line_centroid_full_kms"][0, 2], 500.0)
            self.assertGreater(payload["line_sigma_full_kms"][0, 0], payload["line_sigma_window_kms"][0, 0])
        for field in ("dL_dv_erg_s_per_kms", "full_line_luminosity_erg_s",
                      "line_centroid_full_kms", "line_sigma_full_kms"):
            np.testing.assert_array_equal(payloads[0][field], payloads[1][field])


if __name__ == "__main__":
    unittest.main()
