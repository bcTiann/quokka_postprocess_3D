"""Product-owned batch mapping, ordered merging, and conservation checks."""
from types import SimpleNamespace
import gc
import json
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from quokka2s.constants import ATOMIC_MASS_UNIT_G, BOLTZMANN_ERG_K, SPEED_OF_LIGHT_KMS
from quokka2s.products.integrated_spectra import (
    IntegratedSpectra,
    LINE_MASSES_AMU,
    accumulate_velocity_spectra,
    check_spectrum_luminosity,
)
from quokka2s.products.gas_phase_velocity import GasPhaseVelocityAccumulator, check_phase_accounting
from quokka2s.products.line_luminosity_images import LineLuminosityImageAccumulator, DUST_STATES, check_image_luminosity
from quokka2s.products.emission_products import EmissionProducts
from quokka2s.physics.cell_emission import LineEmission
from quokka2s.physics.line_emissivity import ATOMIC_LINE_KEYS, CO_LINE_KEYS, CIII_CIV_LINE_KEYS
from quokka2s.result_files import add_output_metadata


LINE_KEYS = ('hi21', 'cii')
VELOCITY_EDGES = np.linspace(-200., 200., 401)


def snapshot_geometry(shape, projected_area_cm2, xy_region=None):
    """Provide the production geometry attributes without retaining a dataset."""
    region = {'x': (0, shape[0]), 'y': (0, shape[1])} if xy_region is None else xy_region
    nx = region['x'][1] - region['x'][0]
    ny = region['y'][1] - region['y'][0]
    processing_shape = (nx, ny, shape[2])
    return SimpleNamespace(
        shape=shape,
        cell_count=int(np.prod(shape)),
        projected_area_cm2=projected_area_cm2,
        xy_region=region,
        processing_shape=processing_shape,
        processing_cell_count=int(np.prod(processing_shape)),
        processing_area_cm2=projected_area_cm2 * nx * ny / (shape[0] * shape[1]),
        processing_xy_origin=(region['x'][0], region['y'][0]),
    )


def batches():
    """Include excluded cells, both branches, zero emission, and window losses."""
    temperature = np.array([100., 1000., 1000., 3000., 1e4, 1e6, 200., 400.])
    despotic = np.array([100., 5000., np.nan, 40., 30., 60., 200., 400.])
    velocity = np.array([-250., -25., np.nan, 0., 40., 250., 10., -10.])
    density = np.arange(1., 9.)
    intrinsic = np.array([[1., 2., np.nan, 0., 4., 5., 1., 2.],
                          [2., 0., np.nan, 3., 2., 1., 4., 2.]])
    attenuation = np.array([[1.]*8, [1., .5, np.nan, .2, .4, 0., .8, .1]])
    attenuated = intrinsic * attenuation
    cold = temperature < 3000.
    mixed = np.where(cold, despotic, temperature)
    thermal = np.tile(mixed, (len(LINE_KEYS), 1))
    for start, stop in ((0, 3), (3, 5), (5, 8)):
        part = slice(start, stop)
        cells = SimpleNamespace(
            x_start=0, y_start=0, native_y_size=2,
            slab_shape=(2, 2, 2), batch_start=start,
            cell_volume_cm3=3., velocity_z_kms=velocity[part],
            temperature_QUOKKA_K=temperature[part], density_g_cm3=density[part],
        )
        emission = SimpleNamespace(
            cold_cells=cold[part],
            despotic_temperature_K=despotic[part],
            lines={
                key: LineEmission(
                    temperature_K=thermal[row, part],
                    intrinsic_emissivity_erg_s_cm3=intrinsic[row, part],
                    attenuated_emissivity_erg_s_cm3=attenuated[row, part],
                )
                for row, key in enumerate(LINE_KEYS)
            },
            despotic_coordinate_clipped_cells={'nH': 0, 'NH': 0, 'dVdr': 0},
            cloudy_column_clipped_cells={'below': 0, 'above': 0},
        )
        yield cells, emission


def dense_reference_fields(emission):
    """Pack fixture fields in declared order for independent luminosity references."""
    intrinsic = np.stack([
        emission.lines[key].intrinsic_emissivity_erg_s_cm3 for key in LINE_KEYS
    ])
    attenuated = np.stack([
        emission.lines[key].attenuated_emissivity_erg_s_cm3 for key in LINE_KEYS
    ])
    thermal = np.stack([
        emission.lines[key].temperature_K for key in LINE_KEYS
    ])
    return intrinsic, attenuated, thermal


def new_spectra():
    return IntegratedSpectra(LINE_KEYS, VELOCITY_EDGES,
                             workers=1, cell_chunk=2)


def expected_luminosity():
    totals = np.zeros((2, len(LINE_KEYS), 2))
    for cells, emission in batches():
        intrinsic, attenuated, _ = dense_reference_fields(emission)
        for dust_index, epsilon in enumerate((intrinsic, attenuated)):
            for branch, mask in enumerate((emission.cold_cells, ~emission.cold_cells)):
                totals[dust_index, :, branch] += (
                    np.nansum(epsilon[:, mask], axis=1) * cells.cell_volume_cm3)
    return totals


class ProductBatchTests(unittest.TestCase):
    def test_missing_hot_co_does_not_remove_atomic_images_spectra_or_gas(self):
        line_keys = ATOMIC_LINE_KEYS + CO_LINE_KEYS
        snapshot = snapshot_geometry(shape=(1, 1, 2), projected_area_cm2=12.)
        cells = SimpleNamespace(
            x_start=0,
            y_start=0,
            native_y_size=1,
            slab_shape=(1, 1, 2),
            batch_start=0,
            cell_volume_cm3=3.,
            density_g_cm3=np.array([1., 2.]),
            temperature_QUOKKA_K=np.array([5000., 1000.]),
            velocity_z_kms=np.array([-10., 10.]),
        )
        lines = {}
        for line_key in line_keys:
            if line_key in CO_LINE_KEYS:
                intrinsic = np.array([np.nan, 2.])
                thermal = np.array([np.nan, 50.])
            elif line_key in CIII_CIV_LINE_KEYS:
                intrinsic = np.array([7., 0.])
                thermal = np.array([5000., 1000.])
            else:
                intrinsic = np.array([7., 11.])
                thermal = np.array([5000., 50.])
            attenuated = intrinsic.copy() if line_key == 'hi21' else .5 * intrinsic
            lines[line_key] = LineEmission(intrinsic, attenuated, thermal)
        emission = SimpleNamespace(
            lines=lines,
            cold_cells=np.array([False, True]),
            despotic_temperature_K=np.array([np.nan, 50.]),
            despotic_coordinate_clipped_cells={'nH': 0, 'NH': 0, 'dVdr': 0},
            cloudy_column_clipped_cells={'below': 0, 'above': 0},
        )
        products = EmissionProducts(snapshot, line_keys, spectral_workers=1)
        products.images.add_batch(cells, emission)
        products.spectra.add_batch(cells, emission)
        products.phases.add_batch(cells, emission)
        products.record_batch(cells, emission)
        outputs = products.build_outputs()
        products.check_outputs(outputs)
        self.assertTrue(outputs.processing_complete)
        self.assertTrue(outputs.full_snapshot)

        halpha = line_keys.index('halpha')
        co10 = line_keys.index('co10')
        self.assertEqual(outputs.image_payload['total_luminosity_erg_s'][0, halpha], 54.)
        self.assertEqual(outputs.image_payload['total_luminosity_erg_s'][0, co10], 6.)
        np.testing.assert_array_equal(outputs.spectrum_payload['cell_counts_by_regime'][halpha], [1, 1])
        np.testing.assert_array_equal(outputs.spectrum_payload['cell_counts_by_regime'][co10], [1, 0])
        self.assertEqual(products.missing_emissivity_cells['halpha'], 0)
        self.assertEqual(products.missing_emissivity_cells['co10'], 1)
        self.assertEqual(products.counts['gas_temperature_missing'], 0)
        self.assertEqual(outputs.phase_payload['group_count'][-1], 2)
        self.assertEqual(outputs.phase_payload['group_mass_g'][-1], 9.)

    def test_selected_region_keeps_native_pixels_area_and_completion_metadata(self):
        snapshot = snapshot_geometry(
            shape=(5, 7, 3),
            projected_area_cm2=70.,
            xy_region={'x': (1, 4), 'y': (2, 6)},
        )
        products = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        initial = products.build_outputs()
        self.assertFalse(initial.processing_complete)
        self.assertFalse(initial.full_snapshot)
        native_values = np.arange(1., 106.).reshape(snapshot.shape)
        selected_values = native_values[1:4, 2:6].ravel()
        for start, stop in ((0, 5), (5, 19), (19, selected_values.size)):
            values = selected_values[start:stop]
            intrinsic = np.where(values % 7 == 0, np.nan, 2. * values)
            cells = SimpleNamespace(
                x_start=1,
                y_start=2,
                native_y_size=7,
                slab_shape=snapshot.processing_shape,
                batch_start=start,
                cell_volume_cm3=2.,
                density_g_cm3=np.ones(values.size),
                temperature_QUOKKA_K=np.full(values.size, 100.),
                velocity_z_kms=np.zeros(values.size),
            )
            emission = SimpleNamespace(
                cold_cells=np.ones(values.size, dtype=bool),
                despotic_temperature_K=np.full(values.size, 100.),
                lines={
                    'hi21': LineEmission(values, values, np.full(values.size, 100.)),
                    'cii': LineEmission(intrinsic, .25 * intrinsic, np.full(values.size, 100.)),
                },
                despotic_coordinate_clipped_cells={'nH': 0, 'NH': 0, 'dVdr': 0},
                cloudy_column_clipped_cells={'below': 0, 'above': 0},
            )
            products.images.add_batch(cells=cells, emission=emission)
            products.spectra.add_batch(cells=cells, emission=emission)
            products.phases.add_batch(cells=cells, emission=emission)
            products.record_batch(cells=cells, emission=emission)

        outputs = products.build_outputs()
        self.assertTrue(outputs.processing_complete)
        self.assertFalse(outputs.full_snapshot)
        self.assertEqual(products.expected_cell_count, 36)
        self.assertEqual(products.snapshot_cell_count, 105)
        self.assertEqual(float(outputs.spectrum_payload['projected_area_cm2']), 24.)
        selected_cube = native_values[1:4, 2:6]
        expected_hi = 2. * selected_cube.sum(axis=2)
        expected_cii = 4. * np.where(selected_cube % 7 == 0, 0., selected_cube).sum(axis=2)
        np.testing.assert_array_equal(
            outputs.image_payload['line_luminosity_image_erg_s'],
            np.asarray([[expected_hi, expected_cii], [expected_hi, .25 * expected_cii]]),
        )
        np.testing.assert_array_equal(outputs.image_payload['image_xy_origin'], [1, 2])

        left = np.array([-2., -3., -1.])
        right = np.array([3., 4., 1.])
        snapshot.dataset = SimpleNamespace(
            domain_left_edge=SimpleNamespace(to=lambda unit: SimpleNamespace(value=left)),
            domain_right_edge=SimpleNamespace(to=lambda unit: SimpleNamespace(value=right)),
        )
        calculator = SimpleNamespace(
            line_keys=LINE_KEYS,
            dust_cross_section_cm2_H={'hi21': 0., 'cii': 1e-21},
        )
        add_output_metadata(
            snapshot=snapshot,
            emission_calculator=calculator,
            outputs=outputs,
        )
        np.testing.assert_array_equal(
            outputs.image_payload['x_edges_kpc'], np.linspace(-2., 3., 6)[1:5],
        )
        np.testing.assert_array_equal(
            outputs.image_payload['y_edges_kpc'], np.linspace(-3., 4., 8)[2:7],
        )
        for payload in (outputs.image_payload, outputs.spectrum_payload, outputs.phase_payload):
            self.assertTrue(bool(payload['processing_complete']))
            self.assertFalse(bool(payload['full_snapshot']))
            region = json.loads(str(payload['processing_region']))
            self.assertEqual(region['xy_region'], {'x': [1, 4], 'y': [2, 6]})
            self.assertEqual(region['z_index_range'], [0, 3])
            self.assertEqual(region['shape'], [3, 4, 3])
            self.assertEqual(region['cell_count'], 36)
            self.assertEqual(region['projected_area_cm2'], 24.)

    def test_equal_size_images_from_different_regions_cannot_merge(self):
        images = LineLuminosityImageAccumulator(
            line_keys=LINE_KEYS,
            native_xy_shape=(3, 4),
            image_xy_origin=(1, 2),
        )
        other = LineLuminosityImageAccumulator(
            line_keys=LINE_KEYS,
            native_xy_shape=(3, 4),
            image_xy_origin=(2, 2),
        )
        with self.assertRaisesRegex(ValueError, 'different native x-y regions'):
            images.merge(other)

    def test_product_geometry_survives_without_retaining_snapshot(self):
        class SnapshotGeometry:
            shape = (2, 3, 4)
            cell_count = 24
            projected_area_cm2 = 12.
            processing_shape = shape
            processing_cell_count = cell_count
            processing_area_cm2 = projected_area_cm2
            processing_xy_origin = (0, 0)

        snapshot = SnapshotGeometry()
        reference = weakref.ref(snapshot)
        products = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        del snapshot
        self.assertIsNone(reference())
        self.assertEqual(products.expected_cell_count, 24)
        self.assertEqual(products.projected_area_cm2, 12.)
        self.assertEqual(products.line_keys, LINE_KEYS)
        self.assertEqual(products.images.native_images.shape, (2, 2, 2, 3))

    def test_outputs_are_built_before_separate_accounting_checks(self):
        snapshot = snapshot_geometry(shape=(2, 3, 4), projected_area_cm2=12.)
        products = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        products.mass_g['all'] = 1.  # Deliberately disagrees with cold + hot.
        outputs = products.build_outputs()
        self.assertEqual(outputs.image_payload['line_luminosity_image_erg_s'].shape,
                         (2, 2, 2, 3))
        with self.assertRaisesRegex(ValueError, 'Temperature-regime masses'):
            products.check_outputs(outputs)

    def test_final_process_check_rejects_nonfinite_or_negative_light(self):
        snapshot = snapshot_geometry(shape=(1, 1, 1), projected_area_cm2=1.)
        corrupt_fields = (
            ("image pixels", "image_payload", "line_luminosity_image_erg_s", np.nan),
            ("image pixels", "image_payload", "line_luminosity_image_erg_s", -1.),
            ("spectral channels", "spectrum_payload", "dL_dv_erg_s_per_kms", np.inf),
            ("spectrum outside luminosities", "spectrum_payload", "outside_velocity_luminosity_erg_s", -1.),
        )
        for description, product_name, field_name, invalid_value in corrupt_fields:
            with self.subTest(field=field_name, value=invalid_value):
                products = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
                outputs = products.build_outputs()
                payload = getattr(outputs, product_name)
                payload[field_name].flat[0] = invalid_value
                with self.assertRaisesRegex(ValueError, f"{description} must be finite and nonnegative"):
                    products.check_outputs(outputs)

    def assert_payload_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

    def test_image_batch_maps_native_pixels_and_merges_in_batch_order(self):
        serial = LineLuminosityImageAccumulator(LINE_KEYS, (2, 2))
        merged = LineLuminosityImageAccumulator(LINE_KEYS, (2, 2))
        for cells, emission in batches():
            serial.add_batch(cells, emission)
            part = LineLuminosityImageAccumulator(LINE_KEYS, (2, 2))
            part.add_batch(cells, emission)
            merged.merge(part)
        expected = np.array([
            [[[9., 0.], [27., 9.]], [[6., 9.], [9., 18.]]],
            [[[9., 0.], [27., 9.]], [[6., 1.8], [2.4, 10.2]]],
        ])
        payload = serial.build_output()
        np.testing.assert_allclose(payload['line_luminosity_image_erg_s'], expected,
                                   rtol=1e-14, atol=0.)
        self.assert_payload_equal(merged.build_output(), payload)
        report = check_image_luminosity(
            payload['total_luminosity_erg_s'], expected_luminosity().sum(axis=-1), LINE_KEYS)
        self.assertLessEqual(report['maximum_relative_difference'], report['relative_tolerance'])

    def test_spectral_batch_matches_direct_kernel_and_ordered_merge(self):
        serial = new_spectra()
        merged = new_spectra()
        reference_profiles = np.zeros_like(serial.dL_dv)
        for cells, emission in batches():
            serial.add_batch(cells=cells, emission=emission)
            partial = new_spectra()
            partial.add_batch(cells=cells, emission=emission)
            merged.merge(other=partial)
            for line_index, key in enumerate(LINE_KEYS):
                line = emission.lines[key]
                available = ~line.emissivity_is_missing
                for branch_index, branch_cells in enumerate((emission.cold_cells, ~emission.cold_cells)):
                    selected = available & branch_cells
                    velocity = cells.velocity_z_kms[selected]
                    width = np.sqrt(
                        BOLTZMANN_ERG_K * line.temperature_K[selected]
                        / (LINE_MASSES_AMU[key] * ATOMIC_MASS_UNIT_G)
                    ) / 1.e5
                    width *= 1. - velocity / SPEED_OF_LIGHT_KMS
                    luminosity = np.column_stack((
                        line.intrinsic_emissivity_erg_s_cm3[selected] * cells.cell_volume_cm3,
                        line.attenuated_emissivity_erg_s_cm3[selected] * cells.cell_volume_cm3,
                    ))
                    profiles = accumulate_velocity_spectra(
                        velocity_kms=velocity,
                        thermal_width_kms=width,
                        luminosity_matrix=luminosity,
                        velocity_edges_kms=VELOCITY_EDGES,
                        cell_chunk=2,
                        workers=1,
                    )
                    reference_profiles[:, line_index, branch_index] += profiles.T
        payload, report = serial.build_output(projected_area_cm2=12.)
        merged_payload, merged_report = merged.build_output(projected_area_cm2=12.)
        check_spectrum_luminosity(payload, expected_luminosity())
        check_spectrum_luminosity(merged_payload, expected_luminosity())
        np.testing.assert_array_equal(payload['dL_dv_erg_s_per_kms'], reference_profiles)
        self.assert_payload_equal(merged_payload, payload)
        self.assertEqual(merged_report, report)
        np.testing.assert_array_equal(payload['cell_counts_by_regime'], [[4, 3], [4, 3]])
        self.assertTrue(np.any(payload['outside_velocity_luminosity_erg_s'] > 0.))
        np.testing.assert_array_equal(payload['dL_dv_erg_s_per_kms'][0, 0],
                                      payload['dL_dv_erg_s_per_kms'][1, 0])

    def test_line_dictionary_order_does_not_change_images_spectra_or_totals(self):
        snapshot = snapshot_geometry(shape=(2, 2, 2), projected_area_cm2=12.)
        declared = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        reversed_dictionary = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        for cells, emission in batches():
            reordered = SimpleNamespace(**vars(emission))
            reordered.lines = dict(reversed(tuple(emission.lines.items())))
            for products, result in ((declared, emission), (reversed_dictionary, reordered)):
                products.images.add_batch(cells, result)
                products.spectra.add_batch(cells, result)
                products.record_batch(cells, result)
        self.assert_payload_equal(
            reversed_dictionary.images.build_output(), declared.images.build_output(),
        )
        expected_spectrum, expected_report = declared.spectra.build_output(12.)
        actual_spectrum, actual_report = reversed_dictionary.spectra.build_output(12.)
        self.assert_payload_equal(actual_spectrum, expected_spectrum)
        self.assertEqual(actual_report, expected_report)
        for field in ('intrinsic_luminosity_erg_s', 'attenuated_luminosity_erg_s'):
            np.testing.assert_array_equal(
                getattr(reversed_dictionary, field), getattr(declared, field),
            )

    def test_spectral_boundary_packs_only_retained_cells_in_declared_line_order(self):
        _, emission = next(batches())
        emission.lines = dict(reversed(tuple(emission.lines.items())))
        spectra = new_spectra()
        available_cells = ~emission.lines[LINE_KEYS[0]].emissivity_is_missing
        packed = spectra.pack_available_line_fields(
            emission=emission,
            line_keys=LINE_KEYS,
            available_cells=available_cells,
        )
        reference = dense_reference_fields(emission)
        for actual, expected in zip(packed, reference):
            self.assertEqual(actual.shape, (2, 2))  # Two retained cells, not all three.
            self.assertTrue(actual.flags.f_contiguous)
            np.testing.assert_array_equal(actual, expected[:, available_cells])

    def test_product_accumulators_release_named_batch_and_spectral_scratch_arrays(self):
        cells, source_emission = next(batches())
        # Own every array here; generator-backed views would keep their base alive.
        emission = SimpleNamespace(
            cold_cells=source_emission.cold_cells.copy(),
            despotic_temperature_K=source_emission.despotic_temperature_K.copy(),
            lines={
                key: LineEmission(**{
                    name: values.copy() for name, values in vars(line).items()
                })
                for key, line in source_emission.lines.items()
            },
            despotic_coordinate_clipped_cells={'nH': 0, 'NH': 0, 'dVdr': 0},
            cloudy_column_clipped_cells={'below': 0, 'above': 0},
        )
        references = [weakref.ref(values) for line in emission.lines.values()
                      for values in vars(line).values()]
        references.extend(weakref.ref(values) for values in (
            emission.cold_cells, emission.despotic_temperature_K,
        ))
        spectra = new_spectra()
        scratch_references = []
        pack = spectra.pack_available_line_fields

        def remember_scratch(**arguments):
            arrays = pack(**arguments)
            scratch_references.extend(weakref.ref(array) for array in arrays)
            return arrays

        images = LineLuminosityImageAccumulator(LINE_KEYS, (2, 2))
        phases = GasPhaseVelocityAccumulator(VELOCITY_EDGES)
        snapshot = snapshot_geometry(shape=(2, 2, 2), projected_area_cm2=12.)
        accounting = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        with patch.object(spectra, 'pack_available_line_fields', side_effect=remember_scratch):
            images.add_batch(cells, emission)
            spectra.add_batch(cells, emission)
            phases.add_batch(cells, emission)
            accounting.record_batch(cells, emission)
        del emission
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertTrue(all(reference() is None for reference in scratch_references))

    def test_named_accounting_matches_dense_reference_with_roundoff_tolerance(self):
        rng = np.random.default_rng(44)
        intrinsic = 10. ** rng.uniform(-30., 30., (2, 10000))
        attenuated = .3 * intrinsic
        selected = rng.random(10000) > .3
        emission = SimpleNamespace(lines={
            key: LineEmission(
                intrinsic_emissivity_erg_s_cm3=intrinsic[row],
                attenuated_emissivity_erg_s_cm3=attenuated[row],
                temperature_K=np.ones(intrinsic.shape[1]),
            )
            for row, key in enumerate(LINE_KEYS)
        })
        snapshot = snapshot_geometry(shape=(2, 2, 2), projected_area_cm2=12.)
        products = EmissionProducts(snapshot, LINE_KEYS, spectral_workers=1)
        actual_intrinsic, actual_attenuated = products.sum_named_line_luminosities(
            emission=emission,
            selected_cells=selected,
            cell_volume_cm3=3.,
        )
        np.testing.assert_allclose(actual_intrinsic, intrinsic[:, selected].sum(axis=1) * 3., rtol=1e-12)
        np.testing.assert_allclose(actual_attenuated, attenuated[:, selected].sum(axis=1) * 3., rtol=1e-12)

    def test_phase_batch_uses_mixed_temperature_and_exact_retained_cells(self):
        serial = GasPhaseVelocityAccumulator(VELOCITY_EDGES)
        merged = GasPhaseVelocityAccumulator(VELOCITY_EDGES)
        for cells, emission in batches():
            serial.add_batch(cells, emission)
            merged.merge(GasPhaseVelocityAccumulator(VELOCITY_EDGES).add_batch(cells, emission))
        payload, report = serial.build_output()
        merged_payload, merged_report = merged.build_output()
        check_phase_accounting(report['groups'], payload['histogram_mass_g'], 7, 99.)
        check_phase_accounting(merged_report['groups'], merged_payload['histogram_mass_g'], 7, 99.)
        self.assert_payload_equal(merged_payload, payload)
        self.assertEqual(merged_report, report)
        # The cold QUOKKA=1000 K cell uses DESPOTIC=5000 K and belongs to WNM;
        # the hot QUOKKA=1e6 K cell remains HIM despite DESPOTIC=60 K.
        np.testing.assert_array_equal(payload['group_count'], [1, 2, 2, 1, 1, 7])
        np.testing.assert_array_equal(payload['group_mass_g'], [3., 45., 18., 15., 18., 99.])
        self.assertEqual(report['groups']['total']['below_window']['count'], 1)
        self.assertEqual(report['groups']['total']['above_window']['count'], 1)

    def test_product_checks_reject_inconsistent_totals_after_building_outputs(self):
        spectra = new_spectra()
        phases = GasPhaseVelocityAccumulator(VELOCITY_EDGES)
        images = LineLuminosityImageAccumulator(LINE_KEYS, (2, 2))
        for cells, emission in batches():
            spectra.add_batch(cells, emission)
            phases.add_batch(cells, emission)
            images.add_batch(cells, emission)
        spectrum_payload, _ = spectra.build_output(12.)
        phase_payload, phase_report = phases.build_output()
        with self.assertRaisesRegex(ValueError, 'Spectrum input differs'):
            check_spectrum_luminosity(spectrum_payload, expected_luminosity()*1.01)
        with self.assertRaisesRegex(ValueError, 'Phase cell counts'):
            check_phase_accounting(phase_report['groups'], phase_payload['histogram_mass_g'],
                                   gas_cell_count=8, gas_mass_g=99.)
        with self.assertRaisesRegex(ValueError, 'Phase masses differ'):
            check_phase_accounting(phase_report['groups'], phase_payload['histogram_mass_g'],
                                   gas_cell_count=7, gas_mass_g=100.)
        with self.assertRaisesRegex(ValueError, 'Image pixels do not sum'):
            check_image_luminosity(images.build_output()['total_luminosity_erg_s'],
                                   expected_luminosity().sum(axis=-1)*1.01, LINE_KEYS)


if __name__ == '__main__':
    unittest.main()
