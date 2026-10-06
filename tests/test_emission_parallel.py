"""Parallel batches retain ordering, bounded scheduling, and all accumulated products."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from quokka2s.process_snapshot import accumulate_batch, process_parallel_batches
from quokka2s.products.emission_products import EmissionProducts
from quokka2s.physics.cell_emission import LineEmission
from quokka2s.snapshot_reader import SlabArrays


class FakeSlab:
    """Supply batch locations without invoking snapshot I/O in scheduler tests."""

    def __init__(self, cell_count, first_cell_id=0):
        self.cell_count = cell_count
        self.first_cell_id = first_cell_id

    def batch(self, start, stop):
        offset = self.first_cell_id
        return SimpleNamespace(
            batch_start=start, cell_count=stop-start,
            first_cell_id=offset+start, last_cell_id=offset+stop-1,
        )

    def cell_id_at(self, slab_index):
        return self.first_cell_id + slab_index


class ParallelBatchTests(unittest.TestCase):
    def test_calculator_runs_once_before_the_four_product_updates(self):
        cells = SimpleNamespace(first_cell_id=12, last_cell_id=15)
        emission = object()
        emission_calculator = SimpleNamespace(calculate=Mock(return_value=emission))
        updates = Mock()
        products = SimpleNamespace(
            failure_context={},
            images=SimpleNamespace(add_batch=updates.images),
            spectra=SimpleNamespace(add_batch=updates.spectra),
            phases=SimpleNamespace(add_batch=updates.phases),
            record_batch=updates.accounting,
        )
        updates.attach_mock(emission_calculator.calculate, 'calculate')

        accumulate_batch(
            cells=cells,
            emission_calculator=emission_calculator,
            products=products,
        )

        self.assertEqual(
            [entry[0] for entry in updates.mock_calls],
            ['calculate', 'images', 'spectra', 'phases', 'accounting'],
        )
        emission_calculator.calculate.assert_called_once_with(cells=cells)
        for update in (updates.images, updates.spectra, updates.phases, updates.accounting):
            update.assert_called_once_with(cells=cells, emission=emission)
        self.assertEqual(products.failure_context, {'first_cell_id': 12, 'last_cell_id': 15})

    def test_batches_are_private_bounded_and_merged_in_order(self):
        args = SimpleNamespace(query_chunk=2, chunk_workers=2, spectral_workers=3)
        snapshot = SimpleNamespace(shape=(8, 1, 1))
        slab = FakeSlab(8)
        second_finished = Event()
        lock = Lock()
        merged, identities = [], []
        active = peak = 0
        shared_calculator = SimpleNamespace(line_keys=())

        def process(cells, emission_calculator, products):
            nonlocal active, peak
            self.assertIs(emission_calculator, shared_calculator)
            batch = products
            start = cells.batch_start
            with lock:
                active += 1
                peak = max(peak, active)
                identities.append(batch)
            batch.start = start
            if start == 0:
                self.assertTrue(second_finished.wait(2.))
            elif start == 2:
                second_finished.set()
            else:
                # Later work is submitted only after a completed result is merged.
                self.assertTrue(merged)
            with lock:
                active -= 1

        def merge(batch):
            merged.append(batch.start)

        products = SimpleNamespace(failure_context={}, merge=merge)
        with patch('quokka2s.process_snapshot.EmissionProducts',
                   side_effect=lambda **kwargs: SimpleNamespace()), \
                patch('quokka2s.process_snapshot.accumulate_batch', side_effect=process), \
                ThreadPoolExecutor(max_workers=2) as executor:
            process_parallel_batches(
                config=args,
                snapshot=snapshot,
                emission_calculator=shared_calculator,
                products=products,
                slab=slab,
                executor=executor,
            )
        self.assertEqual(merged, [0, 2, 4, 6])
        self.assertEqual(len({id(batch) for batch in identities}), 4)
        self.assertEqual(peak, 2)

    def test_failed_batch_preserves_cell_identifiers(self):
        args = SimpleNamespace(query_chunk=2, chunk_workers=2, spectral_workers=1)
        snapshot = SimpleNamespace(shape=(4, 2, 3))
        slab = FakeSlab(4, first_cell_id=12)
        products = SimpleNamespace(failure_context={}, merge=lambda batch: None)

        def fail(*a, **kwargs):
            raise ValueError('unavailable test batch')

        with patch('quokka2s.process_snapshot.EmissionProducts', return_value=object()), \
                patch('quokka2s.process_snapshot.accumulate_batch', side_effect=fail), \
                ThreadPoolExecutor(max_workers=2) as executor:
            with self.assertRaisesRegex(ValueError, 'unavailable test batch'):
                process_parallel_batches(
                    config=args,
                    snapshot=snapshot,
                    emission_calculator=SimpleNamespace(line_keys=()),
                    products=products,
                    slab=slab,
                    executor=executor,
                )
        self.assertEqual(products.failure_context, {'first_cell_id': 12, 'last_cell_id': 13})

    def test_private_products_merge_like_one_continuous_accumulator(self):
        snapshot = SimpleNamespace(shape=(2, 2, 2), cell_volume_cm3=2.,
                                   cell_count=8, projected_area_cm2=4.,
                                   processing_shape=(2, 2, 2),
                                   processing_cell_count=8,
                                   processing_area_cm2=4.,
                                   processing_xy_origin=(0, 0))
        emission_calculator = SimpleNamespace(
            line_keys=('halpha', 'co10'),
            calculate=Mock(),
        )
        merged = EmissionProducts(snapshot, emission_calculator.line_keys, 1)
        direct = EmissionProducts(snapshot, emission_calculator.line_keys, 1)
        velocity = np.array([-10., -4., 0., 3., 15., 500., -350., 1.])
        gas_temperature = np.array([50., 400., 4000., 2e4, 1e6, 1e6, 50., 400.])
        density = np.arange(1., 9.)
        epsilon = np.stack((density, density**2))
        thermal = np.broadcast_to(gas_temperature, epsilon.shape)
        slab = SlabArrays(
            density_g_cm3=density, foreground_NH_cm2=np.ones(8),
            temperature_QUOKKA_K=gas_temperature, shielding_NH_cm2=np.ones(8),
            velocity_gradient_s=np.ones(8), velocity_z_kms=velocity,
            x_start=0, shape=(2, 2, 2), cell_volume_cm3=2.,
        )

        def add(products, start, stop):
            s = slice(start, stop)
            cold = gas_temperature[s] < 3000.
            valid = np.ones(stop-start, dtype=bool)
            emission = SimpleNamespace(
                cold_cells=cold,
                lines={
                    key: LineEmission(
                        intrinsic_emissivity_erg_s_cm3=epsilon[row, s],
                        attenuated_emissivity_erg_s_cm3=.5 * epsilon[row, s],
                        temperature_K=thermal[row, s],
                    )
                    for row, key in enumerate(emission_calculator.line_keys)
                },
                despotic_temperature_K=gas_temperature[s],
                despotic_coordinate_clipped_cells={'nH': int(valid.sum()), 'NH': 0, 'dVdr': 0},
                cloudy_column_clipped_cells={'below': 0, 'above': stop-start},
            )
            cells = slab.batch(start, stop)
            with patch.object(emission_calculator, 'calculate', return_value=emission) as calculate:
                accumulate_batch(
                    cells=cells,
                    emission_calculator=emission_calculator,
                    products=products,
                )
            calculate.assert_called_once_with(cells=cells)

        add(direct, 0, 8)
        for start, stop in ((0, 3), (3, 6), (6, 8)):
            batch = EmissionProducts(snapshot, emission_calculator.line_keys, 1)
            add(batch, start, stop)
            merged.merge(batch)
        np.testing.assert_allclose(merged.images.native_images, direct.images.native_images)
        for field in ('dL_dv', 'input_luminosity', 'cell_counts'):
            np.testing.assert_allclose(
                getattr(merged.spectra, field),
                getattr(direct.spectra, field),
                rtol=1e-12,
                atol=0,
            )
        for field in ('intrinsic_luminosity_erg_s', 'attenuated_luminosity_erg_s'):
            np.testing.assert_allclose(getattr(merged, field), getattr(direct, field))
        for field in ('counts', 'mass_g', 'despotic_coordinate_clipped_cells', 'cloudy_column_clipped_cells'):
            self.assertEqual(getattr(merged, field), getattr(direct, field))
        a, b = merged.phases.report(), direct.phases.report()
        np.testing.assert_allclose(a['global_mean_velocity_kms'], b['global_mean_velocity_kms'])
        for group in a['groups']:
            for field in ('mass_g', 'mean_velocity_kms', 'sigma_internal_kms',
                          'sigma_about_global_mean_kms', 'histogram_mass_g'):
                np.testing.assert_allclose(a['groups'][group][field], b['groups'][group][field],
                                           rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(merged.phases.histogram_mass_g[group],
                                          direct.phases.histogram_mass_g[group])


if __name__ == '__main__':
    unittest.main()
