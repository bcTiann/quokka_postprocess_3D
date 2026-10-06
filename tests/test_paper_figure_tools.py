"""The retained manuscript renderers use prepared arrays without old tasks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from quokka2s.figures.table_input_slices import (
    TABLE_INPUT_PANEL_KEYS,
    plot_table_input_slice,
    prepare_slice_panels,
)
from quokka2s.snapshot_reader import SlabArrays


def load_figure_script(filename):
    """Import the script's batch helper without running its command-line main."""
    path = Path(__file__).resolve().parents[1] / 'tools/figures' / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperFigureToolsTests(unittest.TestCase):
    def test_slice_and_projection_helpers_reuse_one_calculator_for_all_batches(self):
        slab = SlabArrays(
            density_g_cm3=np.array([1., 2., 3., 4.]),
            temperature_QUOKKA_K=np.array([100., 10000., 200., 20000.]),
            shielding_NH_cm2=np.ones(4),
            velocity_gradient_s=np.ones(4),
            velocity_z_kms=np.zeros(4),
            foreground_NH_cm2=np.zeros(4),
            x_start=0,
            shape=(1, 2, 2),
            cell_volume_cm3=2.,
        )
        snapshot = SimpleNamespace(
            shape=slab.shape,
            cell_volume_cm3=slab.cell_volume_cm3,
            read_slab=Mock(return_value=slab),
            dataset=SimpleNamespace(
                domain_left_edge=SimpleNamespace(
                    to=lambda unit: SimpleNamespace(value=np.array([0., 0., -1.])),
                ),
                domain_right_edge=SimpleNamespace(
                    to=lambda unit: SimpleNamespace(value=np.array([1., 1., 1.])),
                ),
            ),
        )
        retained_cells = np.ones(4, dtype=bool)
        despotic_temperature = .5 * slab.temperature_QUOKKA_K
        despotic_temperature[1] = np.nan  # Hot gas remains available via T_QUOKKA.

        def calculate(*, cells):
            start = cells.batch_start
            stop = start + cells.cell_count
            return SimpleNamespace(
                despotic_temperature_K=despotic_temperature[start:stop],
                cold_cells=cells.temperature_QUOKKA_K < 3000.0,
            )

        emission_calculator = SimpleNamespace(calculate=Mock(side_effect=calculate))
        slices = load_figure_script('build_table_input_slice.py')
        payload = slices.calculate_slice_fields(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            slice_index=0,
            query_chunk=2,
        )
        self.assertEqual(emission_calculator.calculate.call_count, 2)
        np.testing.assert_array_equal(
            payload['T_dsp_slice'],
            despotic_temperature.reshape((2, 2)),
        )
        np.testing.assert_array_equal(payload['valid'], retained_cells.reshape((2, 2)))

        emission_calculator.calculate.reset_mock()
        projections = load_figure_script('build_gas_projection_maps.py')
        accumulator = SimpleNamespace(add=Mock())
        counts, masses = projections.accumulate_projection_slab(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            accumulator=accumulator,
            x_start=0,
            x_stop=1,
            query_chunk=2,
        )
        self.assertEqual(emission_calculator.calculate.call_count, 2)
        np.testing.assert_array_equal(
            accumulator.add.call_args.kwargs['mixed_K'],
            np.array([50., 10000., 100., 20000.]).reshape(slab.shape),
        )
        np.testing.assert_array_equal(
            accumulator.add.call_args.kwargs['valid'], retained_cells.reshape(slab.shape),
        )
        self.assertEqual(counts['retained'], 4)
        self.assertEqual(masses['retained'], 20.)

    def test_phase_batch_uses_the_calculator_result_without_an_extra_query(self):
        script = load_figure_script('build_emission_phase_histograms.py')
        cells = SimpleNamespace(
            density_g_cm3=np.array([1., 2.]),
            temperature_QUOKKA_K=np.array([100., 10000.]),
            shielding_NH_cm2=np.ones(2),
            cell_volume_cm3=2.,
            cell_count=2,
        )
        emission = SimpleNamespace(
            despotic_temperature_K=np.array([50., np.nan]),
            cold_cells=np.array([True, False]),
        )
        emission_calculator = SimpleNamespace(calculate=Mock(return_value=emission))
        histograms = {}
        with patch.object(script, 'accumulate_emission_phase_histograms') as accumulate:
            totals = script.accumulate_phase_batch(
                histograms=histograms,
                cells=cells,
                emission_calculator=emission_calculator,
            )
        emission_calculator.calculate.assert_called_once_with(cells=cells)
        self.assertIs(accumulate.call_args.kwargs['emission'], emission)
        self.assertEqual(totals['processed_cells'], 2)
        self.assertEqual(totals['missing_despotic_temperature_cells'], 1)
        self.assertEqual(totals['missing_mixed_temperature_cells'], 0)
        self.assertEqual(totals['total_mass_g'], 6.)

    def test_slice_orientation_mask_and_original_arrays_are_preserved(self):
        values = np.array([[1.0, 10.0, 0.0], [100.0, np.nan, -1.0]])
        slices = {key: values.copy() for key in TABLE_INPUT_PANEL_KEYS}
        panels = prepare_slice_panels(slices)
        # A y-z array becomes a plotted z-y image; invalid cells stay blank.
        expected = np.array([[0.0, 2.0], [1.0, np.nan], [np.nan, np.nan]])
        for key in TABLE_INPUT_PANEL_KEYS:
            np.testing.assert_array_equal(panels[key]['log_data'], expected)
            np.testing.assert_array_equal(slices[key], values)

    def test_array_only_renderer_saves_png_and_pdf(self):
        values = np.geomspace(1.0, 100.0, 12).reshape(3, 4)
        slices = {key: values.copy() for key in TABLE_INPUT_PANEL_KEYS}
        with tempfile.TemporaryDirectory() as directory:
            png = Path(directory) / 'slice.png'
            plot_table_input_slice(
                slices=slices,
                extent_kpc=(0.0, 1.0, -4.0, 4.0),
                output_path=png,
                slice_index=216,
                dataset_name='plt0655228',
                show_title=True,
                save_pdf=True,
            )
            self.assertTrue(png.read_bytes().startswith(b'\x89PNG'))
            self.assertTrue(png.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

    def test_paper_figure_imports_do_not_load_snapshot_or_task_framework(self):
        code = '''
import json
import sys
import quokka2s.figures.table_input_slices
import quokka2s.figures.emission_phase_histograms
blocked = [name for name in sys.modules
           if name == 'yt' or name.startswith('yt.')
           or name == 'quokka2s.pipeline' or name.startswith('quokka2s.pipeline.')]
print(json.dumps(blocked))
'''
        result = subprocess.run(
            [sys.executable, '-c', code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == '__main__':
    unittest.main()
