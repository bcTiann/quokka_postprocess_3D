"""Current processing and figure helpers must not load the old task registry."""
from pathlib import Path
import os
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ImportBoundaryTests(unittest.TestCase):
    def assert_import_boundary(self, imports, forbidden_prefixes):
        # Run in a fresh interpreter so earlier imports cannot hide dependencies.
        script = "\n".join(imports) + f"""
import sys
forbidden = {forbidden_prefixes!r}
unexpected = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.')
           for prefix in forbidden)
)
assert not unexpected, 'Unrelated historical modules loaded: ' + repr(unexpected)
"""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            filter(None, (str(ROOT / 'src'), env.get('PYTHONPATH')))
        )
        env.setdefault('MPLBACKEND', 'Agg')
        result = subprocess.run(
            [sys.executable, '-c', script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_current_processing_and_shared_physics_skip_old_framework(self):
        self.assert_import_boundary([
            'import quokka2s.process_snapshot',
            'import quokka2s.figures.emission_results',
            'from quokka2s.physics import gas_fields',
        ], (
            'quokka2s.pipeline.base',
            'quokka2s.pipeline.cache',
            'quokka2s.pipeline.intermediate_io',
            'quokka2s.pipeline.tasks',
            'quokka2s.data_handling',
        ))

    def test_process_entrypoint_skips_plotting(self):
        self.assert_import_boundary([
            'import quokka2s.process_snapshot',
        ], (
            'quokka2s.plot_emission_results',
            'quokka2s.figures',
            'matplotlib',
        ))

    def test_plot_entrypoint_skips_snapshot_processing(self):
        self.assert_import_boundary([
            'import quokka2s.plot_emission_results',
        ], (
            'quokka2s.process_snapshot',
            'quokka2s.processing_inputs',
            'quokka2s.snapshot_reader',
            'yt',
            'despotic',
        ))

    def test_phase_histogram_helper_skips_other_tasks_and_field_caches(self):
        self.assert_import_boundary([
            'import quokka2s.products.emission_phase_histograms',
        ], (
            'quokka2s.pipeline.base',
            'quokka2s.pipeline.cache',
            'quokka2s.pipeline.tasks.run_pipeline',
            'quokka2s.pipeline.tasks.species_spectrum',
            'quokka2s.pipeline.tasks.multi_field_slices',
            'quokka2s.data_handling',
        ))

    def test_figure_one_renderer_skips_unrelated_tasks(self):
        self.assert_import_boundary([
            'from quokka2s.figures.table_input_slices import plot_table_input_slice',
        ], (
            'quokka2s.pipeline.tasks.run_pipeline',
            'quokka2s.pipeline.tasks.species_spectrum',
            'quokka2s.pipeline.tasks.velocity_phase',
            'quokka2s.pipeline.tasks.halpha_cloudy_comparison',
        ))


if __name__ == '__main__':
    unittest.main()
