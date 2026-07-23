from __future__ import annotations

import unittest

import numpy as np

from quokka2s.pipeline.tasks.phase_combined_plot import _LAYOUT, _Y_SHARE_GROUPS
from quokka2s.pipeline.tasks.phase_hist import _aligned_edges
from quokka2s.pipeline.tasks.hi_comparison import (
    _HI_PHASE_CONFIG,
    _intrinsic_surface_spectrum,
)
from quokka2s.pipeline.tasks.run_pipeline import build_pipeline


class PhaseTemperaturePolicyTests(unittest.TestCase):
    def test_species_panels_use_species_specific_temperature_tags(self):
        layout_tags = {tag for tag, _, _ in _LAYOUT}

        self.assertIn('CO_T_DSP', layout_tags)
        self.assertIn('Cplus_T_QK', layout_tags)
        self.assertIn('HI_T_2R', layout_tags)
        self.assertNotIn('CO_T_2R', layout_tags)
        self.assertNotIn('Cplus_T_2R', layout_tags)
        self.assertNotIn('HI_DSP_T_DSP', layout_tags)
        self.assertNotIn('HI_QK_T_QK', layout_tags)
        self.assertEqual(len(_LAYOUT), 8)

    def test_species_panels_share_y_with_matching_mass_temperature(self):
        share_groups = [set(group) for group in _Y_SHARE_GROUPS]

        self.assertIn(
            {'mass_T_DSP', 'CO_T_DSP'},
            share_groups,
        )
        self.assertIn(
            {'mass_T_QK', 'Cplus_T_QK'},
            share_groups,
        )
        self.assertIn(
            {'mass_T_2R', 'Halpha_T_2R', 'HI_T_2R'},
            share_groups,
        )

    def test_point_two_dex_edges_align_with_integer_decades(self):
        edges = _aligned_edges(-29.17, -20.03, 0.2)

        np.testing.assert_allclose(np.diff(edges), 0.2, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(edges / 0.2, np.round(edges / 0.2))
        for decade in range(-29, -20):
            index = np.flatnonzero(np.isclose(edges, float(decade), atol=1e-14))
            self.assertEqual(index.size, 1)
            next_index = np.flatnonzero(
                np.isclose(edges, float(decade + 1), atol=1e-14)
            )
            self.assertEqual(int(next_index[0] - index[0]), 5)

    def test_dedicated_hi_comparison_keeps_both_all_temperature_models(self):
        comparison_tags = {tag for tag, _, _ in _HI_PHASE_CONFIG}
        self.assertEqual(
            comparison_tags,
            {'HI_DSP_T_DSP', 'HI_QK_T_QK'},
        )

        pipeline = build_pipeline()
        task_classes = [task.__class__.__name__ for task in pipeline._tasks]
        phase_tags = {
            task.tag
            for task in pipeline._tasks
            if task.__class__.__name__ == 'Build_PhaseHist'
        }
        self.assertIn('Plot_HIComparison', task_classes)
        self.assertIn('HI_T_2R', phase_tags)
        self.assertTrue(comparison_tags.issubset(phase_tags))

    def test_hi_comparison_uses_intrinsic_r_infinity_spectrum(self):
        block = {
            'dsigma_dv': np.array([1.0, 2.0, 3.0]),
            'dsigma_dv_obs': np.array([10.0, 20.0, 30.0]),
        }
        np.testing.assert_array_equal(
            _intrinsic_surface_spectrum(block),
            block['dsigma_dv'],
        )


if __name__ == '__main__':
    unittest.main()
