from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py

from quokka2s.pipeline.intermediate_io import load_all_builds


class IntermediateIOTests(unittest.TestCase):
    def test_load_all_builds_skips_obsolete_task_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            cache_dir = output_dir / 'task_intermediates'
            cache_dir.mkdir()
            current = cache_dir / 'Build_PhaseHist_aaaaaaaa.h5'
            stale = cache_dir / 'Build_PhaseHist_bbbbbbbb.h5'
            with h5py.File(current, 'w') as handle:
                handle.attrs['cache_key'] = f'current:{current.name}'
                handle.attrs['tag'] = 'current'
            with h5py.File(stale, 'w') as handle:
                handle.attrs['cache_key'] = 'old-key'
                handle.attrs['tag'] = 'obsolete'

            expected = lambda _config, name: f'current:{name}'
            with patch(
                'quokka2s.pipeline.intermediate_io._expected_sibling_key',
                side_effect=expected,
            ):
                results = load_all_builds(
                    output_dir,
                    'Build_PhaseHist',
                    config=object(),
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['tag'], 'current')


if __name__ == '__main__':
    unittest.main()
