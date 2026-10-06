import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.despotic.build_table import _parse_args, _snapshot_grids


class SnapshotDomainTests(unittest.TestCase):
    def domain(self):
        return dict(selection='all simulation cells', total_cells=20, axes={
            name: dict(minimum=lo, maximum=hi, count=20, invalid_count=0)
            for name, lo, hi in (('nH', 0.123, 987.6), ('NH', 1.2e18, 9.8e22),
                                 ('dVdr', 1.23e-22, 2.78e-12))})

    def load(self, domain):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'domain.json'
            path.write_text(json.dumps(domain))
            return _snapshot_grids(path)

    def test_preserves_exact_all_cell_endpoints_and_log_spacing(self):
        domain = self.domain()
        *grids, recorded = self.load(domain)
        self.assertEqual(recorded, domain)
        for grid, name, count in zip(grids, ('nH', 'NH', 'dVdr'), (35,35,53)):
            axis = grid.sample()
            self.assertEqual(len(axis), count)
            self.assertEqual(axis[0], domain['axes'][name]['minimum'])
            self.assertEqual(axis[-1], domain['axes'][name]['maximum'])
            np.testing.assert_allclose(np.diff(np.log10(axis)), np.diff(np.log10(axis))[0])

    def test_rejects_partial_selection_or_incomplete_axis(self):
        for key, value in (('selection', 'cold subset'), ('total_cells', 21)):
            domain = self.domain()
            domain[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(domain)

    def test_rejects_invalid_or_nonpositive_extrema(self):
        for key, value in (('minimum', 0), ('maximum', float('inf')),
                           ('invalid_count', 1), ('maximum', 0.01)):
            domain = self.domain()
            domain['axes']['nH'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(domain)

    def test_build_command_requires_measured_snapshot_domain(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as missing:
            _parse_args([])
        self.assertEqual(missing.exception.code, 2)
        parsed = _parse_args(["--snapshot-domain", "snapshot_domain.json"])
        self.assertEqual(parsed.snapshot_domain, Path("snapshot_domain.json"))
