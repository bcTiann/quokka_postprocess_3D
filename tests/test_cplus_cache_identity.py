from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quokka2s.pipeline.cache import field_cache_key


class CplusCacheIdentityTests(unittest.TestCase):
    def test_unrelated_field_uses_base_key(self):
        self.assertEqual(
            field_cache_key('base', ('gas', 'CO_luminosity')),
            'base',
        )

    def test_chianti_field_key_tracks_nu_table_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'nu.npz'
            table.write_bytes(b'v1')
            first = field_cache_key(
                'base',
                ('gas', 'C+_luminosity_chianti'),
                cii_table_path=table,
            )
            table.write_bytes(b'v2-longer')
            second = field_cache_key(
                'base',
                ('gas', 'C+_luminosity_chianti'),
                cii_table_path=table,
            )
        self.assertNotEqual(first, second)

    def test_lte_and_chianti_fields_have_distinct_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'nu.npz'
            table.write_bytes(b'lookup')
            lte = field_cache_key('base', ('gas', 'C+_luminosity_lte'))
            chianti = field_cache_key(
                'base',
                ('gas', 'C+_luminosity_chianti'),
                cii_table_path=table,
            )
        self.assertNotEqual(lte, chianti)


if __name__ == '__main__':
    unittest.main()
