from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.tables import DespoticTable
from quokka2s.tables.abundances import abundance_metadata
from quokka2s.tables.dvdr_domain import legacy_dvdr_values


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extend_despotic_dvdr_table.py"
_SPEC = importlib.util.spec_from_file_location("despotic_extension_under_test", _SCRIPT)
extension = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extension)


def _table(axis, metadata) -> DespoticTable:
    values = np.ones((1, 1, len(axis)))
    return DespoticTable(
        species_data={}, tg_final=values,
        nH_values=np.array([1.0]), col_density_values=np.array([1e20]),
        dVdr_values=np.asarray(axis), mu_values=values,
        cv_values=values, Eint_values=values, build_metadata=metadata,
    )


class DespoticExtensionProvenanceTests(unittest.TestCase):
    def test_unknown_or_different_source_is_rejected_before_solving(self):
        current = {"composition": abundance_metadata()}
        old_metadata = {"composition": {"setup": "old abundances"}}
        for source_metadata in (None, old_metadata):
            with self.subTest(metadata=source_metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "old.npz"
                source.touch()
                args = argparse.Namespace(
                    source=source, output=root / "new.npz", clean_output=root / "clean.npz",
                    parts_dir=root / "parts", workers=1, force=False, skip_clean=True,
                )
                table = _table(legacy_dvdr_values(), source_metadata)
                with (
                    patch.object(extension, "_parse_args", return_value=args),
                    patch.object(extension, "load_table", return_value=table),
                    patch.object(extension, "validated_solver_metadata", return_value=current),
                    patch.object(extension, "build_gow_lvg_table") as solve,
                ):
                    with self.assertRaisesRegex(ValueError, "source table"):
                        extension.main()
                    solve.assert_not_called()
                self.assertFalse(args.parts_dir.exists())
                self.assertFalse(args.output.exists())

    def test_checkpoint_concatenation_preserves_and_checks_provenance(self):
        metadata = {"composition": abundance_metadata()}
        first = _table([1e-16], metadata)
        second = _table([1e-15], metadata)
        joined = extension._concatenate_dvdr_tables([first, second])
        self.assertEqual(dict(joined.build_metadata), metadata)
        np.testing.assert_array_equal(joined.dVdr_values, [1e-16, 1e-15])
        different = replace(second, build_metadata={"composition": {"setup": "different"}})
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            extension._concatenate_dvdr_tables([first, different])
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            extension._merge_tables(first, different)


if __name__ == "__main__":
    unittest.main()
