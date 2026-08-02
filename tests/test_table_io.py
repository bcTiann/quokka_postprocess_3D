from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.tables import (
    AttemptRecord,
    DespoticTable,
    SpeciesLineGrid,
    SpeciesRecord,
    TableLookup,
    load_table,
    save_table,
)
from quokka2s.tables.augment_co21 import (
    _fill_in_hull,
    _source_invalid_mask,
    _with_co21,
)
from quokka2s.tables.solver import (
    _extract_line_result,
    _extract_transition_result,
)


def _table() -> DespoticTable:
    shape = (2, 2, 2)
    values = np.arange(1, 9, dtype=float).reshape(shape)
    line = SpeciesLineGrid(
        freq=values, intIntensity=values, intTB=values, lumPerH=values,
        tau=values, tauDust=values, abundance=values,
    )
    attempt = AttemptRecord(
        0, 0, 1.0, 1e20, 100.0, 50.0, True,
        message="Success", duration=1.0, dvdr_idx=1, dvdr=1e-14,
    )
    return DespoticTable(
        species_data={"CO": SpeciesRecord("CO", values, line, True)},
        tg_final=values,
        nH_values=np.array([1.0, 10.0]),
        col_density_values=np.array([1e20, 1e21]),
        dVdr_values=np.array([1e-15, 1e-14]),
        mu_values=values, cv_values=values, Eint_values=values,
        failure_mask=np.zeros(shape, dtype=bool),
        energy_terms={"LambdaLine.CO": values},
        attempts=(attempt,),
    )


class TableIOTests(unittest.TestCase):
    def test_v5_round_trip_and_lookup(self):
        source = _table()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            save_table(source, path)
            loaded = load_table(path)
        self.assertEqual(loaded.chemistry_network, "GOW")
        self.assertEqual(loaded.escape_geometry, "LVG")
        self.assertEqual(loaded.attempts[0].dvdr_idx, 1)
        self.assertEqual(loaded.attempts[0].dvdr, 1e-14)
        lookup = TableLookup(loaded)
        actual = lookup.temperature(10.0, 1e21, 1e-14)
        self.assertEqual(float(actual), float(source.tg_final[1, 1, 1]))

    def test_co21_record_round_trips_and_looks_up_independently(self):
        source = _table()
        values = source.tg_final * 21.0
        fields = {
            field: values
            for field in ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
        }
        augmented = _with_co21(source, fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "co21_table.npz"
            save_table(augmented, path)
            loaded = load_table(path)

        self.assertIn("CO21", loaded.species)
        np.testing.assert_array_equal(
            loaded.species_data["CO21"].abundance,
            loaded.species_data["CO"].abundance,
        )
        lookup = TableLookup(loaded)
        actual = lookup.line_field("CO21", "lumPerH", 10.0, 1e21, 1e-14)
        self.assertEqual(float(actual), float(values[1, 1, 1]))

    def test_co21_convex_hull_fill_is_log_linear(self):
        axis = np.arange(3, dtype=float)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
        values = 10.0 ** (xx + yy + zz)
        values[1, 1, 1] = np.nan
        filled = _fill_in_hull(values, (axis, axis, axis))
        self.assertAlmostEqual(filled[1, 1, 1], 1.0e3)

    def test_co21_source_mask_rejects_cleaner_garbage_cells(self):
        source = _table()
        tg = np.array(source.tg_final, copy=True)
        failure_mask = np.array(source.failure_mask, copy=True)
        tg[0, 0, 0] = 1.0e8
        tg[0, 1, 0] = np.nan
        failure_mask[0, 0, 1] = True
        invalid = _source_invalid_mask(replace(
            source,
            tg_final=tg,
            failure_mask=failure_mask,
        ))
        self.assertTrue(invalid[0, 0, 0])
        self.assertTrue(invalid[0, 1, 0])
        self.assertTrue(invalid[0, 0, 1])
        self.assertEqual(int(invalid.sum()), 3)

    def test_extracts_second_transition_without_schema_dimension(self):
        fields = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
        transitions = [
            {field: float(index + offset) for offset, field in enumerate(fields)}
            for index in (10, 20)
        ]
        second = _extract_line_result(transitions, 1)
        self.assertEqual(second.freq, 20.0)
        self.assertEqual(second.lumPerH, 23.0)

    def test_extracts_co21_by_levels_when_lamda_order_changes(self):
        fields = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
        co21 = {field: float(20 + offset) for offset, field in enumerate(fields)}
        co21.update(upper=2, lower=1)
        co10 = {field: float(10 + offset) for offset, field in enumerate(fields)}
        co10.update(upper=1, lower=0)
        second = _extract_transition_result([co21, co10], 2, 1)
        self.assertEqual(second.freq, 20.0)
        self.assertEqual(second.lumPerH, 23.0)

    def test_loads_legacy_v4_attempts_without_dvdr_metadata(self):
        source = _table()
        old_attempts = np.empty(1, dtype=[
            ("row_idx", np.int32), ("col_idx", np.int32),
            ("nH", float), ("colDen", float), ("tg_guess", float),
            ("final_Tg", float), ("converged", np.bool_),
            ("message", object), ("duration", float),
        ])
        old_attempts[0] = (0, 0, 1.0, 1e20, 100.0, 50.0, True, "Success", 1.0)
        line = source.species_data["CO"].require_line()
        payload = {
            "version": np.array([4], dtype=np.int32),
            "nH_values": source.nH_values,
            "col_density_values": source.col_density_values,
            "dVdr_values": source.dVdr_values,
            "tg_final": source.tg_final,
            "mu_values": source.mu_values,
            "cv_values": source.cv_values,
            "Eint_values": source.Eint_values,
            "failure_mask": source.failure_mask,
            "species_names": np.array(["CO"], dtype=object),
            "species_is_emitter": np.array([True]),
            "attempts": old_attempts,
            "CO_abundance": source.species_data["CO"].abundance,
        }
        for field in ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust"):
            payload[f"CO_{field}"] = getattr(line, field)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez_compressed(path, **payload)
            loaded = load_table(path)
        self.assertIsNone(loaded.attempts[0].dvdr_idx)
        self.assertIsNone(loaded.attempts[0].dvdr)
        self.assertEqual(loaded.chemistry_network, "GOW")


if __name__ == "__main__":
    unittest.main()
