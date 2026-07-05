from __future__ import annotations

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
