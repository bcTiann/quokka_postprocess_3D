from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quokka2s.despotic.table_data import AttemptRecord, DespoticTable, SpeciesLineGrid, SpeciesRecord
from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.despotic.table_files import load_table, save_table
from quokka2s.physics.composition import abundance_metadata, SUPERSEDED_ABUNDANCE_SETUP
from quokka2s.despotic.cell_solver import _extract_line_result, _extract_transition_result, validated_solver_metadata


def _table(include_co21=False) -> DespoticTable:
    shape = (2, 2, 2)
    values = np.arange(1, 9, dtype=float).reshape(shape)
    line = SpeciesLineGrid(
        freq=values, intIntensity=values, intTB=values, lumPerH=values,
        tau=values, tauDust=values,
    )
    attempt = AttemptRecord(
        0, 0, 1.0, 1e20, 100.0, 50.0, True,
        message="Success", duration=1.0, dvdr_idx=1, dvdr=1e-14,
    )
    species = {"CO": SpeciesRecord("CO", values, line, True)}
    if include_co21:
        co21_values = values * 21.0
        co21_line = SpeciesLineGrid(
            freq=co21_values, intIntensity=co21_values, intTB=co21_values,
            lumPerH=co21_values, tau=co21_values, tauDust=co21_values,
        )
        species["CO21"] = SpeciesRecord("CO21", values, co21_line, True)
    return DespoticTable(
        species_data=species,
        tg_final=values,
        nH_values=np.array([1.0, 10.0]),
        col_density_values=np.array([1e20, 1e21]),
        dVdr_values=np.array([1e-15, 1e-14]),
        mu_values=values, cv_values=values, Eint_values=values,
        failure_mask=np.zeros(shape, dtype=bool),
        energy_terms={"LambdaLine.CO": values},
        attempts=(attempt,),
    )


class DespoticTableFileTests(unittest.TestCase):
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
        self.assertIsNone(loaded.build_metadata)
        lookup = DespoticLookup(loaded)
        queries = lookup.prepare_queries(
            hydrogen_density_cm3=10.0,
            shielding_NH_cm2=1e21,
            velocity_gradient_s=1e-14,
        )
        actual = lookup.temperature(queries=queries)
        self.assertEqual(float(actual), float(source.tg_final[1, 1, 1]))

    def test_superseded_composition_requires_historical_opt_in(self):
        source = replace(_table(), build_metadata={
            "composition": {"setup": SUPERSEDED_ABUNDANCE_SETUP}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            save_table(source, path)
            with self.assertRaisesRegex(ValueError, "superseded XYZ"):
                load_table(path)
            old = load_table(path, allow_superseded_composition=True)
            self.assertEqual(old.build_metadata["composition"]["setup"],
                             SUPERSEDED_ABUNDANCE_SETUP)

    def test_build_metadata_survives_round_trip(self):
        metadata = {
            "composition": abundance_metadata(),
            "despotic": {"gow_patch": "test recorded patch", "gow_source_sha256": "test hash"},
        }
        source = replace(_table(include_co21=True), build_metadata=metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            save_table(source, path)
            loaded = load_table(path)
        self.assertEqual(dict(loaded.build_metadata), metadata)
        self.assertEqual(
            loaded.build_metadata["composition"]["gow_elemental_abundances"]["xHe"],
            0.1,
        )

    def test_legacy_unknown_composition_is_not_assigned_on_resave(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "old.npz"
            resaved = Path(directory) / "resaved.npz"
            save_table(_table(), original)
            loaded = load_table(original)
            self.assertIsNone(loaded.build_metadata)
            save_table(loaded, resaved)
            with np.load(resaved, allow_pickle=True) as blob:
                self.assertNotIn("build_metadata_json", blob.files)

    def test_build_guard_rejects_unreviewed_gow_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "GOW.py"
            source.write_text("# A GOW implementation without the reviewed refresh\n")
            with patch("quokka2s.despotic.cell_solver.distribution") as package:
                package.return_value.locate_file.return_value = source
                with self.assertRaisesRegex(RuntimeError, "patch_gow_composition.py"):
                    validated_solver_metadata()

    def test_co21_record_round_trips_and_looks_up_independently(self):
        source = _table(include_co21=True)
        values = source.tg_final * 21.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "co21_table.npz"
            save_table(source, path)
            loaded = load_table(path)

        self.assertIn("CO21", loaded.species)
        np.testing.assert_array_equal(
            loaded.species_data["CO21"].abundance,
            loaded.species_data["CO"].abundance,
        )
        lookup = DespoticLookup(loaded)
        queries = lookup.prepare_queries(
            hydrogen_density_cm3=10.0,
            shielding_NH_cm2=1e21,
            velocity_gradient_s=1e-14,
        )
        actual = lookup.line_field(
            species="CO21",
            field_name="lumPerH",
            queries=queries,
        )
        self.assertEqual(float(actual), float(values[1, 1, 1]))

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
        self.assertIsNone(loaded.build_metadata)


if __name__ == "__main__":
    unittest.main()
