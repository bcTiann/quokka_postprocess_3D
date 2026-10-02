"""Check the scientific distinction between failed solves and unavailable nodes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.tables.interpolate_failed import interpolate_table


class DespoticInterpolationTests(unittest.TestCase):
    def test_fills_interior_only_and_preserves_solved_high_temperature(self):
        shape = (4, 4, 4)
        failure = np.zeros(shape, dtype=bool)
        failure[1, 1, 1] = True
        failure[0, 0, 0] = True
        temperature = (100 + np.arange(np.prod(shape), dtype=float)).reshape(shape)
        temperature[3, 3, 3] = 2e6  # A successful solve, not a failed node.
        temperature[failure] = np.nan
        mu = np.full(shape, 1.3)
        mu[failure] = np.nan
        source_fields = {
            "tg_final": temperature,
            "mu_values": mu,
            "cv_values": mu * 2,
            "Eint_values": mu * 3,
            "CO_abundance": mu / 10,
            "CO_intIntensity": mu * 4,
            "CO_intTB": mu * 4,
            "CO_lumPerH": mu * 4,
            "CO_tau": mu * 4,
            "CO_tauDust": mu * 4,
            "energy::signed": np.where(np.indices(shape)[0] == 0, -1.0, 1.0),
        }
        source_fields["energy::signed"][failure] = np.nan
        frequency = np.full(shape, 115e9)
        frequency[failure] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.npz"
            filled = Path(directory) / "interpolated.npz"
            np.savez_compressed(
                raw,
                nH_values=np.geomspace(1, 1000, 4),
                col_density_values=np.geomspace(1e18, 1e21, 4),
                dVdr_values=np.geomspace(1e-16, 1e-13, 4),
                failure_mask=failure,
                species_names=np.array(["CO"], dtype=object),
                species_is_emitter=np.array([True]),
                energy_term_names=np.array(["signed"], dtype=object),
                CO_freq=frequency,
                **source_fields,
            )
            counts = interpolate_table(raw, filled)
            with np.load(filled, allow_pickle=True) as result:
                self.assertEqual(counts["source_failure_mask"], 2)
                self.assertEqual(counts["filled_T_mu_node_mask"], 1)
                self.assertEqual(counts["remaining_unavailable_T_mu_node_mask"], 1)
                self.assertTrue(np.array_equal(result["failure_mask"], failure))
                self.assertTrue(result["interpolation_target_mask"][1, 1, 1])
                self.assertTrue(result["filled_T_mu_node_mask"][1, 1, 1])
                self.assertTrue(np.isfinite(result["tg_final"][1, 1, 1]))
                self.assertTrue(result["remaining_unavailable_T_mu_node_mask"][0, 0, 0])
                self.assertTrue(np.isnan(result["tg_final"][0, 0, 0]))
                self.assertEqual(result["tg_final"][3, 3, 3], 2e6)
                for key, original in source_fields.items():
                    np.testing.assert_array_equal(result[key][~failure], original[~failure])
                self.assertTrue(np.all(result["CO_freq"] == 115e9))
            with self.assertRaises(FileExistsError):
                interpolate_table(raw, filled)

            with np.load(raw, allow_pickle=True) as blob:
                payload = {key: blob[key] for key in blob.files}
            payload["new_cell_field"] = np.ones(shape)
            np.savez_compressed(raw, **payload)
            with self.assertRaisesRegex(ValueError, "Unknown raw-table fields"):
                interpolate_table(raw, filled, force=True)

            del payload["new_cell_field"]
            payload["CO_lumPerH"] = np.ones((3, 4, 4))
            np.savez_compressed(raw, **payload)
            with self.assertRaisesRegex(ValueError, "CO_lumPerH has shape"):
                interpolate_table(raw, filled, force=True)


if __name__ == "__main__":
    unittest.main()
