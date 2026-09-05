import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_cloudy_sixline_tables import (
    HM12_LOG_NH,
    LINES,
    LOG_NH_DENSITY,
    _write_parameter_file,
)
from scripts.build_hm12_filtered_ism_sixline_bundles import (
    LINES as BUNDLE_LINES,
    _load_grid,
)


class CloudySixLineBuildWorkflowTests(unittest.TestCase):
    def test_parameter_file_uses_attenuation_init_axis_and_jeans_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "grid.par"
            _write_parameter_file(
                path,
                cloudy_exe=directory / "cloudy.exe",
                output_dir=directory / "output",
                smoke=False,
            )
            text = path.read_text()
            self.assertIn("coolingMapUseJeansLength = 1", text)
            self.assertIn("coolingMapMaximumJeansLength = 3.086e20", text)
            self.assertNotIn("loop [stop column density]", text)
            self.assertIn(
                'loop [init "HM12_ATTENUATION_ISM_NH21/logNH*.out"] '
                "18 18.5 19 19.5 20 20.5 21",
                text,
            )
            self.assertIn("command CMB redshift 0", text)
            self.assertEqual(text.count("lineMapLine = "), len(LINES))

    def test_bundle_loader_uses_headers_not_run_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_t = np.linspace(np.log10(3.6), 9.0, 21)
            combinations = [
                (attenuation, density)
                for density in reversed(LOG_NH_DENSITY)
                for attenuation in HM12_LOG_NH
            ]
            for run, (attenuation, density) in enumerate(combinations, start=1):
                path = directory / f"synthetic_run{run}.dat"
                lines = [
                    "# Line Emissivity Map File",
                    f"# hden {density:.15g}",
                    (
                        '# init "HM12_ATTENUATION_ISM_NH21/'
                        f'logNH{attenuation:g}.out"'
                    ),
                    "#Te   " + "  ".join(item[2] for item in BUNDLE_LINES),
                ]
                for temperature in log_t:
                    value = attenuation + density + temperature
                    lines.append(
                        f"{temperature:.8f}  "
                        + "  ".join(
                            f"{value + index:.8f}"
                            for index in range(len(BUNDLE_LINES))
                        )
                    )
                path.write_text("\n".join(lines) + "\n")

            attenuation, density, raw = _load_grid(
                directory, log_t, np.asarray(HM12_LOG_NH)
            )
            np.testing.assert_array_equal(attenuation, np.asarray(HM12_LOG_NH))
            np.testing.assert_allclose(density, np.sort(LOG_NH_DENSITY))
            self.assertEqual(raw.shape, (len(BUNDLE_LINES), 7, 10, 21))
            expected = attenuation[3] + density[4] + log_t[5]
            self.assertAlmostEqual(raw[0, 3, 4, 5], expected, places=6)


if __name__ == "__main__":
    unittest.main()
