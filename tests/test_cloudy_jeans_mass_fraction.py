"""Exercise the actual Perl Jeans function and parameter parser without Cloudy.

The harness removes only CIAOLoop's main execution block. Its declarations,
parameter reader, and Jeans implementation are loaded verbatim from the source.
No external Cloudy executable or existing table is used or modified.
"""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from quokka2s.tables.abundances import QUOKKA_MASS_FRACTIONS


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vendor/cloudy_cooling_tools/CIAOLoop_lines"
)
HYDROGEN_MASS_G = 1.67373522381e-24
BOLTZMANN_ERG_K = 1.3806488e-16
GRAVITATIONAL_CGS = 6.67384e-8
CAP_CM = 3.086e20
X_H = QUOKKA_MASS_FRACTIONS["X"]


@unittest.skipUnless(shutil.which("perl"), "CIAOLoop requires Perl")
class CloudyJeansMassFractionTests(unittest.TestCase):
    def run_jeans(self, n_h, temperature, parameters=""):
        source = SOURCE.read_text()
        declarations = source.split("# parse command line", 1)[0]
        subroutines = source[source.index("sub printHelp {"):]
        harness = declarations + subroutines + r'''
readParameterFile(shift @ARGV);
my ($n_h, $temperature) = @ARGV;
printf "%.17g %.17g\n", $coolingMapHydrogenMassFraction,
    calculate_jeans_length($n_h, $temperature);
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            program = directory / "jeans_test.pl"
            program.write_text(harness)
            parameter = directory / "test.par"
            parameter.write_text(parameters)
            return subprocess.run(
                ["perl", str(program), str(parameter), str(n_h), str(temperature)],
                cwd=directory, capture_output=True, text=True, timeout=10,
            )

    def calculate(self, n_h, temperature, parameters=""):
        result = self.run_jeans(n_h, temperature, parameters)
        self.assertEqual(result.returncode, 0, result.stderr)
        return tuple(map(float, result.stdout.split()))

    def test_default_matches_quokka_and_restores_simulation_density(self):
        rho = 1e-18
        temperature = 100.0
        n_h = X_H * rho / HYDROGEN_MASS_G
        fraction, length = self.calculate(n_h, temperature)
        self.assertEqual(fraction, X_H)
        expected = math.pi * math.sqrt(
            (5 / 3) * BOLTZMANN_ERG_K * temperature
            / (GRAVITATIONAL_CGS * HYDROGEN_MASS_G * rho)
        )
        self.assertLess(expected, CAP_CM)
        self.assertAlmostEqual(length / expected, 1.0, places=14)

    def test_historical_override_has_expected_uncapped_scaling(self):
        n_h, temperature = 1e6, 100.0
        _, corrected = self.calculate(n_h, temperature)
        fraction, historical = self.calculate(
            n_h, temperature, "coolingMapHydrogenMassFraction = 0.76\n"
        )
        self.assertEqual(fraction, 0.76)
        self.assertLess(historical, CAP_CM)
        self.assertAlmostEqual(corrected / historical, math.sqrt(X_H / 0.76), places=14)

    def test_cap_is_identical_when_both_uncapped_lengths_exceed_it(self):
        for parameter in ("", "coolingMapHydrogenMassFraction = 0.76\n"):
            with self.subTest(parameter=parameter):
                _, length = self.calculate(1.0, 10000.0, parameter)
                self.assertEqual(length, CAP_CM)

    def test_correction_can_uncap_other_states_near_the_boundary(self):
        # The no-change result is specific to nodes remaining above the cap.
        temperature = 100.0
        historical_uncapped = CAP_CM * 1.02
        n_h = (
            math.pi**2 * (5 / 3) * BOLTZMANN_ERG_K * temperature * 0.76
            / (GRAVITATIONAL_CGS * HYDROGEN_MASS_G**2 * historical_uncapped**2)
        )
        _, historical = self.calculate(
            n_h, temperature, "coolingMapHydrogenMassFraction = 0.76\n"
        )
        _, corrected = self.calculate(n_h, temperature)
        self.assertEqual(historical, CAP_CM)
        self.assertLess(corrected, CAP_CM)
        self.assertAlmostEqual(
            corrected / (historical_uncapped * math.sqrt(X_H / 0.76)),
            1.0, places=14,
        )

    def test_parser_accepts_both_assignment_styles_and_trailing_comment(self):
        for parameter in (
            "coolingMapHydrogenMassFraction 7.6e-1\n",
            "coolingMapHydrogenMassFraction = +.76 # historical diagnostic\n",
            "coolingMapHydrogenMassFraction = 1\n",
        ):
            with self.subTest(parameter=parameter):
                fraction, _ = self.calculate(1.0, 10000.0, parameter)
                self.assertIn(fraction, (0.76, 1.0))

    def test_invalid_fraction_is_rejected_by_production_parser(self):
        for invalid in (
            "", "0", "-0.1", "1.01", "NaN", "nan", "Inf", "-Inf",
            "Infinity", "1e999", "1e-999", "not-a-number", "0.76garbage",
        ):
            with self.subTest(invalid=invalid):
                result = self.run_jeans(
                    1.0, 10000.0, f"coolingMapHydrogenMassFraction = {invalid}\n"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite number with 0 < X_H <= 1", result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
