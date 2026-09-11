"""Exercise CIAOLoop's real file lifecycle with a deliberately fake Cloudy.

The fake produces one successful state then fails before writing any save file.
It tests evidence retention only and does not represent a physical calculation.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def run_success_then_early_crash(directory: Path) -> dict:
    """Run the real Perl loop in a fresh directory with a two-state test double."""
    directory.mkdir()
    executable = directory / "fake_cloudy.py"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import re
import sys
from pathlib import Path

commands = sys.stdin.read()
temperature = float(re.search(r"constant temperature ([0-9.eE+-]+)", commands)[1])
if temperature > 15000:
    print("INTENTIONAL MOCK FAILURE before save commands; exit code 7")
    sys.exit(7)
for line in commands.splitlines():
    if line.startswith("save last lines"):
        name = re.search(r'"(.*?)"', line)[1]
        Path(name).write_text("# MOCK emissivities\n" + "\t".join(["1e18"] + ["1e-24"] * 8) + "\n")
    elif line.startswith("punch last physical"):
        name = re.search(r'"(.*?)"', line)[1]
        Path(name).write_text("# MOCK physical\n1e18\t10000\t1\t0.01\n")
    elif line.startswith("save last radius"):
        name = re.search(r'"(.*?)"', line)[1]
        Path(name).write_text("# MOCK radius\n1\t1e30\t1e18\t1e17\n")
print("MOCK calculation only; Cloudy exited OK")
''')
    executable.chmod(0o755)
    output = directory / "maps"
    output.mkdir()
    parameter = directory / "mock.par"
    parameter.write_text("\n".join([
        f"cloudyExe = {executable}", "saveCloudyOutputFiles = 1", "exitOnCrash = 0",
        "outputFilePrefix = mock", f"outputDir = {output}", "cloudyRunMode = 4",
        *(f"lineMapLine = H 1 {number}A" for number in range(1, 9)),
        "coolingMapUseJeansLength = 0", "coolingMapTmin = 10000",
        "coolingMapTmax = 20000", "coolingMapTpoints = 2",
        "loop [hden] 0", "loop [radius 1e30 * linear] 3.0856775809623245e19",
    ]) + "\n")
    source = Path(__file__).resolve().parents[1] / "vendor/cloudy_cooling_tools/CIAOLoop_lines"
    result = subprocess.run(["perl", str(source), str(parameter)], cwd=directory,
                            capture_output=True, text=True, timeout=30)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise AssertionError(f"CIAOLoop failed: {result.returncode}\n{result.stdout}\n{result.stderr}")
    prefix = output / "mock_run1"
    contents = {suffix: prefix.with_suffix(suffix).read_text()
                for suffix in (".dat", ".cloudyIn", ".cloudyOut", ".lines", ".physical", ".radius", ".process")}
    rows = [line.split() for line in contents[".dat"].splitlines()
            if line.strip() and not line.startswith("#")]
    if len(rows) != 2 or len(rows[0]) != 9 or rows[1] != ["4.301"]:
        raise AssertionError(f"wrong success/crash map rows: {rows}")
    for suffix in (".cloudyIn", ".cloudyOut"):
        if "T = 1.000e+04" not in contents[suffix] or "T = 2.000e+04" not in contents[suffix]:
            raise AssertionError(f"missing successful/failed input or output: {suffix}")
    if contents[".cloudyIn"].splitlines().count("radius 1e30 3.0856775809623245e19 linear") != 2:
        raise AssertionError("fixed model thickness was not preserved for both temperatures")
    for suffix in (".lines", ".physical", ".radius"):
        if "T = 1.000e+04" not in contents[suffix] or "T = 2.000e+04" in contents[suffix]:
            raise AssertionError(f"stale successful diagnostic attributed to failed state: {suffix}")
    process = [[float(a), int(b)] for a, b in
               (line.split() for line in contents[".process"].splitlines())]
    if process != [[10000.0, 0], [20000.0, 7 << 8]]:
        raise AssertionError(f"wrong process exit evidence: {process}")
    if list(output.glob("*.temp")):
        raise AssertionError("temporary output files remain after completion")
    return {"test_double_only": True, "states": 2, "success_rows": 1, "crash_rows": 1,
            "stale_diagnostics_under_failed_temperature": False,
            "raw_process_status": process, "maps": str(output), "runner_returncode": result.returncode}


@unittest.skipUnless(shutil.which("perl"), "CIAOLoop requires Perl")
class CloudyLineMapDiagnosticsTests(unittest.TestCase):
    def test_success_then_early_crash_retains_input_and_exit_without_stale_saves(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_success_then_early_crash(Path(temporary) / "case")


if __name__ == "__main__":
    unittest.main()
