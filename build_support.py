"""Shared extension inventory and compiler checks for normal/offline builds."""
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


MODULES = (
    "quokka2s.products._gaussian_fractions",
    "quokka2s.products.gaussian._histogram_deposit",
    "quokka2s.products.gaussian._histogram_merge",
    "quokka2s.products.gaussian._row_select",
    "quokka2s.products.gaussian._a_templates",
    "quokka2s.cloudy._trilinear",
    "quokka2s.despotic._linear",
    "quokka2s.physics._slab_derive",
    "quokka2s.products.gaussian._spectral_fields",
    "quokka2s.products.gaussian._sparse_expand",
    "quokka2s.products.gaussian._prefix_expand",
    "quokka2s.products.gaussian._layered_deposit",
    "quokka2s.products.gaussian._template_integrate",
)
OPENMP_MODULES = {
    "quokka2s.products.gaussian._histogram_merge",
    "quokka2s.products.gaussian._sparse_expand",
    "quokka2s.products.gaussian._prefix_expand",
}
NUMERICAL_FLAGS = ["-O3", "-fno-fast-math", "-ffp-contract=off"]


def compiler_command():
    """Return Python's configured C compiler, including a compiler wrapper."""
    return shlex.split(sysconfig.get_config_var("CC") or "cc")


def probe_openmp(command=None):
    """Enable OpenMP only when the actual Linux C compiler links its runtime.

    macOS builds deliberately use the serial kernel with no optional runtime.
    Runtime team availability is checked again by the main accumulator.
    """
    if not sys.platform.startswith("linux"):
        return {"supported": False, "reason": "Serial build on this platform"}
    command = compiler_command() if command is None else list(command)
    program = (
        "#include <omp.h>\n"
        "int main(void) { return omp_get_max_threads() < 1; }\n"
    )
    with tempfile.TemporaryDirectory(prefix="quokka-openmp-") as temporary:
        directory = Path(temporary)
        source = directory / "probe.c"
        executable = directory / "probe"
        source.write_text(program)
        invocation = command + ["-fopenmp", str(source), "-o", str(executable)]
        try:
            result = subprocess.run(invocation, capture_output=True, text=True)
        except OSError as error:
            return {"supported": False, "reason": str(error), "command": invocation}
        return {
            "supported": result.returncode == 0,
            "reason": "Compiler/runtime link check passed" if result.returncode == 0 else result.stderr[-4000:],
            "command": invocation,
        }


def extension_flags(module, *, openmp_supported):
    """Keep all float operations strict; only independent output rows use OpenMP."""
    compile_flags = list(NUMERICAL_FLAGS)
    link_flags = []
    if module in OPENMP_MODULES and openmp_supported:
        compile_flags.append("-fopenmp")
        link_flags.append("-fopenmp")
    if module in {
        "quokka2s.products.gaussian._sparse_expand",
        "quokka2s.products.gaussian._prefix_expand",
        "quokka2s.products.gaussian._layered_deposit",
        "quokka2s.products.gaussian._template_integrate",
    } and sys.platform.startswith("linux"):
        link_flags.append("-lm")
    return compile_flags, link_flags
