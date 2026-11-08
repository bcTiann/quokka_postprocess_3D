"""Normal package build with an optional, compiler-verified Linux native teams."""
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

from build_support import MODULES, extension_flags, probe_openmp


class VerifiedBuildExt(build_ext):
    """Choose flags after setuptools has selected the active C compiler."""

    def build_extensions(self):
        command = getattr(self.compiler, "compiler", None)
        support = probe_openmp(command)
        self.announce("QUOKKA OpenMP kernels: " + support["reason"], level=2)
        for extension in self.extensions:
            compile_flags, link_flags = extension_flags(
                extension.name,
                openmp_supported=support["supported"],
            )
            extension.extra_compile_args = compile_flags
            extension.extra_link_args = link_flags
        super().build_extensions()


extensions = [
    Extension(
        name=module,
        sources=[str(Path("src") / (module.replace(".", "/") + ".pyx"))],
        include_dirs=[np.get_include()],
    )
    for module in MODULES
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    cmdclass={"build_ext": VerifiedBuildExt},
)
