"""Build numerical loops; grids, tables and physical settings stay in Python."""

from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        name="quokka2s._table_kernels",
        sources=["src/quokka2s/_table_kernels.pyx"],
        extra_compile_args=["-O3", "-ffp-contract=off"],
    ),
    Extension(
        name="quokka2s.products.gaussian._gaussian_kernels",
        sources=["src/quokka2s/products/gaussian/_gaussian_kernels.pyx"],
        extra_compile_args=["-O3", "-ffp-contract=off"],
    ),
    Extension(
        name="quokka2s.products.gaussian._spectral_kernels",
        sources=["src/quokka2s/products/gaussian/_spectral_kernels.pyx"],
        extra_compile_args=["-O3", "-ffp-contract=off"],
    ),
]

setup(
    ext_modules=cythonize(
        module_list=extensions,
        compiler_directives={"language_level": 3},
        build_dir="build/cython",
    ),
)
