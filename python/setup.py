from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "covreg_bindings",
    ext_modules = cythonize('covreg/*.pyx', output_dir='covreg')
)