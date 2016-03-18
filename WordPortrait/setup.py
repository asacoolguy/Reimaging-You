try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize

setup(
  name = 'Query Integral Image',
  package_dir={'wordcloud': ''},
  ext_modules = cythonize("query_integral_image.pyx"),
)