from setuptools import Extension, setup

ext = Extension(
    name='webapp',
    sources=['./webapp.c'],
    define_macros=[('PY_SSIZE_T_CLEAN', None)],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
)

setup(
    name='webapp',
    version='0.1.0',
    ext_modules=[ext],
)
