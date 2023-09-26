from setuptools import Extension, setup

ext = Extension(
    name='webgl',
    sources=['./webgl.c'],
    define_macros=[('PY_SSIZE_T_CLEAN', None)],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
)

setup(
    name='webgl',
    version='0.1.0',
    ext_modules=[ext],
)
