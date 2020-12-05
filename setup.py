from setuptools import setup, find_packages, Command, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Having Cython installed is required")

import numpy
import os

_project_name = "FL"
_version = 1.0
_license = ""
_short_description = "A Python Module to perform Federated Learning"
_long_description = """
"""

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: GPU :: NVIDIA CUDA",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]

includes = [
    numpy.get_include(),
    '.'
]

extra_compile_args = [
    "-fopenmp", # OpenMP
    "-DMKL_ILP64", # Intel MKL
    "-m64", # Intel MKL
    "-I${MKLROOT}/include", # Intel MKL
    # "-fsanitize=address" # Uncomment to check for addressing issues in C code
]

extra_link_args = [
    "-fopenmp", # Do not remove
    "-Wl,-rpath,${MKLROOT}/lib", # Intel MKL
    # "-fsanitize=address" # Uncomment to check for addressing issues in C code
]

lib_paths = [
    "${MKLROOT}/lib" # Intel MKL
]

libs = [
    # Intel MKL and requisites
    "mkl_intel_ilp64",
    "mkl_tbb_thread",
    "mkl_core",
    "tbb",
    "c++",
    "pthread",
    "m",
    "dl"
]

def find_file_by_ext(ext, path='.'):
    _files = []
    for root, dirn, files in os.walk(path):
        if dirn == '.venv':
            continue
        for fname in files:
            if fname.endswith(ext):
                _files.append(os.path.join(root, fname))
    return _files

def find_dirs(path='.', names=None):
    _dirs = []
    if names is not None and isinstance(names, list):
        for root, dirs, _ in os.walk(path):
            if len(dirs) > 0 and dirs[0] in names:
                dname = "{}/{}".format(root, dirs[0])
                if not dname in dirs:
                    _dirs.append(dname)
    return _dirs

def build_extensions():
    print("Building extensions...")
    exts = []
    files = find_file_by_ext(ext='.pyx', path='FL')
    for file in files:
        module = os.path.splitext(file)[0].replace(os.path.sep, '.')
        ext = Extension(module, [file], 
                        include_dirs=includes, 
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        libraries=libs,
                        library_dirs=lib_paths)
        exts.append(ext)
    return exts

class CleanCmd(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        srcs = ['./build', './*.egg-info', './dist', '*/__pycache__']
        files = []
        files.extend([f.replace('.pyx', '.c') for f in find_file_by_ext(ext='.pyx', path="./FL")])
        files.extend(find_file_by_ext(ext='.pyc', path="./FL"))
        for f in files + srcs:
            os.system('rm -vrf {}'.format(f))
            pass

_cmdclass = {
    'clean': CleanCmd
}

setup(
    name=_project_name,
    version=_version,
    license=_license,
    description=_short_description,
    long_description=_long_description,
    setup_requires=['cython'],
    ext_modules= cythonize(build_extensions(), language_level=3, annotate=True),
    cmdclass=_cmdclass,
    packages=find_packages(),
    classifiers=classifiers,
    zip_safe=False
)
