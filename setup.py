####################################################################
# History
####################################################################


VERSION = '0.0.0' # 2024-01-23. Documentation updated

####################################################################
# Imports
####################################################################


import sys
import os
import re
import time
import subprocess
import numpy as np
from glob import glob

import setuptools
from setuptools import setup, find_namespace_packages
from collections import defaultdict



######################################################################
# Directories and inports relative to these driectories
######################################################################


def mmgroup_subdir(*path):
    import mmgroup
    try:
        import mmgroup
        from mmgroup import generate_c
    except:
        ERR = "Please istall the mmgroup package before building"
        raise ModuleNotFoundError(ERR)
    mmgroup_dir = os.path.split(mmgroup.__file__)[0]
    return os.path.join(mmgroup_dir, *path) 
    


ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR =  os.path.realpath(os.path.join(ROOT_DIR, 'src'))
PACKAGE_DIR = os.path.join(SRC_DIR, 'mmgroup_fast')
DEV_DIR = os.path.join(PACKAGE_DIR, 'dev')
C_DIR = os.path.join(DEV_DIR, 'c_files')
LIB_DIR = os.path.join(DEV_DIR, 'lib_files')
PXD_DIR = os.path.join(DEV_DIR, 'pxd_files')
MMGROUP_DIR = mmgroup_subdir()
HEADERS_DIR = os.path.join(DEV_DIR, 'headers')

sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)



####################################################################
# Print platform and command line arguments (if desired)
####################################################################


def print_platform():
    import platform
    print('Build running on platform')
    uname = platform.uname()
    for key in [
        'system', 'node', 'release', 'version', 'machine'
        ]:
        print(" " + key + ":", getattr(uname, key, None))
    print("")


def print_commandline_args():
    print('Command line arguments of setup.py (in mmgroup_fast project):')
    for arg in sys.argv:
        print(' ' + arg)
    print('Current working directory os.path.getcwd():')
    print(' ' + os.getcwd())
    print('Absolute path of file setup.py:')
    print(' ' + os.path.abspath(__file__))
    print('')

print_platform()
print_commandline_args()

####################################################################
# Global options
####################################################################

STAGE = 1
STATIC_LIB = False
NPROCESSES = 16
COMPILER = None
CFLAGS = None
LFLAGS = None
MOCKUP = False
VERBOSE = False
MARCH = None
CFLAGS = None
CFLAGS_AVX512HI = None


#MARCH = "-march=skylake"
#CFLAGS_AVX512HI =  ",".join(["-ffixed-xmm%d" % x for x in range(16,32)])


def parse_global_args(arg_list, start = 1):
    global STAGE, STATIC_LIB, NPROCESSES, COMPILER, CFLAGS, MOCKUP
    global VERBOSE, MARCH, CFLAGS, LFLAGS, CFLAGS_AVX512HI
    # Parse a global option '--stage=i', '--compiler=c', and set variable 
    # ``STAGE`` to the integer value i if such an option is present.
    for i in range(start, len(arg_list)):
        s = arg_list[i]
        if s.startswith('--stage='):
            STAGE = int(s[8:])
            arg_list[i] = None
        elif s.startswith('--compiler='):
            COMPILER = s[11:]
            arg_list[i] = None
        elif s.startswith('--cflags='):
            CFLAGS = s[9:]
            arg_list[i] = None
        elif s.startswith('--lflags='):
            LFLAGS = s[9:]
            arg_list[i] = None
        elif s.startswith('--march='):
            MARCH = s[1:]
            arg_list[i] = None
        elif s.startswith('--static'):
            STATIC_LIB = True
            arg_list[i] = None
        elif s.startswith('--n='):
            NPROCESSES = int(s[4:])
            arg_list[i] = None
        elif s == '--mockup':
            MOCKUP = True
            arg_list[i] = None
        elif s == '-v':
            VERBOSE = True
            arg_list[i] = None
        elif s == '--ffixed-xmmhi':
            # Do not use the AVX512 registers zmm16,...,zmm31
            CFLAGS_AVX512HI = ",".join(["-ffixed-xmm%d" % x for x in range(16,32)])
            arg_list[i] = None
        elif s[:1].isalpha:
            break

global_start = 1
for i, s in enumerate(sys.argv[1:]):
    if s == "global":
        sys.argv[i + 1] = None
        global_start = i + 2
parse_global_args(sys.argv, global_start)
print(sys.argv)
while None in sys.argv: 
    sys.argv.remove(None)


CFLAGS_LIST = [x for x in [CFLAGS, MARCH, CFLAGS_AVX512HI ] if x is not None]
CFLAGS = ",".join(CFLAGS_LIST)

LFLAGS_LIST = [x for x in [LFLAGS] if x is not None]
LFLAGS = ",".join(LFLAGS_LIST)



if os.name == 'nt' and COMPILER not in ['msvc']:
    COMPILER = 'mingw32'


if COMPILER and COMPILER not in ['unix','msvc', 'mingw32']:
    raise ValueError("Unknown compiler '%s'" % COMPILER)


####################################################################
# Check if we are in a 'readthedocs' environment
####################################################################

on_readthedocs = MOCKUP or os.environ.get('READTHEDOCS') == 'True'



####################################################################
# Remove old stuff and import build tools
####################################################################


# Remove old shared libraries before(!) anybody else can grab them!!!


subprocess.check_call([sys.executable, 'cleanup.py',
   '--check-uninstalled', '-pcxs']) 



####################################################################
# import build tools
####################################################################



from mmgroup.generate_c.build_ext_steps import Extension, CustomBuildStep
from mmgroup.generate_c.build_ext_steps import BuildExtCmd
from mmgroup.generate_c.build_shared import shared_lib_name
from config import EXTRA_COMPILE_ARGS, EXTRA_LINK_ARGS


####################################################################
# import shared libraries from mmgroup
####################################################################


def before_all():
    import os, shutil
    import mmgroup
    lib_dir = os.path.split(mmgroup.__file__)[0]
    libs = ["mmgroup_mat24", "mmgroup_mm_op", "mmgroup_mm_reduce"]
    dest_dir = os.path.join("src", "mmgroup_fast")
    #os.makedirs(real_path, exist_ok=True)
    for lib in libs:
        libname = shared_lib_name(lib,'load')
        shutil.copyfile(os.path.join(lib_dir, libname),
            os.path.join(dest_dir, libname))
    h_dir = os.path.join(lib_dir, 'dev', 'headers')
    h_dest_dir = os.path.join(dest_dir, 'dev', 'c_files')
    headers = [f for f in os.listdir(h_dir) if f.endswith('.h')]
    os.makedirs(h_dest_dir, exist_ok=True)
    for h_name in headers:
        shutil.copyfile(os.path.join(h_dir, h_name),
            os.path.join(h_dest_dir, h_name))

before_all()


####################################################################
# Add extensions and shared libraries to package data
####################################################################


header_wildcards = ['*.h']
extension_wildcards =  []
if os.name in ['nt']:
    extension_wildcards =  ['*.pyd', '*.dll']
    header_wildcards = ['*.h', '*.lib']
#elif os.name in ['posix']:
#    extension_wildcards =  ['*.so']


package_data = {
        # If any package contains files as given above, include them:
        'mmgroup_fast': extension_wildcards,
        'mmgroup_fast.dev.headers': header_wildcards,
        'mmgroup_fast.tests.test_mm_op_c': ['*.c', '*.txt'],

}





####################################################################
# Initialize list of external build operations
####################################################################



ext_modules = []


####################################################################
# We have to divide the code generation process 
# into stages, since a library built in a certain stage may be 
# for generating the code used in a subsequent stage.
####################################################################

DIR_DICT = {
   'SRC_DIR' : SRC_DIR,
   'C_DIR' : C_DIR,
   'LIB_DIR' : LIB_DIR,
   'DEV_DIR' : DEV_DIR,
   'PXD_DIR' : PXD_DIR,
   'PACKAGE_DIR': PACKAGE_DIR,
   'STATIC_LIB' : int(STATIC_LIB),
   'NPROCESSES' : str(NPROCESSES),
   'MMGROUP_DIR' : MMGROUP_DIR,
   'HEADERS_DIR' : HEADERS_DIR,
}

DIR_DICT['MOCKUP'] = '--mockup\n' if on_readthedocs else ''
DIR_DICT['COMPILER'] = '--compiler %s\n' % COMPILER if COMPILER else ''
DIR_DICT['CFLAGS'] = '--cflags=' + CFLAGS + "\n" if CFLAGS else ''
DIR_DICT['LFLAGS'] = '--lflags=' + LFLAGS + "\n" if LFLAGS else ''
DIR_DICT['VERBOSE'] = '-v\n' if VERBOSE else ""

GENERATE_START = '''
 {VERBOSE}
 {MOCKUP}
 --py-path {SRC_DIR}
 --out-dir {C_DIR}
 --out-pxd-dir {PXD_DIR}
 --library-path  {PACKAGE_DIR} {MMGROUP_DIR}
'''.format(**DIR_DICT)

MMGROUP_LIBS = ['mmgroup_mat24', 'mmgroup_mm_op', 'mmgroup_mm_reduce']
MMGROUP_BUILD_LIBS = shared_lib_name(MMGROUP_LIBS, 'build', 
                    static=STATIC_LIB)
DIR_DICT['MMGROUP_BUILD_LIBS'] = " ".join(MMGROUP_BUILD_LIBS)


SHARED_START = '''
    {COMPILER}
    {CFLAGS}
    {LFLAGS}
    {MOCKUP}
    --source-dir {C_DIR}
    --include-path {PACKAGE_DIR} {LIB_DIR} {C_DIR}
    --library-path {PACKAGE_DIR} {LIB_DIR}
    --library-path {MMGROUP_DIR}
    --library-dir {LIB_DIR}
    --libraries {MMGROUP_BUILD_LIBS}
    --shared-dir {PACKAGE_DIR}
    --rpath $ORIGIN $ORIGIN/../mmgroup
    --define
    --static {STATIC_LIB}
    --n {NPROCESSES}
'''.format(**DIR_DICT)


SHARED_MMGROUP_START = SHARED_START + '''
    --rpath $ORIGIN/../mmgroup
'''



####################################################################
# copying general files
####################################################################

GENERAL_GENERATE = GENERATE_START + '''
 --source-path {SRC_DIR}/mmgroup_fast/dev/general
 --copy mm_op_fast_types.h
'''.format(**DIR_DICT)


general_steps = CustomBuildStep(
  'Copying general files',
  [sys.executable, 'generate_code.py'] + GENERAL_GENERATE.split(),
)



ext_modules += [
    general_steps
]


####################################################################
# Building a basic extension
####################################################################

DIR_DICT["DLL_NAME"] = "None" if STATIC_LIB else "DISPLAY"

MMGROUP_FAST_DISPLAY_SOURCES = '''
   fast_display.c
'''

MMGROUP_FAST_DISPLAY_GENERATE = GENERATE_START + '''
 --dll {DLL_NAME}
 --source-path {SRC_DIR}/mmgroup_fast/dev/general
 --sources mmgroup_fast_display.h
 --sources
'''.format(**DIR_DICT) + MMGROUP_FAST_DISPLAY_SOURCES + '''
 --pxd  mmgroup_fast_display.pxd
 --pxi
 --pyx  mmgroup_fast_display.pyx
'''

MMGROUP_FAST_DISPLAY_SHARED = SHARED_START + '''
    --name mmgroup_fast_display 
    --sources 
''' + MMGROUP_FAST_DISPLAY_SOURCES 
 

mmgroup_fast_display_extension = Extension('mmgroup_fast.display',
        sources=[
            os.path.join(PXD_DIR, 'mmgroup_fast_display.pyx'),
        ],
        #libraries=['m'] # Unix-like specific
        include_dirs = [ C_DIR ],
        library_dirs = [PACKAGE_DIR, LIB_DIR ],
        libraries = [shared_lib_name('mmgroup_fast_display', 'build_ext', 
                   static=STATIC_LIB)],
        #runtime_library_dirs = ['.'],
        extra_compile_args = EXTRA_COMPILE_ARGS, 
        extra_link_args = EXTRA_LINK_ARGS, 
)

ext_modules += [
    CustomBuildStep(
        'Generating code for extension mmgroup_fast_display',
        [sys.executable, 'generate_code.py'] +
            MMGROUP_FAST_DISPLAY_GENERATE.split(),
        [sys.executable, 'build_shared.py'] +
            MMGROUP_FAST_DISPLAY_SHARED.split(), 
   ),    
]

if not on_readthedocs:
    ext_modules += [
        mmgroup_fast_display_extension,
    ]



####################################################################
# Building an internal header for the extensions at stage 1
####################################################################

MM_OP_FAST_GENERATE_H = GENERATE_START + '''
 --source-path {SRC_DIR}/mmgroup_fast/dev/mm_op_fast
 --tables  mmgroup_fast.dev.mm_op_fast.mm_op_fast_pi
 --sources mm_op_fast_intern.h
 --sources mm_op_fast_permutations.h mm_op_fast_hadamard.h
'''.format(**DIR_DICT)

ext_modules += [
    CustomBuildStep(
        'Generating internal heaader for extension mmgroup_op_fast',
        [sys.executable, 'generate_code.py'] +
            MM_OP_FAST_GENERATE_H.split(),
   ),    
]



####################################################################
# Building the extensions at stage 1
####################################################################

DIR_DICT["DLL_NAME"] = "None" if STATIC_LIB else "MM_OP_FAST"


MM_OP_FAST_SOURCES = '''
   from_mmgroup.c  mm_op_fast_perm.c mm_op_fast_xy.c
   mm_op_fast_conv_mmv.c  mm_op_fast_word.c mm_op_fast_init.c
   mm_op_fast_mode1.c
   mm_op_fast_shuffle.c mm_op_fast_t.c mm_op_fast_xi.c
   mm_op_fast_xi_monomial.c  mm_op_fast_xi_tables.c
   mm_axis3_fast_sym.c mm_axis3_case6A.c mm_axis3_case10A.c
   mm_axis3_case6F.c mm_axis3_case2B.c
   mm_axis3_hadamard256.c
   mm_axis3_case8B.c
   mm_axis3_case4C.c
   mm_axis3_case12C4B.c
   mm_axis3_case6C.c
   mm_axis3_case4A.c
   mm_axis3_fast_reduce.c
'''

MM_OP_FAST_GENERATE = GENERATE_START + '''
 --dll {DLL_NAME}
 --source-path {SRC_DIR}/mmgroup_fast/dev/mm_op_fast
 --source-path {SRC_DIR}/mmgroup_fast/dev/general
 --source-path {SRC_DIR}/mmgroup_fast/dev/mm_op_axis_mod3
 --tables  mmgroup_fast.dev.mm_op_fast.mm_op_fast_pi
           mmgroup_fast.dev.mm_op_fast.mm_op_fast_xi_monomial
           mmgroup_fast.dev.mm_op_axis_mod3.Case6A
           mmgroup_fast.dev.mm_op_axis_mod3.Case10A
           mmgroup_fast.dev.case_2B.axis_class_2B_tables
           mmgroup_fast.dev.mm_op_axis_mod3.Case4C
           mmgroup_fast.dev.mm_op_axis_mod3.Case12C4B
           mmgroup_fast.dev.mm_op_axis_mod3.Case6C
 --sources mm_op_fast.h
 --sources
'''.format(**DIR_DICT) + MM_OP_FAST_SOURCES + '''
 --pxd  mm_op_fast.pxd
 --pxi
 --pyx  mm_op_fast.pyx
'''

# mmgroup_fast.dev.mm_op_axis_mod3.Case10A




STAGE1_SOURCES = MM_OP_FAST_SOURCES
STAGE1_LIBS =  ['mmgroup_fast_display', 'mmgroup_mm_op_fast']
STAGE1_BUILD_LIBS = shared_lib_name(STAGE1_LIBS, 'build', 
                   static=STATIC_LIB)
STAGE1_BUILD_EXT = shared_lib_name(STAGE1_LIBS, 'build_ext', 
                   static=STATIC_LIB)


MM_OP_FAST_SHARED = SHARED_MMGROUP_START + '''
    --name mmgroup_mm_op_fast 
    --sources 
''' + STAGE1_SOURCES + '''
    --libraries 
''' + " ".join([])


mm_op_fast_extension = Extension('mmgroup_fast.mm_op_fast',
        sources=[
            os.path.join(PXD_DIR, 'mm_op_fast.pyx'),
        ],
        #libraries=['m'] # Unix-like specific
        include_dirs = [ C_DIR ],
        library_dirs = [PACKAGE_DIR, LIB_DIR ],
        libraries = STAGE1_BUILD_EXT, 
        #runtime_library_dirs = ['.'],
        extra_compile_args = EXTRA_COMPILE_ARGS, 
        extra_link_args = EXTRA_LINK_ARGS, 
)


ext_modules += [
    CustomBuildStep(
        'Generating code for extension mmgroup_fast_display',
        [sys.executable, 'generate_code.py'] +
            MM_OP_FAST_GENERATE.split(),
        [sys.executable, 'build_shared.py'] +
            MM_OP_FAST_SHARED.split(), 
   )    
]
if not on_readthedocs:
    ext_modules += [
        mm_op_fast_extension,
    ]


####################################################################
# Copy generated header files
####################################################################


print("abc")

COPY_HEADERS = """
  --header-path {C_DIR}
  --header-dir {HEADERS_DIR}
  --lib-path {LIB_DIR}
  --lib-dir {HEADERS_DIR}
""".format(**DIR_DICT)

if not STATIC_LIB and not on_readthedocs:
    copy_headers_step = CustomBuildStep(
       'Copy header files',
      [sys.executable, 'shared_headers.py'] + COPY_HEADERS.split(),
      [sys.executable, 'copy_shared.py', r'${build_lib}'],
    )
    ext_modules.append(copy_headers_step)


####################################################################
# Don't build any externals when building the documentation.
####################################################################



def read(fname):
    '''Return the text in the file with name 'fname' ''' 
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

####################################################################
# The main setup program.
####################################################################

if os.name ==  'posix': 
   EXCLUDE = ['*.dll', '*.pyd', '*.*.dll', '*.*.pyd'] 
elif os.name ==  'nt': 
   EXCLUDE = ['*.so', '*.*.so'] 
else:
   EXCLUDE = [] 


setup(
    name = 'mmgroup_fast',    
    version = VERSION,    
    license='BSD-2-Clause',
    description='Implementation of the sporadic simple monster group.',
    long_description=read('README.rst'),
    author='Martin Seysen',
    author_email='m.seysen@gmx.de',
    url='https://github.com/Martin-Seysen/mmgroup_fast',
    packages=find_namespace_packages(
        where = 'src',
        exclude = EXCLUDE
    ),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=False,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        #'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.6',
        #'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    project_urls={
       # 'Changelog': 'yet unknown',
       'Issue Tracker': 'https://github.com/Martin-Seysen/mmgroup_fast/issues',
    },
    keywords=[
        'sporadic group', 'monster group', 'finite simple group'
    ],
    python_requires='>=3.6',
    install_requires=[
         'numpy', 'regex', 'mmgroup',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'numpy', 'pytest-runner', 'cython', 'regex',
        # 'sphinx',  'sphinxcontrib-bibtex', 'mmgroup',
    ],
    tests_require=[
        'pytest', 'numpy', 'regex', 'pytest-xdist', 'mmgroup',
    ],
    cmdclass={
        'build_ext': BuildExtCmd,
    },
    ext_modules = ext_modules,
    package_data = package_data,
    include_dirs=[np.get_include()],  # This gets all the required Numpy core files
)







