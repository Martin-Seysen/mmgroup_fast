
from __future__ import absolute_import, division, print_function
#from __future__ import  unicode_literals



import sys
import os
import itertools
from collections import defaultdict, Counter
from numbers import Integral
import numpy as np
import subprocess
import shutil

import pytest

import mmgroup
from mmgroup.generate_c.build_shared import shared_lib_name 

 
MY_PATH = os.path.split(__file__)[0]
print(MY_PATH)
OUT_PATH = os.path.join(MY_PATH, "out")
print(OUT_PATH)

BASE_PATH = os.path.realpath(os.path.join(MY_PATH,'..','..'))
DEV_PATH = os.path.join(BASE_PATH, 'dev')
print(DEV_PATH)

MMGROUP_BASE_PATH = os.path.split(mmgroup.__file__)[0]
MMGROUP_DEV_PATH = os.path.join(MMGROUP_BASE_PATH, 'dev')

ALL_BASE_PATHS = [BASE_PATH, MMGROUP_BASE_PATH]
ALL_DEV_PATHS = [DEV_PATH, MMGROUP_DEV_PATH]


def default_compiler():
    return "gcc"

CC = default_compiler()

SOURCES = ["test_mmfast3.c"]
SOURCE_PATHS = [os.path.join(MY_PATH, src) for src in SOURCES] 

SHARED = [
   "mmgroup_fast_display", "mmgroup_mat24",
   "mmgroup_mm_op", "mmgroup_mm_op_fast",
   "mmgroup_mm_reduce",
]


def load_shared_name(shared_lib):
    if os.name == "nt":
        return shared_lib_name(shared_lib, "load")
    else:
        return shared_lib_name(shared_lib, "load")
 
def build_shared_name(shared_lib):
    if os.name == "nt":
        return shared_lib
    else:
        return shared_lib



   
BUILD_SHARED = [
   build_shared_name(shared_lib) for shared_lib in SHARED
]

H_PATHS =  [
   os.path.join(path, "headers") for path in ALL_DEV_PATHS
]

LIB_PATHS = H_PATHS + ALL_BASE_PATHS


"""
def lib_paths():
    if os.name == "posix":
        LIB_PATHS = ALL_BASE_PATHS
    elif os.name == "nt":
        LIB_PATHS = ALL_BASE_PATHS
    else:
        ERR = "Don't know how to use a shared library on platform '%s'" 
        raise ValueError(ERR % os.name)
    return LIB_PATHS

LIB_PATHS = lib_paths()
"""

def exe_name():
    if os.name == "posix":
        name = "a.out"
    elif os.name == "nt":
        name = "a.exe"
    else:
        ERR = "Don't know the standard executable name on platform '%s'" 
        raise ValueError(ERR % os.name)
    return name

EXE_NAME = os.path.join(OUT_PATH, exe_name())
print(EXE_NAME)

CFLAGS = ["-O3", "-g"]


def ensure_directory(path):
    """
    Create the directory at 'path' if it does not already exist.
    Equivalent to 'mkdir -p' in Unix shells.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory ensured: {path}")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")
        sys.exit(1)



def find_and_copy(filenames, src_dirs, dest_dir, verbose = 0):
    """
    Search for each file in 'filenames' in each directory in 'src_dirs'.
    If found, copy it into 'dest_dir' and return the path it was copied from.
    """
    for filename in filenames:
        done = False
        for src in src_dirs:
            candidate = os.path.join(src, filename)
            if os.path.isfile(candidate):
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(candidate, dest_path)
                if verbose:
                    print(f"Copied '{candidate}' → '{dest_path}'")
                done = True
                break
        if not done:
            raise IOError(f"File '{filename}' not found")
        


def link_rpath():
    if os.name == "posix":
        return ["-Wl,-rpath,$ORIGIN"]
    else:
        return []


def gcc_invocation():
    l = [CC] + CFLAGS + SOURCE_PATHS + ["-o", EXE_NAME]
    for h in H_PATHS:
        l.append("-I"+h)
    for lib in LIB_PATHS:
        l.append("-L"+lib)
    for shared in BUILD_SHARED:
        l.append("-l"+shared)
    l += link_rpath()
    return l



def compile():
    ensure_directory(OUT_PATH)
    gcc_par = gcc_invocation()
    print(" ".join(gcc_par))
    subprocess.check_call(gcc_par) 
    shared = [load_shared_name(shared) for shared in SHARED]
    find_and_copy(shared, ALL_BASE_PATHS, OUT_PATH)



IN_FILE = "inp.txt"
OUT_FILE = "outp.txt"


def run(align=0):
    find_and_copy([IN_FILE], [MY_PATH], OUT_PATH)
    args = [EXE_NAME, IN_FILE, OUT_FILE, str(align)]
    subprocess.check_call(
        args,
        cwd = OUT_PATH
    )


@pytest.mark.compiler
@pytest.mark.slow
@pytest.mark.alignment
def test_C_program():
    print("\nTest compiling and running pure C program")
    compile()
    for align in (4, 8, 16):
        run(align)
    print("passed")

    




#print(os.name)
