
# cython: language_level=3

from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


cdef extern from "mmgroup_fast_display.h":
    const char *GCC_CAPABILITIES 



include "mmgroup_fast_display.pxi"
cimport mmgroup_fast_display as mfd



def  gcc_capabilities():
     cdef char* c_string = mfd.mmgroup_fast_gcc_capabilities()
     cdef bytes py_string
     py_string = <bytes> c_string
     return py_string.decode('UTF-8') 




