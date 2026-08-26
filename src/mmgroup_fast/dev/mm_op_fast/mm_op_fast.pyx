
# cython: language_level=3

from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


#cimport mm_op_fast as op_fast

from collections.abc import Iterable
from numbers import Integral
from mmgroup.structures.abstract_mm_group import AbstractMMGroupWord
from mmgroup import MMVector, MM0
from mmgroup.generators import mm_group_invert_word



import numpy as np
from libc.string cimport memcpy 
from mm_op_fast cimport mm_op_fast_alloc, mm_op_fast_dealloc
from mm_op_fast cimport mm_axis3_fast_mode1_set_vstd
from mm_op_fast cimport mm_op_fast_to_mmv, mm_op_fast_from_mmv
from mm_op_fast cimport mm_op_fast_word, mm_op_fast_raw_vb_data
from mm_op_fast cimport mm_op_fast_get_slack
from mm_op_fast cimport mm_axis3_fast_load
from mm_op_fast cimport mm_axis3_fast_load_a
from mm_op_fast cimport mm_axis3_fast_load_sub_row
from mm_op_fast cimport mm_axis3_fast_copy
from mm_op_fast cimport mm_axis3_fast_data_ptr 
from mm_op_fast cimport mm_axis3_fast_echelon
from mm_op_fast cimport mm_axis3_fast_intersect
from mm_op_fast cimport mm_axis3_fast_to_leech_mod3
from mm_op_fast cimport mm_axis3_fast_rand_short_nonzero
from mm_op_fast cimport mm_axis3_fast_num_entries_BC
from mm_op_fast cimport mm_axis3_fast_analyze_case_6A
from mm_op_fast cimport mm_axis3_fast_analyze_v4
from mm_op_fast cimport mm_axis3_fast_find_v4
from mm_op_fast cimport mm_axis3_fast_rand_v
from mm_op_fast cimport mm_axis3_fast_op_G_x0
from mm_op_fast cimport mm_axis3_fast_num_entries_A_t
from mm_op_fast cimport mm_axis3_fast_find_exp_t
from mm_op_fast cimport mm_op_fast_mode1_put
from mm_op_fast cimport mm_op_fast_mode1_get
from mm_op_fast cimport mm_op_fast_mode1_zero
from mm_op_fast cimport mm_op_fast_mode1_poke
from mm_op_fast cimport mm_axis3_fast_reduce_axes
from mm_op_fast cimport mm_axis3_fast_reduce_G_x0
from mm_op_fast cimport mm_axis3_fast_reduce_v_g
from mm_op_fast cimport mm_op_fast_copy_data
from mm_op_fast cimport mm_op_fast_buffer_alloc
from mm_op_fast cimport mm_op_fast_buffer_free
from mm_op_fast cimport mm_op_fast_buffer_gc
from mm_op_fast cimport mm_op_fast_buffer_stat
from mm_op_fast cimport mm_op_fast_buffer_test_start
from mm_op_fast cimport mm_op_fast_buffer_test_stop

from mm_op_fast cimport fast_g_obj_new
from mm_op_fast cimport fast_g_obj_delete
from mm_op_fast cimport fast_g_obj_freeze
from mm_op_fast cimport fast_g_obj_store_g
from mm_op_fast cimport fast_g_obj_copy
from mm_op_fast cimport fast_g_obj_mulexp
from mm_op_fast cimport fast_g_obj_mulexp_obj
from mm_op_fast cimport fast_g_obj_get_mat
from mm_op_fast cimport fast_g_obj_store_int_fast
from mm_op_fast cimport fast_g_obj_nonneutral
from mm_op_fast cimport fast_g_obj_setpower
from mm_op_fast cimport fast_g_obj_get_status
from mm_op_fast cimport fast_g_obj_augment

include "mm_op_fast.pxi"


MAX_NROWS = {
   3:4, 7:2, 15:1, 31:1, 127:1, 255:1
}

ORBIT_TYPES = [
    '2Ae', '2A','2B','4A','4B','4C','6A','6C','8B','6F','10A','10B','12C'
]
ORBIT_DICT = {}
for i, s in enumerate(ORBIT_TYPES):
    ORBIT_DICT[s] = i


def mm_axis3_fast_orbit_dict():
    return ORBIT_DICT



cdef object mm_compress_array_to_int(const uint64_t *a):
    cdef Py_ssize_t i
    cdef object result = 0
    for i in range(3, -1, -1):
        result = (result << 64) + a[i]
    return result

cdef class MMOpFastMatrix:
    cdef mmv_fast_matrix_type *m 
   
    @staticmethod
    def _complain(res, method):
        if (res < 0):
            err = "Internal error %s in class MMOpFastArray, method %s"
            raise ValueError(err % (hex(res), method))  

    def __cinit__(self, *args, **kwds):
        self.m = NULL

    def  __dealloc__(self):
        if self.m != NULL:
            mm_op_fast_dealloc(self.m)

    def __init__(self, uint32_t p, uint32_t nrows, uint32_t mode = 1):
        if not p in MAX_NROWS:
            raise ValueError("Bad modulus %s for class MMOpFastArray" % p) 
        self.m = mm_op_fast_alloc(p, nrows, mode)
        if self.m == NULL:
             raise ValueError("Too many rows or bad modulus for class MMOpFastArray") 

    def copy(self):
        """Return deep copy of matrix object"""
        cp = MMOpFastMatrix(self.m.p, self.m.nrows, self.m.mode)
        cdef mmv_fast_matrix_type *pc = cp.m
        cdef int32_t status = mm_op_fast_copy_data(self.m, pc)
        assert status >= 0
        return cp

    def _display_status(self):
        print("Status of MMOpFastMatrix object:")
        print(
         "  p = %d, nrows = %d, mode = %d, chk_undrfl = %d, refcnt = %d"
         % (self.m.p, self.m.nrows, self.m.mode, self.m.check_underflow, 
             self.m.work_refcount))

    def set_vstd(self, uint32_t hash = 0):
        cdef int32_t status = mm_axis3_fast_mode1_set_vstd(self.m, hash)
        assert status == 0, status

              
    def set_row(self, uint32_t i, row):
        if i >= self.m.nrows:
            raise IndexError("Row index out of range in class MMOpFastArray")
        cdef uint_mmv_t[:] row_view
        cdef int32_t status
        if isinstance(row, MMVector):
            if row.p == self.m.p:
                row_view = row.data
                status = mm_op_fast_from_mmv(self.m, i, &row_view[0])
                if status < 0:
                    self._complain(status, "set_row")
            else:
                raise ValueError("Mismatch of modulus in class  MMOpFastArray")
        else:
            raise TypeError("Bad type of row object in class  MMOpFastArray")

    def row_as_mmv(self, uint32_t i):
        if i >= self.m.nrows:
            raise IndexError("Row index out of range in class MMOpFastArray")
        cdef int32_t status
        v = MMVector(self.m.p)
        cdef uint_mmv_t[:] row_view = v.data
        status = mm_op_fast_to_mmv(self.m, i, &row_view[0], len(v.data))
        if status < 0:
            self._complain(status, "row_as_mmv")
        return v

    @cython.boundscheck(False)
    def mul_exp(self, g, int32_t e = 1):
        """Multiply the vector with ``g ** e`` inplace

        Here ``g`` is an element of the monster group represented
        as an instance of class |MM| and ``e`` is an integer.
        The vector is updated and the updated vector is returned.
        """
        cdef uint32_t[:] g_data
        if isinstance(g, np.ndarray):
            g_data = g
        else:
            g_data = g.mmdata
        cdef int32_t status
        status = mm_op_fast_word(self.m, &g_data[0], len(g_data), e)
        if status >= 0:
            #status =  mm_op_fast_dealloc(self.m, 1)
            pass
        if status < 0:
            self._complain(status, "mul_exp")
        return self

    @cython.boundscheck(False)
    def mul_exp_bench(self, g, int32_t e = 1, uint32_t n = 1):
        """Multiply the vector with ``g ** (e * n)`` inplace

        Here ``g`` is an element of the monster group represented
        as an instance of class |MM| and ``e`` is an integer.
        The vector is updated and the updated vector is returned.
        """
        import time
        cdef uint32_t[:] g_data
        if isinstance(g, np.ndarray):
            g_data = g
        else:
            g_data = g.mmdata
        cdef uint32_t status = 0, i
        t = time.time()
        for i in range(n):
             status |= mm_op_fast_word(
                 self.m, &g_data[0], len(g_data), e) < 0
        t =  time.time() - t
        if status:
            err = "Error in class MMOpFastArray, method mul_exp_bench"
            raise ValueError(err) 
        return t 

    def num_entries_A_t(self, uint32_t row):
        assert 0 <= row < 4
        cdef int32_t res = mm_axis3_fast_num_entries_A_t(self.m, row)
        assert res >= 0, res
        return res & 0xffff, res >> 16

    def find_exp_t(self, uint32_t row, ax_type):
        if isinstance(ax_type, str):
            ax_type = ORBIT_DICT[ax_type] 
        cdef uint32_t ax_t = ax_type 
        cdef t = mm_axis3_fast_find_exp_t(self.m, row, ax_t)
        assert t >= 0, (ax_t, t)
        return t

    def get_data(self, data):
        a = np.array(data, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef uint32_t la = len(a)
        cdef int32_t status
        if la:
            status = mm_op_fast_mode1_get(self.m, &r[0], la)
            assert status >= 0
        return a

    def put_data(self, data):
        a = np.array(data, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef uint32_t la = len(a)
        cdef int32_t status
        if la:
            status = mm_op_fast_mode1_put(self.m, &r[0], la)
            assert status >= 0

    def zero_data(self):
        mm_op_fast_mode1_zero(self.m)

    def _poke(self, uint32_t index, uint32_t value):
        mm_op_fast_mode1_poke(self.m, index, value)

    def reduce_axes(self):
        a = np.zeros(128, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef int32_t status
        status = mm_axis3_fast_reduce_axes(self.m, &r[0], 128)
        assert status >= 0, status
        return a[:status]    

    def reduce_G_x0(self):
        a = np.zeros(12, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef int32_t status
        status = mm_axis3_fast_reduce_G_x0(self.m, &r[0])
        assert status >= 0, status
        return a[:status]    

    def reduce_v_g(self, uint32_t mode = 0x1e):
        a = np.zeros(80, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef int32_t status
        cdef uint64_t *p_dummy = NULL
        status = mm_axis3_fast_reduce_v_g(self.m, &r[0], len(a), p_dummy, mode);
        assert status >= 0, status
        return a[:status]    

    def reduce_v_g_as_int(self):
        cdef uint32_t d[80]
        cdef uint32_t[:] d_view = d
        cdef uint64_t a[4]
        cdef uint64_t[:] a_view = a
        cdef int32_t status
        status = mm_axis3_fast_reduce_v_g(self.m, &d_view[0], 80, &a_view[0], 0x1b);
        assert status >= 0, status
        return (int(a_view[0]) + (int(a_view[1]) << 64) +
             (int(a_view[2]) << 128) + (int(a_view[3]) << 192))

    def dump(self):
        return MMOpFastMatrixDump(self)

    def _slack_size(self):
        cdef uint32_t ssize = 0
        cdef void *p_slack = mm_op_fast_get_slack(self.m, &ssize)
        return 0 if (p_slack == NULL) else ssize

class MMOpFastMatrixDump:
    def __init__(self, matrix):
        assert isinstance(matrix, MMOpFastMatrix)
        cdef MMOpFastMatrix mymatrix = matrix
        cdef mmv_fast_matrix_type *m = mymatrix.m
        self.mode = m.mode
        self.p = m.p 
        self.nrows = m.nrows
        self.current = m.current
        self.v = [None, None]
        cdef mmv_fast_matrix_union_type p_v = m.p_v
        cdef mmv_fast_type *p_fast
        cdef uint8_t[:] a_view
        if m.mode == 1:
             for i in [0, 1]:
                 p_fast = p_v.p_vb[i]
                 if p_fast != NULL:
                     a = np.zeros(MM_FAST_BYTELENGTH, dtype = np.uint8)
                     self.v[i] = a
                     a_view = a
                     memcpy(&a_view[0], p_fast.b, MM_FAST_BYTELENGTH)     
                 else:            
                     self.v[i] = np.zeros(0, dtype = np.uint8)





FastBuffer_IND_ERR = "Index out of range in class FastBuffer"

cdef class FastBuffer:
    cdef uint8_t *ptr
    cdef uint32_t bufsize
    cdef void * p_test
    def __cinit__(self, uint32_t bufsize, *args, **kwds):
        self.bufsize = bufsize
        self.p_test = NULL
        self.ptr = <uint8_t *>mm_op_fast_buffer_alloc(bufsize)
        if self.ptr == NULL:
            self.bufsize = 0
            raise MemoryError("Out of memory for class FastBuffer")

    def  __dealloc__(self):
        mm_op_fast_buffer_free(self.ptr, self.bufsize)
        self.ptr = NULL
        self.bufsize = 0
        mm_op_fast_buffer_test_stop(self.p_test)
        self.p_test = NULL

    def __init__(self, uint32_t index):
        pass

    def __getitem__(self, uint32_t index):
        if index < self.bufsize:
            return self.ptr[index]
        raise IndexError(FastBuffer_IND_ERR)

    def __setitem__(self, uint32_t index, uint32_t value):
        if index < self.bufsize:
            self.ptr[index] = <uint8_t>(value & 0xff)
        else:
            raise IndexError(FastBuffer_IND_ERR)

    def __len__(self):
        return self.bufsize

    def start_test(self):
        """Start a simple allocator test

        This test starts 4 threads in C that cause traffic at the
        allocator by contiuously allocating and dallocating buffers.
        Note that Python usually does not support real mulltithreading,
        so that we have do do it this way. Use method ``stop_test``
        to stop the traffic!
        """
        if self.p_test == NULL:
            self.p_test = mm_op_fast_buffer_test_start()
            if self.p_test == NULL:
                ERR = "Could not start allocator test with threads"
                raise ValueError(ERR)

    def stop_test(self):
        err = mm_op_fast_buffer_test_stop(self.p_test)
        self.p_test = NULL
        assert err == 0, hex(err)

    @classmethod
    def gc(cls):
        """Global garbage collector"""
        mm_op_fast_buffer_gc()

    @classmethod
    def statistics(cls):
        """Display statistics"""
        a = np.zeros(48, dtype = np.int32)
        cdef int32_t[:] pa = a
        cdef int32_t n3 = mm_op_fast_buffer_stat(&pa[0], 48), i, found = 0
        assert n3 > 0
        a = a[:n3].reshape((n3//3, 3))
        print("Buffers managed by mmgroup_fast allocator")
        print("   Size     allocated    free")
        for i in range(n3//3):
            if (a[i,1:] != 0).any():
                print("%7d %13d %7d" % (tuple(a[i])))
                found = 1
        if not found:
            print("   <No buffers present>")


include "mm_op_fast_axis_mod3.pxi"
include "mm_op_fast_g.pxi"



