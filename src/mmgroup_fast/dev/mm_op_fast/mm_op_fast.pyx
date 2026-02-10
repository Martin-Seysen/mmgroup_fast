
# cython: language_level=3

from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


#cimport mm_op_fast as op_fast

from collections.abc import Iterable

import numpy as np
from libc.string cimport memcpy 
from mm_op_fast cimport mm_op_fast_init,  mm_op_fast_dealloc
from mm_op_fast cimport mm_op_fast_normalize
from mm_op_fast cimport mm_op_fast_to_mmv, mm_op_fast_from_mmv
from mm_op_fast cimport mm_op_fast_word, mm_op_fast_raw_vb_data
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
from mm_op_fast cimport mm_op_fast_copy_data

from mmgroup import MMVector

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


cdef class MMOpFastMatrix:
    cdef mmv_fast_matrix_type m 
   
    @staticmethod
    def _complain(res, method):
        if (res < 0):
            err = "Internal error %s in class MMOpFastArray, method %s"
            raise ValueError(err % (hex(res), method))  

    def __cinit__(self, *args, **kwds):
        mm_op_fast_init(&self.m, 0, 0, 0)

    def  __dealloc__(self):
        mm_op_fast_dealloc(&self.m, 0)

    def __init__(self, uint32_t p, uint32_t nrows, uint32_t mode = 0):
        if not p in MAX_NROWS:
            raise ValueError("Bad modulus %s for class MMOpFastArray" % p) 
        if mode == 0:
            mode = 1 if  nrows <= MAX_NROWS[p] else 2
        if mm_op_fast_init(&self.m, p, nrows, mode) != 0:
             raise ValueError("Too many rows or bad modulus for class MMOpFastArray") 

    def copy(self):
        cp = MMOpFastMatrix(self.m.p, self.m.nrows, self.m.mode)
        cdef mmv_fast_matrix_type *pc = &cp.m
        cdef int32_t status = mm_op_fast_copy_data(&self.m, pc)
        assert status >= 0
        return cp

    def normalize(self, normalize_data = 0):
        cdef int32_t status = mm_op_fast_normalize(&self.m, normalize_data)
        assert status >= 0
              
    def set_row(self, uint32_t i, row):
        if i >= self.m.nrows:
            raise IndexError("Row index out of range in class MMOpFastArray")
        cdef uint_mmv_t[:] row_view
        cdef int32_t status
        if isinstance(row, MMVector):
            if row.p == self.m.p:
                row_view = row.data
                status = mm_op_fast_from_mmv(&self.m, i, &row_view[0])
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
        status = mm_op_fast_to_mmv(&self.m, i, &row_view[0], len(v.data))
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
        cdef uint32_t[:] g_data = g.mmdata
        cdef int32_t status
        status = mm_op_fast_word(&self.m, &g_data[0], len(g_data), e)
        if status >= 0:
            #status =  mm_op_fast_dealloc(&self.m, 1)
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
        cdef uint32_t[:] g_data = g.mmdata
        cdef uint32_t status = 0, i
        t = time.time()
        for i in range(n):
             status |= mm_op_fast_word(
                 &self.m, &g_data[0], len(g_data), e) < 0
        t =  time.time() - t
        if status:
            err = "Error in class MMOpFastArray, method mul_exp_bench"
            raise ValueError(err) 
        return t 

    def num_entries_A_t(self, uint32_t row):
        assert 0 <= row < 4
        cdef int32_t res = mm_axis3_fast_num_entries_A_t(&self.m, row)
        assert res >= 0, res
        return res & 0xffff, res >> 16

    def find_exp_t(self, uint32_t row, ax_type):
        if isinstance(ax_type, str):
            ax_type = ORBIT_DICT[ax_type] 
        cdef uint32_t ax_t = ax_type 
        cdef t = mm_axis3_fast_find_exp_t(&self.m, row, ax_t)
        assert t >= 0, (ax_t, t)
        return t

    def get_data(self, data):
        a = np.array(data, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef uint32_t la = len(a)
        cdef int32_t status
        if la:
            status = mm_op_fast_mode1_get(&self.m, &r[0], la)
            assert status >= 0
        return a

    def put_data(self, data):
        a = np.array(data, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef uint32_t la = len(a)
        cdef int32_t status
        if la:
            status = mm_op_fast_mode1_put(&self.m, &r[0], la)
            assert status >= 0

    def zero_data(self):
        mm_op_fast_mode1_zero(&self.m)

    def _poke(self, uint32_t index, uint32_t value):
        mm_op_fast_mode1_poke(&self.m, index, value)

    def reduce_axes(self):
        a = np.zeros(128, dtype = np.uint32)
        cdef uint32_t[:] r = a
        cdef int32_t status
        status = mm_axis3_fast_reduce_axes(&self.m, &r[0], 128)
        assert status >= 0, status
        return a[:status]    

    def dump(self):
        return MMOpFastMatrixDump(self)




class MMOpFastMatrixDump:
    def __init__(self, matrix):
        assert isinstance(matrix, MMOpFastMatrix)
        cdef MMOpFastMatrix mymatrix = matrix
        cdef mmv_fast_matrix_type m = mymatrix.m
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





cdef _store_mmv_fast_Amod3_to_array(mmv_fast_Amod3_type *source):
    cdef int i, j
    cdef uint8_t *pm = &((source.a[0].b)[0])
    cdef uint8_t[:] m =  <uint8_t[:24*32]> pm
    a = np.zeros((24,24), dtype = np.uint8)
    cdef uint8_t[:,:] pa = a
    for i in range(24):
        for j in range(24):
            pa[i,j] = m[(i << 5) + j]
    return a


cdef class MMOpFastAmod3:
    cdef mmv_fast_Amod3_type a
    cdef object _source

    @staticmethod
    def _complain(res, method):
        if (res < 0):
            err = "Internal error %s in class MMOpFastAmod3, method %s"
            raise ValueError(err % (hex(res), method))  


    def __init__(self, source, row=-1, sub_row=-1):
        cdef int32_t status = -1
        cdef int32_t row_
        cdef mmv_fast_matrix_type *pmatrix
        cdef MMOpFastMatrix src
        cdef MMOpFastAmod3 src3
        cdef uint32_t p
        cdef uint8_t[:] m_src
        self._source = None
        #print("mmv_fast_Amod3_type object at address", hex(<size_t>(&self.a))) 
        if isinstance(source, MMOpFastMatrix):
            src = source
            pmatrix = &(src.m)
            row_ = row
            status = mm_axis3_fast_load(pmatrix, row_, &self.a)
            if status >= 0:
                 self._source = source
        elif isinstance(source, MMOpFastAmod3):
            src3 = source
            mm_axis3_fast_copy(&src3.a, &self.a)
            self._source = src3._source
            status = 0
        elif isinstance(source, str) and source == 'r':
            a = np.random.randint(0, 5, size=576, dtype=np.uint8) 
            b = np.where(a==3, a, a % 3)          
            m_src = b
            status = mm_axis3_fast_load_a(&m_src[0], &self.a)
        elif isinstance(source, Iterable):
            a = np.array(source, dtype = np.int64) % 3
            b = np.array(a, dtype = np.uint8)[:24,:24]
            if b.shape == (24,24):
                c = b.ravel()
                m_src = c
                status = mm_axis3_fast_load_a(&m_src[0], &self.a)
            else:
                status = -11
        else:
            from mmgroup import MMVector
            if isinstance(source, MMVector):
                p = source.p
                if p % 3 == 0:
                    a = (source['A'] % 3).ravel()
                    m_src = a
                    status = mm_axis3_fast_load_a(&m_src[0], &self.a) 
                else:
                    status = -9
            else:
                from mmgroup.axes import Axis
                if isinstance(source, Axis):
                     a = (source.v15['A'] % 3).ravel()
                     m_src = a
                     status = mm_axis3_fast_load_a(&m_src[0], &self.a) 
                else:    
                     status = -10
        if status < 0:
             E = "Could not contruct MMOpFastAmod3 from %s object, status = %d"
             raise ValueError(E % (type(source), status)) 
        if sub_row >= 0:
             self.load_sub_row(sub_row)

    def load_sub_row(self, sub_row):
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int32_t status = mm_axis3_fast_load_sub_row(pa, sub_row)
        if status < 0:
            M = "Could not set subrow in MMOpFastAmod3 object"
            raise ValueError(M) 

    cpdef raw_echelon(self, uint32_t diag = 0):
        cdef mmv_fast_Amod3_type *pa = &self.a
        mm_axis3_fast_echelon(pa, diag)
        return self.dim_img, self.raw_data[1]

    def ker_img(self, uint32_t mode_B = 0):
        """This function has not been tested!
        """
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int32_t status = mm_axis3_fast_intersect(pa, mode_B)
        if status < 0:
            self._complain(status, "ker_img")
        b = self.data[1]
        lb0, lb1 = self.len_B
        return b[:lb0], b[lb0:lb0+lb1]

    @property
    def raw_data_flat(self):
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef uint8_t *pdata = mm_axis3_fast_data_ptr(pa)
        cdef uint8_t[:] src = <uint8_t[:2*24*32]> pdata
        return np.array(src, dtype = np.uint8) 

    @property
    def raw_data_flat_uint32(self):
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef uint8_t *pdata = mm_axis3_fast_data_ptr(pa)
        cdef uint32_t *pdata32 = <uint32_t*> pdata
        cdef uint32_t[:] src = <uint32_t[:2*24*8]> pdata32
        return np.array(src, dtype = np.uint32) 

    @property
    def raw_data(self):
        return self.raw_data_flat.reshape((2,24,32))[:,:,:24]

    @property
    def data(self):
        return (self.raw_data & 3) % 3

    @property
    def norm(self):
        return self.a.norm

    @property
    def diag(self):
        return self.a.diag

    @property
    def dim_img(self):
        return self.a.dim_img

    @property
    def len_B(self):
        return self.a.len_B[0], self.a.len_B[1]

    @property
    def mode_B(self):
        return self.a.mode_B

    @property
    def source(self):
        if self._source is not None and self.a.row_source >= 0:
            return self._source, self.a.row_source
        raise ValueError("MMOpFastAmod3 has no source matrix")

    def leech3vector(self, uint32_t matrix, uint32_t row):
        """Return row vector in *Leech lattice mod 3* encoding

        For an object ``a`` of this class the method returns the
        row vector ``a.raw_data[matrix, row]`` as an integer in
        *Leech lattice mod 3* encoding.
        """
        assert 0 <= row < 24
        assert 0 <= matrix < 2
        cdef mmv_fast_Amod3_type *pa = &self.a
        return mm_axis3_fast_to_leech_mod3(pa, 24*matrix+row)

    def rand_short_nonzero(self, uint32_t scalar):
        r"""Return short Leech lattice vector with nonzero entry

        If a source vector in the Griess algebra is stored in this
        object then we return a random vector ``v`` of the Leech
        lattice modulo 2 such that the corresponding entry in the
        source vector in nonzero (mod 3).
        Parameter ``scalar`` is interpreted as follows:

        Bits 23...0: a vector ``w`` in the Leech lattice modulo 2

        Bit 24: an element ``k`` of ``GF(2)``

        Bit 25: an boolan value ``small``

        Then we return only vectors ``v`` such that the scalar
        product ``<v, w>`` is equal to ``k``

        If  ``small`` is set then we always return a vector in the
        Leech lattice modulo 2 corresponding to a Golay cocode vector
        (plus, possibly, the standard frame :math:`\Omega`) instead.

        This function has not yet been tested!!! 
        """
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int32_t v = mm_axis3_fast_rand_short_nonzero(pa, scalar)
        if v <= 0:
            raise ValueError("No suitable short random vector found")
        return v

    def num_entries_BC(self):
        cdef mmv_fast_Amod3_type *pa = &self.a
        return mm_axis3_fast_num_entries_BC(pa)

    def rand_v(self, uint32_t dim, uint32_t n, uint32_t adv = 0):
        """Return random vector in a subspace
 
        The function returns a random vector in the subpace spanned
        by ``self.data[1][:dim]``. This modifies the basis of that
        subspace. The returned vector has norm ``n`` (mod 3).

        The function performs ``adv`` random steps before returning
        a vector. The function returns a random vector in
        *Leech lattice mod 3 encoding* or a negative value in case of
        error. In the special case ``n == 4`` the function selects a
        random vector that is of type 4 also in the Leech lattice
        mod 2, and returns that vector as a vector in the Leech
        lattice mod 2 in *Leech lattice encoding*.
        """
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int64_t v = mm_axis3_fast_rand_v(pa, dim, n, adv)
        assert v >= 0, v
        return v

    def analyze_v4(self, uint32_t v2 = 0):
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef uint32_t l_buf = 200
        buf = np.zeros(l_buf, dtype = np.uint32)
        cdef uint32_t[:] pb = buf
        cdef int64_t lb = mm_axis3_fast_analyze_v4(pa, v2, &pb[0], l_buf)
        assert lb >= 0, lb
        return buf[:lb]
        
    def find_v4(self, uint32_t v2 = 0, ignore_error = 0):
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int32_t status = mm_axis3_fast_find_v4(pa, v2)
        if ignore_error:
            return status
        assert status >= 0, status
        vtype, v = divmod(status, 0x2000000)
        try:
            return ORBIT_TYPES[vtype], v
        except:
            ERR = "Method find_v4() returned with status %d = %s"
            raise ValueError(ERR % (status, hex(status)))
        
    def op_Gx0(self, g, uint32_t no_copy = False):
        """Multiply matrix with group element ``g``

        Here ``g`` must be an element of the group :math:`G_{x0}`.
        The function copies the member ``matrix[0]`` to member
        ``matrix[1]`` and multiplies ``matrix[1]`` with ``g``.

        If ``no_copy`` is ``True`` then ``matrix[1]`` is multiplied
        with ``g`` without previously copying it from ``matrix[1]``.
        """
        if not isinstance(g, np.ndarray):
            g = g.mmdata 
        cdef uint32_t[:] pg = g
        cdef uint32_t lg = len(g)
        cdef uint32_t *ppg = &pg[0] if lg > 0 else NULL
        cdef mmv_fast_Amod3_type *pa = &self.a
        cdef int32_t status
        status = mm_axis3_fast_op_G_x0(pa, ppg, lg, no_copy)
        assert status >= 0, status
        return self
       
