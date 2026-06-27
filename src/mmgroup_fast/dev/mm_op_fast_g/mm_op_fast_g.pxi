from mmgroup.structures.construct_mm import iter_strings_from_atoms


cdef class MMOpFastG:
    ERRORS = {
        1: "Multiplication in class MMOpFastG failed",
        2: "Copying in class MMOpFastG failed",
        3: "Could not obtain word from MMOpFastG object",
        4: "Could not obtain matrix from MMOpFastG object",
        5: "Could not convert MMOpFastG object to int",
        6: "Could check if MMOpFastG object is neutral element",
    }
    _MAXEXP = 1 << 62
    cdef mm_fast_g_type *ptr

    def __cinit__(self, *args, **kwds):
        self.ptr = <mm_fast_g_type *>fast_g_obj_new()
        if self.ptr == NULL:
            raise MemoryError("Out of memory for class MMOpFastG")

    def  __dealloc__(self):
        fast_g_obj_delete(self.ptr)
        self.ptr = NULL

    def __init__(self, *g):
        if len(g):
             self.mulexp(MM0(*g))
        pass

    def _display_flags(self):
        flags = fast_g_obj_get_flags(self.ptr)
        print("Flags of MMOpFastG object are: 0x%016x" % flags)

    def _chk(self, result, errno):
        if result < 0:
            print("Error in MMOpFastG object, status = %d" % result)
            self._display_flags()
            try:
                err = self.ERRORS[errno]
            except:
                err = "Unknown error in MMOpFastG object"
            raise ValueError(err)
        return result

    def freeze(self):
        fast_g_obj_freeze(self.ptr)
        return self

    def g(self, uint32_t reduced = False):
        cdef uint32_t *my_g
        cdef uint32_t my_len, i
        my_g = fast_g_obj_get_g(self.ptr, reduced,&my_len)
        if  my_g == NULL:
            self._chk(-1, 3)
        arr = np.empty(my_len, dtype=np.uint32)
        memcpy(<void*>arr.data, <void*>my_g, my_len * sizeof(uint32_t))
        return arr

    @property
    def mmdata(self):
        return self.g(1)

    def mat(self):
        cdef mmv_fast_matrix_type *pmat = fast_g_obj_get_mat(self.ptr)
        if pmat == NULL:
            self._chk(-1, 4)
        m = MMOpFastMatrix(3, 4, 1)
        cdef mmv_fast_matrix_type *pc = &m.m
        cdef int32_t status = mm_op_fast_copy_data(pmat, pc)
        assert status >= 0
        return m

    def as_int(self):
        """Warning: not compatible to method as_int of class MM"""
        cdef uint64_t *a
        a = fast_g_obj_as_int_fast(self.ptr)
        if a == NULL:
            self._chk(-1, 5)
        return (int(a[0]) + (int(a[1]) << 64) +
             (int(a[2]) << 128) + (int(a[3]) << 192))

    def neutral(self, uint32_t reduce = True):
        """Fast check if object is the neutral element

        Return 1 if not neutral, 0 if neutral, -1 if unknown.
        if "reduce" is set, the result will always be known,
        otherwise a faster, but less reliable check is made.
        """
        cdef int32_t n = fast_g_obj_nonneutral(self.ptr, reduce)
        if -1 <= n <= 1:
            return n
            self._chk(-1, 6)

    @cython.boundscheck(False)
    def mulexp(self, other, int32_t e = 1):
        cdef int32_t status
        cdef mm_fast_g_type *other_ptr
        cdef uint32_t[:] m_data
        cdef uint32_t *other_g
        cdef uint32_t other_len
        if isinstance(other, MMOpFastG):
            other_ptr = <mm_fast_g_type *>other.ptr
            status = fast_g_obj_mulexp_obj(self.ptr, other_ptr, e)
        else:
            if isinstance(other, AbstractMMGroupWord):
                m_data = other.mmdata
            else:
                m_data = np.array(other, dtype=np.uint32_t)
            other_g = &m_data[0]
            other_len = len(m_data)
            status = fast_g_obj_mulexp(self.ptr, other_g, other_len, e)
            self._chk(status, 1)
        return self


    def copy(self, uint32_t g_only = 0):
        mycopy = MMOpFastG()
        cdef  mm_fast_g_type *myptr = <mm_fast_g_type *>mycopy.ptr
        cdef int32_t status = fast_g_obj_copy(myptr, self.ptr, g_only)
        self._chk(status, 2)
        return mycopy
        fast_g_obj_copy


    def exp(self, e):
        mycopy = MMOpFastG()
        cdef  mm_fast_g_type *myptr = <mm_fast_g_type *>mycopy.ptr
        cdef int32_t status
        if abs(e) < self._MAXEXP:
            status = fast_g_obj_mulexp_obj(myptr, self.ptr, e)
            self._chk(status, 1)
            return mycopy
        eh, el = divmod(e, self._MAXEXP)
        h = self.exp(eh)
        cdef  mm_fast_g_type *hptr = <mm_fast_g_type *>h.ptr
        status = fast_g_obj_mulexp_obj(myptr, hptr, self._MAXEXP)
        self._chk(status, 1)
        status = fast_g_obj_mulexp_obj(myptr, self.ptr, el)
        self._chk(status, 1)
        return mycopy
        


    def __mul__(self, other):
        return self.copy().mulexp(other)

    def __rmul__(self, other):
        mycopy = MMOpFastG()
        mycopy.mulexp(other)
        mycopy.mulexp(self)
        return mycopy

    def conj(self, other):
        inv = MMOpFastG().mulexp(other, -1)
        inv.mulexp(self)
        return inv.mulexp(other)

    def __pwr__(self, other):
        if isinstance(other, Integral):
           if abs(other) <= 4:
               return self.exp(other)
           else:
               q, r = divmod(other, 4)
               return ((self**q).exp(4)).mulexp(self, r)   
        else:
           return self.conj(other)
                
                
    def raw_str_word(self):
        """Convert group atom ``g`` to a string

        For an element ``g`` of this group ``g.group.raw_str_word(g)``
        should be equivalent   to ``g.raw_str()``.
        """
        atoms = self.mmdata
        s = "*".join(iter_strings_from_atoms(atoms, abort_if_error=0))
        return s if s else "1"
                 
    def __str__(self):
        return "MMFG<%s>" % self.raw_str_word()  

    __repr__ = __str__
