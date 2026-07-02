from mmgroup.structures.construct_mm import iter_strings_from_atoms


cdef class MMOpFastG:
    ERRORS = {
        1: "Multiplication in class MMOpFastG failed",
        2: "Copying in class MMOpFastG failed",
        3: "Could not obtain word from MMOpFastG object",
        4: "Could not obtain matrix from MMOpFastG object",
        5: "Could not convert MMOpFastG object to int",
        6: "Could check if MMOpFastG object is neutral element",
        7: "Internal error in method _augment",
    }
    _MAXEXP = 1 << 62
    FLAG_ERROR = 0x20
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

    def _display_flags(self):
        cdef int64_t status[5]
        fast_g_obj_get_status(self.ptr, status)
        print("Flags of MMOpFastG object are: 0x%016x" % status[0])
        print("  Status: op = %d, err = %d" % (status[1], status[2]))       
        print("  Lengths: g: %d, reduced: %d" % (status[3], status[4]))       

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
        my_g = fast_g_obj_get_g(self.ptr, reduced, &my_len)
        if  my_g == NULL:
            self._chk(-1, 3)
        arr = np.empty(my_len, dtype=np.uint32)
        for i in range(my_len):
            arr.data[i] = my_g[i]
        return arr

    @property
    def mmdata(self):
        return self.g(1)


    @property
    def inverse_mmdata(self):
        mm = self.g(1)
        mm_group_invert_word(mm.data, len(mm));
        return mm


    def _augment(self, int64_t flags, int64_t kill=0):
        flags = fast_g_obj_augment(self.ptr, flags, kill)
        if flags & self.FLAG_ERROR:
            self._display_flags()
            raise ValueError(self.ERRORS[7])
        return flags


    def mat(self):
        cdef mmv_fast_matrix_type *pmat = fast_g_obj_get_mat(self.ptr)
        if pmat == NULL:
            self._chk(-1, 4)
        m = MMOpFastMatrix(3, 4, 1)
        cdef mmv_fast_matrix_type *pc = &m.m
        self._chk(mm_op_fast_copy_data(pmat, pc), 2)
        return m

    def mmv3(self, i):
        return self.mat().row_as_mmv(i)


    def as_int(self):
        """Warning: not compatible to method as_int of class MM"""
        cdef uint64_t *a
        a = fast_g_obj_as_int_fast(self.ptr)
        if a == NULL:
            self._chk(-1, 5)
        return (int(a[0]) + (int(a[1]) << 64) +
             (int(a[2]) << 128) + (int(a[3]) << 192))

    def chk_neutral(self, uint32_t reduce = True):
        """Fast check if object is the neutral element

        Return 1 if not neutral, 0 if neutral, -1 if unknown.
        if "reduce" is set, the result will always be known.
        Otherwise a faster, but less reliable check is made.
        """
        cdef int32_t n = fast_g_obj_nonneutral(self.ptr, reduce)
        if -1 <= n <= 1:
            return n
            self._chk(-1, 6)

    @cython.boundscheck(False)
    def mulexp(self, other, int32_t e = 1):
        cdef int32_t status
        cdef MMOpFastG other_obj
        cdef mm_fast_g_type *other_ptr
        cdef uint32_t[:] m_data
        cdef uint32_t *other_g
        cdef uint32_t other_len
        if isinstance(other, MMOpFastG):
            other_obj = other
            other_ptr = <mm_fast_g_type *>(other_obj.ptr)
            status = fast_g_obj_mulexp_obj(self.ptr, other_ptr, e)
            self._chk(status, 1)
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
        cdef MMOpFastG mycopy = MMOpFastG()
        cdef mm_fast_g_type *myptr = <mm_fast_g_type *>(mycopy.ptr)
        cdef int32_t status = fast_g_obj_copy(myptr, self.ptr, g_only)
        mycopy._chk(status, 2)
        return mycopy
        fast_g_obj_copy


    def exp(self, e):
        cdef MMOpFastG mycopy = MMOpFastG()
        cdef mm_fast_g_type *myptr = <mm_fast_g_type *>(mycopy.ptr)
        cdef int32_t status
        if abs(e) < self._MAXEXP:
            status = fast_g_obj_setpower(myptr, self.ptr, e)
            print("set power, e =", e, ", status=", status)
            mycopy._chk(status, 1)
            return mycopy
        eh, el = divmod(e, self._MAXEXP)
        h = self.exp(eh)
        cdef  mm_fast_g_type *hptr = <mm_fast_g_type *>h.ptr
        status = fast_g_obj_setpower(myptr, hptr, self._MAXEXP)
        mycopy._chk(status, 1)
        status = fast_g_obj_mulexp_obj(myptr, self.ptr, el)
        mycopy._chk(status, 1)
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

    def __pow__(self, other):
        if isinstance(other, Integral):
           return self.exp(other)
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
