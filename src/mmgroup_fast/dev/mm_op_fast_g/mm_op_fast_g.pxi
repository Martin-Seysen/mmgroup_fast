

cdef class MMOpFastG:
    cdef mm_fast_g_type *ptr
    def __cinit__(self, *args, **kwds):
        self.ptr = <mm_fast_g_type *>fast_g_obj_new()
        if self.ptr == NULL:
            raise MemoryError("Out of memory for class class MMOpFastG")

    def  __dealloc__(self):
        fast_g_obj_delete(self.ptr)
        self.ptr = NULL

    def __init__(self):
        pass



