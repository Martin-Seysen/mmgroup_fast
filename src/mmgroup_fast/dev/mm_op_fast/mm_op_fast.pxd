
from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libc.stdint cimport int64_t, int32_t, int8_t
from libc.stdint cimport uint64_t as uint_mmv_t



cdef extern from "mm_op_fast_types.h":
    """

"""

    enum: MM_FAST_BYTELENGTH

    ctypedef uint8_t v64_8_type[64];
    ctypedef uint8_t v32_8_type[32];

    ctypedef union mmv_fast_row64_type:
        uint8_t b[64];
        # v32_8_type v32[1];

    ctypedef union mmv_fast_row32_type:
        uint8_t b[64];
        # v64_8_type v64[1];

    ctypedef union mmv_fast_type:
        uint8_t b[MM_FAST_BYTELENGTH];
        # v64_8_type v64[MM_FAST_BYTELENGTH_BY_64];

    ctypedef union mmv_fast_matrix_union_type:
        mmv_fast_type* p_vb[2];

    ctypedef struct mmv_fast_matrix_type:
        mmv_fast_matrix_union_type p_v;
        uint32_t p;
        uint32_t nrows;
        uint32_t mode;
        uint32_t check_underflow;
        uint32_t current;



    ctypedef struct mmv_fast_Amod3_type:
        mmv_fast_row32_type *a;
        mmv_fast_matrix_type *p_source;
        int32_t row_source;
        int32_t norm;
        int32_t diag;
        int32_t dim_img;
        int32_t mode_B;
        int32_t len_B[2];
        
        




