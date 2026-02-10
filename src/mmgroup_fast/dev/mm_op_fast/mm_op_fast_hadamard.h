// %%GEN h


#ifdef MM_OP_FAST_HADAMARD
/// @cond DO_NOT_DOCUMENT



#include "mmgroup_endianess.h"

/****************************************************************************
** Compute Hadamard matrix mod 3
****************************************************************************/


// Compute c = a + b (mod 3) Here a, b, c are unsingend integers or gcc
// builtin vectors, representing a vector of numbers mod 3, with each
// number stored in a pair of adjacent bits. m must be a mask where each
// high bit is set and each low bit is cleared. 
// c must not overlap with a or b. 
#define add_mod_3(c, a, b, m) \
  c = a & b; \
  c |= (c << 1) & (a ^ b); \
  c &= m; \
  c = a + b - c - (c >> 1);


#define exch_u64_1_byte_pairs(a) \
  (((a >> 8) & 0xff00ff00ff00ffULL) | ((a & 0xff00ff00ff00ffULL) << 8))

#define exch_u64_2_byte_pairs(a) \
  (((a >> 16) & 0xffff0000ffffULL) | ((a & 0xffff0000ffffULL) << 16))

#define exch_u64_4_byte_pairs(a) \
  ((a >> 32) | (a << 32))

#define add_mod_3_u64(c, a, b) add_mod_3(c, a, b, 0xaaaaaaaaaaaaaaaaULL)

#if ENDIANESS == 0
#define LO_BYTE_MASK_1_U64 0x00ff00ff00ff00ffULL
#define HI_BYTE_MASK_1_U64 0xff00ff00ff00ff00ULL
#define LO_BYTE_MASK_2_U64 0x0000ffff0000ffffULL
#define HI_BYTE_MASK_2_U64 0xffff0000ffff0000ULL
#define LO_BYTE_MASK_4_U64 0x00000000ffffffffULL
#define HI_BYTE_MASK_4_U64 0xffffffff00000000ULL
#define PARITY_1_MASK_U64  0xff0000ff00ffff00ULL
#define PARITY_0_MASK_U64  0x00ffff00ff0000ffULL
#endif


#if ENDIANESS == 1
#define HI_BYTE_MASK_1_U64 0x00ff00ff00ff00ffULL
#define LO_BYTE_MASK_1_U64 0xff00ff00ff00ff00ULL
#define HI_BYTE_MASK_2_U64 0x0000ffff0000ffffULL
#define LO_BYTE_MASK_2_U64 0xffff0000ffff0000ULL
#define HI_BYTE_MASK_4_U64 0x00000000ffffffffULL
#define LO_BYTE_MASK_4_U64 0xffffffff00000000ULL
#define PARITY_0_MASK_U64  0xff0000ff00ffff00ULL
#define PARITY_1_MASK_U64  0x00ffff00ff0000ffULL
#endif


ALWAYS_INLINE static inline void hadamard_row64_neg_parities(
    mmv_fast_row64_type * restrict a, uint32_t exp1)
{
  #define F 0xff
  static mmv_fast_row64_type ALIGNED(64) PAR[2] = {
     {{0}},
     {
     // %%PERMUTE_64_PARITY_TABLE(128, 256)
     }
  };
  #undef F 
  mmv_row64_xor(*a, PAR[exp1 & 1], a);
}


ALWAYS_INLINE static inline void hadamard_row64_add_mod3(
    mmv_fast_row64_type a, mmv_fast_row64_type b, mmv_fast_row64_type *r)
{
  #if GCC_VECTOR_ALIGNED >= 64
    static ALIGNED(64) const v16_32_type M = {
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL
    };
    ALIGNED(64) v16_32_type c;
    add_mod_3(c, a.v16_32[0], b.v16_32[0], M);
    r->v16_32[0] = c;
  #elif GCC_VECTOR_ALIGNED >= 32
    static ALIGNED(32) const v8_32_type M = {
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL
    };
    ALIGNED(32) v8_32_type c;
    // %%FOR* i in range(2)
    add_mod_3(c, a.v8_32[%{i}], b.v8_32[%{i}], M);
    r->v8_32[%{i}] = c;
    // %%END FOR
  #else
    uint64_t c;
    uint32_t i;
    for (i = 0; i < 8; ++i) {
      add_mod_3(c, a.u64[i], b.u64[i], 0xaaaaaaaaaaaaaaaaULL);
      r->u64[i] = c;
    }
  #endif
}



ALWAYS_INLINE static inline void hadamard_row32_add_mod3(
    mmv_fast_row32_type a, mmv_fast_row32_type b, mmv_fast_row32_type *r)
{
  #if GCC_VECTOR_ALIGNED >= 32
    static ALIGNED(32) const v8_32_type M = {
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,
      0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL,0xaaaaaaaaUL
    };
    ALIGNED(32) v8_32_type c;
    add_mod_3(c, a.v8_32[0], b.v8_32[0], M);
    r->v8_32[0] = c;
  #else
    uint64_t c;
    uint32_t i;
    for (i = 0; i < 4; ++i) {
      add_mod_3(c, a.u64[i], b.u64[i], 0xaaaaaaaaaaaaaaaaULL);
      r->u64[i] = c;
    }
  #endif
}






/// @endcond
#endif // ifdef MM_OP_FAST_HADAMARD

