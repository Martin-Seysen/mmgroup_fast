// %%GEN h



#ifdef MM_OP_FAST_PERMUTATIONS
/// @cond DO_NOT_DOCUMENT


/****************************************************************************
** Perform 24- and 64-byte XOR operration
****************************************************************************/


static inline void ALWAYS_INLINE
mmv_row24_copy(mmv_fast_row32_type a, mmv_fast_row32_type *r)
{
    // ASSUME_ALIGNED(r, 32);
    #ifdef GCC_VECTORS
        r->v32[0] = a.v32[0];
    #else 
        r->u64[0] = a.u64[0];
        r->u64[1] = a.u64[1];
        r->u64[2] = a.u64[2];
    #endif
}


static inline void ALWAYS_INLINE
mmv_row24_xor(mmv_fast_row32_type a, mmv_fast_row32_type b, mmv_fast_row32_type *r)
{
    // ASSUME_ALIGNED(r, 32);
    #ifdef GCC_VECTORS
        r->v32[0] = a.v32[0] ^ b.v32[0];
    #else 
        r->u64[0] = a.u64[0] ^ b.u64[0];
        r->u64[1] = a.u64[1] ^ b.u64[1];
        r->u64[2] = a.u64[2] ^ b.u64[2];
    #endif
}


static inline void ALWAYS_INLINE
mmv_row24_neg(mmv_fast_row32_type a, mmv_fast_row32_type *r)
{
    // ASSUME_ALIGNED(r, 32);
    #ifdef GCC_VECTORS
        r->v32[0] = a.v32[0] ^ 0xff;
    #else 
        r->u64[0] = ~a.u64[0];
        r->u64[1] = ~a.u64[1];
        r->u64[2] = ~a.u64[2];
    #endif
}


static inline void ALWAYS_INLINE
mmv_row24_and(mmv_fast_row32_type a, mmv_fast_row32_type b, mmv_fast_row32_type *r)
{
    // ASSUME_ALIGNED(r, 32);
    #ifdef GCC_VECTORS
        r->v32[0] = a.v32[0] & b.v32[0];
    #else 
        r->u64[0] = a.u64[0] & b.u64[0];
        r->u64[1] = a.u64[1] & b.u64[1];
        r->u64[2] = a.u64[2] & b.u64[2];
    #endif
}


static inline void ALWAYS_INLINE mmv_extend_row32_row64(
    mmv_fast_row32_type a, mmv_fast_row64_type *r)
{
    // ASSUME_ALIGNED(r, 64);
    #ifdef GCC_VECTORS
      #if defined(__AVX512__) || defined(__AVX512VBMI__)
        // store low 256 bits
        r->m512[0] = _mm512_castsi256_si512(a.m256[0]);
      #else
        r->v32[0] = a.v32[0];
      #endif        
    #else 
        // %%FOR* i in range(4)
        r->u64[%{i}] = a.u64[%{i}];
        // %%END FOR
    #endif
}



static inline void ALWAYS_INLINE mmv_merge_rows32_row64(
    mmv_fast_row32_type lo, mmv_fast_row32_type hi, mmv_fast_row64_type *r)
{
    // ASSUME_ALIGNED(r, 64);
    #ifdef GCC_VECTORS
      #if defined(__AVX512__) || defined(__AVX512VBMI__)
        // store low 256 bits
        r->m512[0] = _mm512_castsi256_si512(lo.m256[0]);
        // insert high 256 bits
        r->m512[0] = _mm512_inserti64x4(r->m512[0], hi.m256[0], 1);
      #else
        r->v32[0] = lo.v32[0];
        r->v32[1] = hi.v32[0];
      #endif        
    #else 
        // %%FOR* i in range(4)
        r->u64[%{i}] = lo.u64[%{i}];
        r->u64[%{int:i+4}] = hi.u64[%{i}];
        // %%END FOR
    #endif
}



static inline void ALWAYS_INLINE mmv_split_row64_rows32(
    mmv_fast_row64_type r, mmv_fast_row32_type *lo, mmv_fast_row32_type *hi)
{
    // ASSUME_ALIGNED(r, 64);
    #ifdef GCC_VECTORS
      #if defined(__AVX512__) || defined(__AVX512VBMI__)
        lo->m256[0] = _mm512_castsi512_si256(r.m512[0]);         // low 256 bits
        hi->m256[0] = _mm512_extracti64x4_epi64(r.m512[0], 1);   // high 256 bits
      #else
        lo->v32[0] = r.v32[0];
        hi->v32[0] = r.v32[1];
      #endif        
    #else 
        // %%FOR* i in range(4)
        lo->u64[%{i}] = r.u64[%{i}];
        hi->u64[%{i}] = r.u64[%{int:i+4}];
        // %%END FOR
    #endif
}




static inline void ALWAYS_INLINE
mmv_row64_copy(mmv_fast_row64_type a, mmv_fast_row64_type *r)
{
    // ASSUME_ALIGNED(r, 64);
    #ifdef GCC_VECTORS
      #if GCC_VECTOR_ALIGNED >= 64
        r->v64[0] = a.v64[0];
      #else
        r->v32[0] = a.v32[0];
        r->v32[1] = a.v32[1];
      #endif        
    #else 
        // %%FOR* i in range(8)
        r->u64[%{i}] = a.u64[%{i}];
        // %%END FOR
    #endif
}




static inline void ALWAYS_INLINE
mmv_row64_xor(mmv_fast_row64_type a, mmv_fast_row64_type b, mmv_fast_row64_type *r)
{
    // ASSUME_ALIGNED(r, 64);
    #ifdef GCC_VECTORS
      #if GCC_VECTOR_ALIGNED >= 64
        r->v64[0] = a.v64[0] ^ b.v64[0];
      #else
        r->v32[0] = a.v32[0] ^ b.v32[0];
        r->v32[1] = a.v32[1] ^ b.v32[1];
      #endif        
    #else 
        // %%FOR* i in range(8)
        r->u64[%{i}] = a.u64[%{i}] ^ b.u64[%{i}];
        // %%END FOR
    #endif
}


static inline void ALWAYS_INLINE
mmv_row64_cond_compl(mmv_fast_row64_type a, uint32_t sign, mmv_fast_row64_type *r)
{
    // ASSUME_ALIGNED(r, 64);   
    #ifdef GCC_VECTORS
      #if GCC_VECTOR_ALIGNED >= 64
        r->v64[0] = a.v64[0] ^ (uint8_t)(-(sign & 1));
      #else
        r->v32[0] = a.v32[0] ^ (uint8_t)(-(sign & 1));
        r->v32[1] = a.v32[1] ^ (uint8_t)(-(sign & 1));
      #endif        
    #else
        uint64_t sign64 = -((uint64_t)sign & 1);
        // %%FOR* i in range(8)
        r->u64[%{i}] = a.u64[%{i}] ^ sign64;
        // %%END FOR
    #endif
}



/****************************************************************************
** Perform 64-byte shuffling
****************************************************************************/


#if defined(GCC_AVX2)
#define BYTEMASK_2_16(x,y) _mm256_setr_epi8( \
   x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, y,y,y,y, y,y,y,y, y,y,y,y, y,y,y,y) 

static inline
__m256i sel_shuf_epi8(__m256i data, __m256i mask, __m256i select)
{
    const __m256i K0 = BYTEMASK_2_16(0x70, 0x70);
    return _mm256_shuffle_epi8(data, 
           _mm256_add_epi8(_mm256_xor_si256(mask, select), K0));
}
#endif


static inline void ALWAYS_INLINE _shuffle64(
    mmv_fast_row64_type data, mmv_fast_row64_type mask, mmv_fast_row64_type *r
)
{
  #if defined(SHUFFLE64)
  #if defined(__AVX512__) || defined(__AVX512VBMI__)
    r->m512[0] = _mm512_permutexvar_epi8(mask.m512[0], data.m512[0]);
  #else
    r->v64[0] = __builtin_shuffle(data.v64[0], mask.v64[0]);
  #endif
  #else
  #if defined(GCC_AVX2)
    __m256i d0, d1, m0, m1, r0, r1;
    const __m256i SEL01 = BYTEMASK_2_16(0x00, 0x10);
    const __m256i SEL10 = BYTEMASK_2_16(0x10, 0x00); 
    const __m256i SEL23 = BYTEMASK_2_16(0x20, 0x30);
    const __m256i SEL32 = BYTEMASK_2_16(0x30, 0x20);
    d0 = data.m256[0];
    d1 = data.m256[1];
    m0 = mask.m256[0];
    m1 = mask.m256[1];
    r0 = sel_shuf_epi8(d0, m0, SEL01);
    r0 =  _mm256_or_si256(r0, sel_shuf_epi8(d1, m0, SEL23));
    r1 = sel_shuf_epi8(d0, m1, SEL01);
    r1 =  _mm256_or_si256(r1, sel_shuf_epi8(d1, m1, SEL23));
    d0 = _mm256_permute4x64_epi64(d0, 0x4E);
    d1 = _mm256_permute4x64_epi64(d1, 0x4E);
    r0 = _mm256_or_si256(r0, sel_shuf_epi8(d0, m0, SEL10));
    r0 = _mm256_or_si256(r0, sel_shuf_epi8(d1, m0, SEL32));
    r1 = _mm256_or_si256(r1, sel_shuf_epi8(d0, m1, SEL10));
    r1 = _mm256_or_si256(r1, sel_shuf_epi8(d1, m1, SEL32));
    r->m256[0] = r0;
    r->m256[1] = r1;
  #else
    // %%FOR* i in range(64)
    r->b[%{i}] = data.b[mask.b[%{i}]];
    // %%END FOR
  #endif
  #endif
}



/****************************************************************************
** Perform 24-bit shuffling
****************************************************************************/



static inline void ALWAYS_INLINE _shuffle24(
   mmv_fast_row32_type data, mmv_fast_shuffle24_type mask, uint32_t sign,
   mmv_fast_row32_type *r
)
{
   //  ASSUME_ALIGNED(r, 64);
      
   #if MM_OP_FAST_SPLIT_SHUFFLE24 == 1
     uint8_t bsign = (uint8_t)(0 - (sign & 1));
     #ifdef GCC_AVX2
       r->m256[0] = _mm256_shuffle_epi8(data.m256[0], mask.m256[0]);
       r->m256[0] = _mm256_permute4x64_epi64(r->m256[0], 0xC6);
       r->m256[0] = _mm256_shuffle_epi8(r->m256[0], mask.m256[1]);
       r->m256[0] = _mm256_permute4x64_epi64(r->m256[0], 0xC6);
       r->m256[0] = _mm256_shuffle_epi8(r->m256[0], mask.m256[2]);
     #else
       static v16_8_type fmask = {0xff, 0xff, 0xff, 0xff, 
       0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0};   
       v16_8_type r0, r1, t;
       r0 = __builtin_shuffle(data.v16[0], mask.v16[0]);
       r1 = data.v16[1];
       t = (r0 ^ r1) & fmask;
       r0 ^= t; r1 ^= t;
       r0 = __builtin_shuffle(r0, mask.v16[2]);
       t = (r0 ^ r1) & fmask;
       r->v16[1] = r1 ^ t;
       r0 ^= t;
       r0 = __builtin_shuffle(r0, mask.v16[4]);
       r->v16[0] = r0;
     #endif
     r->v32[0] ^= bsign;
   #else
     #ifdef SHUFFLE64
       uint8_t bsign = (uint8_t)(0 - (sign & 1));
       #ifdef GCC_AVX2
          r->m256[0] = _mm256_permutexvar_epi8(mask.m256[0], data.m256[0]);
       #else
          r->v32[0] = __builtin_shuffle(data.v32[0], mask.v32[0]);
       #endif
       r->v32[0] ^= bsign;
     #else
       uint64_t bsign = (uint64_t)(0ULL - ((uint64_t)sign & 1ULL));
        // %%FOR* j in range(24)
          r->b[%{j}] = data.b[mask.b[%{j}]];
        // %%END FOR
        // %%FOR* j in range(3)
          r->u64[%{j}] ^= bsign;
        // %%END FOR
     #endif
   #endif
}



/****************************************************************************
** Compute permutation of columns in a row for tag T
****************************************************************************/


#if defined(GCC_VECTORS) && (defined(SHUFFLE64) || defined (GCC_AVX2))
#define MM_OP_FAST_PERM_ROW64_SHUFFLE64
#endif




static inline mmv_fast_shuffle24_type 
_load_op_pi_row64_perm24(mmv_fast_op_pi_type *p_op_pi)
{
  #ifdef MM_OP_FAST_PERM_ROW64_SHUFFLE64
     return p_op_pi->perm24;
  #else
     mmv_fast_shuffle24_type result;
     memcpy(&result, &(p_op_pi->inv_perm), 32);
     return result;
  #endif
}


typedef struct ALIGNED(32) {
    mmv_fast_row32_type pi24;
    uint64_t sign;
    uint32_t src;
    uint32_t row;
} mmv_fast_op_pi_row64_in_type;


static inline ALWAYS_INLINE void _prep_op_pi_row64_in(
  uint32_t row, uint32_t eps, uint16_t *p_autpl, mmv_fast_op_pi_row64_in_type *r
)
{
    ASSUME_ALIGNED(r, 32);
    uint_fast32_t src, dest;

    r->row = row;
    dest = mat24_def_octad_to_gcode(row);
    src = p_autpl[dest & 0x7ff];
    r->sign = ((dest & eps) >> 11) ^ (src >> 12);
    r->sign = 0ULL - (r->sign & 1ULL);
    src &= 0xfff;
    r->src = mat24_def_gcode_to_octad(src);
    r->pi24 = MM_FAST_OCTAD_EXPAND[r->src]; 
}


#if defined(GCC_AVX2)


static inline __m256i shuf_epi8_lc(__m256i value, __m256i shuffle){
/* Ermlg's lane crossing byte shuffle https://stackoverflow.com/a/30669632/2439725 */
const __m256i K0 = _mm256_setr_epi8(
    0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0);
const __m256i K1 = _mm256_setr_epi8(
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70);
return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle, K0)), 
    _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E), _mm256_add_epi8(shuffle, K1)));
}




static inline __m256i shuf_epi8_hi(__m256i value, __m256i shuffle){
// Variant of  shuf_epi8_lc: 
// extract bytes from  value at positions shuffle[24..31]
// and store them at postions [0..7] and [16..23], don't care for other positions
const __m256i K0 = _mm256_setr_epi8(
    0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0, 0, 0, 0, 0, 0, 0, 0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0, 0, 0, 0, 0, 0, 0, 0);

__m256i tmp = _mm256_shuffle_epi8(
    value, _mm256_add_epi8(_mm256_permute4x64_epi64(shuffle, 0x33), K0));
return _mm256_or_si256(_mm256_permute4x64_epi64(tmp, 0x4E), tmp);
}



#endif


static inline void ALWAYS_INLINE
_prep_op_pi_row64(
    mmv_fast_op_pi_row64_in_type *r_in,
    mmv_fast_shuffle24_type perm,
    mmv_fast_row64_type *v_in,
    mmv_fast_row64_type *r_out
    )
{
    ASSUME_ALIGNED(r_in, 32);
    ASSUME_ALIGNED(v_in, 64);
    ASSUME_ALIGNED(r_out, 64);

  #if defined(MM_OP_FAST_PERM_ROW64_SHUFFLE64)
    #ifdef SHUFFLE64
      static mmv_fast_row64_type CS[] = {
        {{24,25,24,25, 24,27,24,27, 24,29,24,29,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}}, 
        {{24,24,26,26, 24,24,28,28, 24,24,30,30,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 
           0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}}, 
        {{ 0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,
           0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,
           0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,
           0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3}},
        {{ 4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
           4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
           4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7,
           4, 4, 4, 4,  5, 5, 5, 5,  6, 6, 6, 6,  7, 7, 7, 7}},
        {{ 8, 8, 8, 8,  8, 8, 8, 8,  8, 8, 8, 8,  8, 8, 8, 8,
           9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,  9, 9, 9, 9,
          10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10,
          11,11,11,11, 11,11,11,11, 11,11,11,11, 11,11,11,11}},
      };
      mmv_fast_row64_type ALIGNED(64) pi_perm, ALIGNED(64) pi;
      mmv_fast_row64_type ALIGNED(64) pi1, ALIGNED(64) pi2;
      mmv_fast_row32_type ALIGNED(32) t;

      mmv_extend_row32_row64(t = perm.v32row, &pi_perm);
      mmv_extend_row32_row64(t = r_in->pi24, &pi1);
      _shuffle64(pi1, pi_perm, &pi1);
      mmv_extend_row32_row64(t = MM_FAST_OCTAD_EXPAND[r_in->row], &pi_perm);
      _shuffle64(pi1, pi_perm, &pi1);
      _shuffle64(pi1, CS[0], &pi2);
      _shuffle64(pi1, CS[1], &pi1);
      pi1.v64[0] ^= pi2.v64[0];
      _shuffle64(pi1, CS[2], &pi);
      _shuffle64(pi1, CS[4], &pi2);
      _shuffle64(pi1, CS[3], &pi1);
      pi.v64[0] ^= pi1.v64[0] ^ pi2.v64[0];
      mmv_row64_copy(v_in[r_in->src], r_out);
      _shuffle64(r_out[0], pi, &r_out[0]);
      r_out[0].v64[0] ^= (uint8_t)(r_in->sign);
    #else
      #ifdef GCC_AVX2
        static mmv_fast_row32_type CS[5] = {
          {{0,1,0,1,0,3,0,3, 6,0,0,0,0,0,0,0, 0,1,0,1,0,3,0,3, 5,0,0,0,0,0,0,0}},
          {{0,0,2,2,0,0,4,4, 0,0,0,0,0,0,0,0, 0,0,2,2,0,0,4,4, 0,0,0,0,0,0,0,0}},
          {{0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3, 0,1,2,3,0,1,2,3}},
          {{4,4,4,4,5,5,5,5, 6,6,6,6,7,7,7,7, 4,4,4,4,5,5,5,5, 6,6,6,6,7,7,7,7}},
          {{0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8}},    
        };

        mmv_fast_row32_type pi_perm, pi24, pi1, pi2;
        mmv_fast_row64_type pi;
        uint8_t r;
        pi24 = r_in->pi24;
        pi24.m256[0] = _mm256_shuffle_epi8(pi24.m256[0], perm.m256[0]);
        pi24.m256[0] = _mm256_permute4x64_epi64(pi24.m256[0], 0xC6);
        pi24.m256[0] = _mm256_shuffle_epi8(pi24.m256[0], perm.m256[1]);
        pi24.m256[0] = _mm256_permute4x64_epi64(pi24.m256[0], 0xC6);
        pi24.m256[0] = _mm256_shuffle_epi8(pi24.m256[0], perm.m256[2]);
 
        pi_perm = MM_FAST_OCTAD_EXPAND[r_in->row];
        pi24.m256[0] = shuf_epi8_hi(pi24.m256[0], pi_perm.m256[0]);
        pi1.m256[0] = _mm256_shuffle_epi8(pi24.m256[0], CS[0].m256[0])
                    ^ _mm256_shuffle_epi8(pi24.m256[0], CS[1].m256[0]);
        r = pi1.b[8];
        pi.m256[0] = _mm256_shuffle_epi8(pi1.m256[0], CS[2].m256[0]);
        pi2.m256[0] = _mm256_shuffle_epi8(pi1.m256[0], CS[4].m256[0]);
        pi1.m256[0] = _mm256_shuffle_epi8(pi1.m256[0], CS[3].m256[0]);
        pi.v32[0] ^= pi1.v32[0] ^ pi2.v32[0];
        pi.v32[1] = pi.v32[0] ^ r;  
        _shuffle64(v_in[r_in->src], pi, r_out);
        r_out[0].v32[0] ^= (uint8_t)(r_in->sign);
        r_out[0].v32[1] ^= (uint8_t)(r_in->sign);
      #else
        #error "Function mmv_fast_row64_type is not implemented"
      #endif
    #endif
  #else
    const uint8_t *p_pi_out;
    uint_fast8_t acc, b0, b1, b2, b3, b4, b5;
    uint8_t *pv_in;
    pv_in = &(v_in[r_in->src].b[0]);
    p_pi_out = MAT24_OCTAD_ELEMENT_TABLE + (r_in->row << 3);
    acc = r_in->pi24.b[perm.b[p_pi_out[0]]];
    b0 = r_in->pi24.b[perm.b[p_pi_out[1]]];
    b1 = r_in->pi24.b[perm.b[p_pi_out[2]]];
    b2 = r_in->pi24.b[perm.b[p_pi_out[3]]];
    b3 = r_in->pi24.b[perm.b[p_pi_out[4]]];
    b4 = r_in->pi24.b[perm.b[p_pi_out[5]]];
    b5 = r_in->pi24.b[perm.b[p_pi_out[6]]];
    b1 ^= b0; b0 ^= acc;
    b2 ^= b1; b3 ^= b2; b2 ^= acc;
    b4 ^= b3; b5 ^= b4; b4 ^= acc;
    r_out[0].b[0] = pv_in[0];
    acc = b0;
    r_out[0].b[1] = pv_in[acc];
    // %%FOR* i in range(2, 64)
    acc ^= b%{lsbit:i};
    r_out[0].b[%{i}] = pv_in[acc];
    // %%END FOR
    // %%FOR* i in range(8)
    r_out[0].u64[%{i}] ^= r_in->sign;
    // %%END FOR
  #endif  
}








static inline  mm_fast_sub_op_pi64_type 
_prep_op_pi_row64_index(
    mmv_fast_op_pi_row64_in_type r_in,
    mmv_fast_row32_type perm
    )
{
    mm_fast_sub_op_pi64_type r;

    const uint8_t *p_pi_out;
    uint_fast8_t acc, b0, b1, b2, b3, b4, b5;
    r.preimage = r_in.src + ((r_in.sign & 1) << 12);
    p_pi_out = MAT24_OCTAD_ELEMENT_TABLE + (r_in.row << 3);
    acc = r_in.pi24.b[perm.b[p_pi_out[0]]];
    b0 = r_in.pi24.b[perm.b[p_pi_out[1]]];
    b1 = r_in.pi24.b[perm.b[p_pi_out[2]]];
    b2 = r_in.pi24.b[perm.b[p_pi_out[3]]];
    b3 = r_in.pi24.b[perm.b[p_pi_out[4]]];
    b4 = r_in.pi24.b[perm.b[p_pi_out[5]]];
    b5 = r_in.pi24.b[perm.b[p_pi_out[6]]];
    b1 ^= b0; b0 ^= acc;
    b2 ^= b1; b3 ^= b2; b2 ^= acc;
    b4 ^= b3; b5 ^= b4; b4 ^= acc;
    // %%FOR* i in range(6)
    r.perm[%{i}] = b%{i};
    // %%END FOR
    return r;
}





/// @endcond
#endif // ifdef MM_OP_FAST_PERMUTATIONS

