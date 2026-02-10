#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include <stddef.h>


#ifndef MM_OP_FAST_TYPES_H_INCLUDED
#define MM_OP_FAST_TYPES_H_INCLUDED


// The following thing seems to work:
// gcc -lOpenCL -c -S -O3 shuffle64.c

// maybe useful for gcc/clang
// #include <immintrin.h>

// intersting page:
// https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics

// problem with clang and __builtin_shuffle
// https://lists.freedesktop.org/archives/pixman/2018-June/004728.html

// wiki for predefine C macros
// https://github.com/cpredef/predef

// get all defined macros:
// gcc -E -dM shuffle64.c >macros.txt
// better e.g.:
// gcc -E -dM -march=native shuffle64.c | find "AVX512"
// important for 64-byte permutations: #define __AVX512VBMI__ 1

// my favourite instructions:
// vpermt2b
// vpermi2pd       merge 64-bit words from two sources

// very useful:
// https://stackoverflow.com/questions/77205933/how-to-transpose-a-8x8-int64-matrix-with-avx512



// Here the author had to learn it the REALLY HARD way:
// The following data structures are usually created by Cython
// (compiled with MSVC for Windows) and used by the C functions 
// in this package (always compiled with gcc).
// Therefore, EVERY compiler that processes these structures
// must respect their alignment!!!


// #define DEBUG_DUMP



#if defined(__GNUC__)
  #define ALWAYS_INLINE __attribute__((always_inline))
  #define ALIGNED(n)  __attribute__((aligned(n)))
#elif defined(_MSC_VER)
  #define ALWAYS_INLINE
  #define ALIGNED(n) __declspec(align(n))
#else  
  #define ALWAYS_INLINE
  #define ALIGNED(n)  // don't know how to force alignment
#endif


#ifdef DEBUG_DUMP
static volatile int _debug_dump_ = 0;
#endif


#if defined(__GNUC__) || defined(__clang__)
// GCC / Clang version
  #ifdef DEBUG_DUMP
    #define ASSUME_ALIGNED(ptr, n) \
    do { (ptr) = __builtin_assume_aligned((ptr), (n));  \
         if ((uintptr_t)(ptr) % n != 0) { \
             printf("pointer %p is not %d-bit aligned", ptr, n); \
             fflush(stdout); \
             ++ _debug_dump_; \
         } \
    } while (0)
  #else
    #define ASSUME_ALIGNED(ptr, n) \
    do { (ptr) = __builtin_assume_aligned((ptr), (n)); } while (0)
  #endif
#elif defined(_MSC_VER)
// MSVC version
#define ASSUME_ALIGNED(ptr, n) \
    __assume(((uintptr_t)(ptr) & ((n) - 1)) == 0)
#else
// Fallback: do nothing
#define ASSUME_ALIGNED(ptr, n) ((void)0)
#endif
// usage: a function call
// ASSUME_ALIGNED(ptr, 64);
// lets the compiler assume that the data referred by ptr is 64-bit aligned.



// Portable, aligned alloca() that works with GCC, Clang, and MSVC
#if defined(__GNUC__) || defined(__clang__)
    #define _raw_alloca(size)  __builtin_alloca(size)
#elif defined(_MSC_VER)
    #define _raw_alloca(size) _alloca(size)
#else
#   error "alloca_aligned is not supported on this compiler."
#endif

#define alloca_aligned(size, alignment) \
    ((void *)(((uintptr_t)_raw_alloca(size + alignment - 1) + (alignment - 1)) & ~(uintptr_t)(alignment - 1)))


#ifdef __GNUC__
#  ifdef __x86_64__
#    if defined(__AVX__) || defined(__SSSE3__) || defined(__AVX2__) || defined(__AVX512__)
#         define GCC_VECTORS 1
#    endif 
#    if defined(__AVX2__)
#         define GCC_AVX2 1
#    endif 
#    if defined(__AVX512VBMI__)
#         define SHUFFLE64 1
#    endif 
#  endif
#endif


#ifdef __GNUC__
#  ifdef __x86_64__
#    if defined(__AVX512__) || defined(__AVX512VBMI__)
#        define GCC_VECTOR_ALIGNED 64
#    elif defined(__AVX2__)
#        define GCC_VECTOR_ALIGNED 32
#    endif 
#  endif
#endif


#ifndef GCC_VECTOR_ALIGNED
#define GCC_VECTOR_ALIGNED 16
#endif


#ifdef __GNUC__
#include <immintrin.h>
typedef uint8_t v64_8_type __attribute__ ((vector_size (64)));
typedef uint8_t v32_8_type __attribute__ ((vector_size (32), aligned(32)));
typedef uint8_t v16_8_type __attribute__ ((vector_size (16), aligned(16)));
typedef uint8_t v8_8_type __attribute__ ((vector_size (8)));
typedef uint16_t v16_16_type __attribute__ ((vector_size (32), aligned(32)));
typedef uint16_t v8_16_type __attribute__ ((vector_size (16), aligned(16)));
typedef uint64_t v4_64_type __attribute__ ((vector_size (32), aligned(32)));
typedef uint32_t v8_32_type __attribute__ ((vector_size (32), aligned(32)));
typedef uint32_t v16_32_type __attribute__ ((vector_size (64), aligned(64)));
#else
typedef ALIGNED(64) uint8_t v64_8_type[64];
typedef ALIGNED(32) uint8_t v32_8_type[32];
typedef ALIGNED(16) uint8_t v16_8_type[16];
typedef uint8_t v8_8_type[8];
typedef ALIGNED(32) uint16_t v16_16_type[16];
typedef ALIGNED(16) uint16_t v8_16_type[8];
typedef ALIGNED(32) uint64_t v4_64_type[4];
typedef ALIGNED(32) uint32_t v8_32_type[8];
typedef ALIGNED(64) uint32_t v16_32_type[16];
#endif





#ifdef NO_GCC_AVX 
#  ifdef GCC_AVX2
#    undef GCC_AVX2  // useful for testing
#  endif
#endif


#if defined(GCC_VECTORS) && (SHUFFLE64 != 1)
  #define  MM_OP_FAST_SPLIT_SHUFFLE24 1
#else
  #define  MM_OP_FAST_SPLIT_SHUFFLE24 0
#endif



/******************************************************************************
# if GCC_VECTORS is defined we assume that the following buitins are available:
__builtin_shuffle

# if SHUFFLE64 we assume that a 64-byte shuffle makes sense with that CPU.
******************************************************************************/


#define MM_FAST_BYTELENGTH 247488

 ALIGNED(32) typedef union {
    uint8_t b[32];
    uint16_t u16[16]; 
    uint32_t u32[8]; 
    uint64_t u64[4]; 
    v8_8_type v8[4];
    v16_8_type v16[2];
    ALIGNED(32) v32_8_type v32[1];
  #ifdef GCC_VECTORS
    v8_32_type v8_32[1];
  #endif
  #ifdef GCC_AVX2
    ALIGNED(32) __m256i m256[1];
    __m128i m128[2];
  #endif
} mmv_fast_row32_type;


ALIGNED(64) typedef union {
    uint8_t b[64];
    uint64_t u64[8]; 
    v8_8_type v8[8];
    v16_8_type v16[4];
    v32_8_type v32[2];
    mmv_fast_row32_type v32row[2];
  #ifdef GCC_VECTORS
    v8_32_type v8_32[2];
  #endif
  #if GCC_VECTOR_ALIGNED >= 64
    ALIGNED(64) v64_8_type v64[1];
    v16_32_type v16_32[1];
  #endif
  #ifdef GCC_AVX2
  #if GCC_VECTOR_ALIGNED >= 64
    ALIGNED(64) __m512i m512[1];
  #endif
    __m256i m256[2];
    __m128i m128[4];
  #endif
} mmv_fast_row64_type;

// Structure mmv_fast_row64_type needs this funny component ``v16_32``
// because AVX512 hardware supports shift of 32-bit integer entries of
// a vector, but not 8-bit integer entries. For addition mod 3, it does
// not matter whether we shift 8-bit or 32-bit intrger entries. So we
// do what the hardware can do best. 
// Similarly, struct mmv_fast_row32_type needs a component ``_v8_32``.



 ALIGNED(64) typedef union {
    uint8_t b[MM_FAST_BYTELENGTH];
    uint64_t u64[MM_FAST_BYTELENGTH/8]; 
    v32_8_type v32[MM_FAST_BYTELENGTH/32];
  #if GCC_VECTOR_ALIGNED >= 64
    v64_8_type v64[MM_FAST_BYTELENGTH/64];
  #endif
  #ifdef GCC_AVX2
    __m256i m256[MM_FAST_BYTELENGTH/32];
    __m128i m126[MM_FAST_BYTELENGTH/16];
  #endif
  mmv_fast_row64_type r64[MM_FAST_BYTELENGTH/64];
  mmv_fast_row32_type r32[MM_FAST_BYTELENGTH/32];
} mmv_fast_type;



typedef union {
   mmv_fast_type* p_vb[2];
} mmv_fast_matrix_union_type;


typedef struct {
    mmv_fast_matrix_union_type p_v;
    uint32_t p;
    uint32_t nrows;
    uint32_t mode;
    uint32_t check_underflow;
    uint32_t current;
} mmv_fast_matrix_type;




ALIGNED(32) typedef struct  {
    mmv_fast_row32_type a_data[49]; // one more the used for slack
    mmv_fast_row32_type *a;
    mmv_fast_matrix_type *p_source;
    uint64_t seed[4];
    int32_t row_source;
    int32_t sub_row_source;
    int32_t norm;
    int32_t diag;
    int32_t dim_img;
    int32_t mode_B;
    int32_t len_B[2];
} mmv_fast_Amod3_type;



#ifdef DEBUG_DUMP
#include <stdio.h>
#include <stdarg.h>
static inline void DEBUG_DUMP_(int cond, char *fmt, ...) {
    if (!cond) return;
    va_list args;
    printf("DUMP: ");
    va_start(args, fmt);
    vprintf(fmt, args);  // Print formatted output directly
    va_end(args);
    printf("\n");
    fflush(stdout);
}
#define DUMP(fmt, ...) \
    do {DEBUG_DUMP_(1, fmt, __VA_ARGS__); ++_debug_dump_;} while (0)
#define DUMP_COND(condition, fmt, ...) \
    do {DEBUG_DUMP_(condition, fmt, __VA_ARGS__); ++_debug_dump_;} while (0)
#else
#define DUMP(fmt, ...)
#define DUMP_COND(condition, fmt, ...)
#endif


#endif  // define MM_OP_FAST_H_INCLUDED


