// %%GEN h
#ifndef MM_OP_FAST_INTERN_H
#define MM_OP_FAST_INTERN_H

/** @file mm_op_fast.h
  File ``mm_op_fast_header.h`` is a header file. It contains internal
  function used for the ``mm_op_fast`` extension.
*/


#include "mmgroup_endianess.h"



/***************************************************************
*** Aligned memory allocation
***************************************************************/

// E.g. in Windows some parts of the project are compiled with
// MSVC, others with gcc. We don't trust both compilers when
// aligned memory has to be allocated.

static inline void *_my_aligned_alloc(size_t size, size_t alignment) {
    // Allocate enough for data + alignment slack + space to store original pointer
    void *raw = malloc(size + alignment - 1 + sizeof(void*));
    if (!raw) return NULL;

    // Align the pointer
    uintptr_t raw_addr = (uintptr_t)raw + sizeof(void*);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(uintptr_t)(alignment - 1);

    // Store the original pointer just before the aligned pointer
    ((void**)aligned_addr)[-1] = raw;

    return (void*)aligned_addr;
}

static inline void _my_aligned_free(void *ptr) {
    if (ptr) {
        free(((void**)ptr)[-1]);
    }
}





/***************************************************************
*** Include macros and inline functions
***************************************************************/


// %%INCLUDE_HEADERS


// %%GEN h
#endif  //  ifndef MM_OP_FAST_INTERN_H
// %%GEN c




