// %%GEN h
#ifndef MM_OP_FAST_H
#define MM_OP_FAST_H

/** @file mm_op_fast.h
  File ``mm_op_fast.h`` is a header file. 
*/



#include "mm_basics.h"
#include "mm_op_fast_types.h"







/***************************************************************
*** 24-byte permutations
***************************************************************/
/// @cond DO_NOT_DOCUMENT 

typedef ALIGNED(64) union{
    uint8_t b[96];
    v16_8_type v16[6];
    v32_8_type v32[3];
  #if GCC_VECTOR_ALIGNED >= 32
    mmv_fast_row32_type v32row;
  #endif
  #ifdef SHUFFLE64
    mmv_fast_row64_type v64row;
  #endif
  #ifdef GCC_AVX2
    __m256i m256[3];
    __m128i m128[6];
  #endif
} mmv_fast_shuffle24_type;    

/// @endcond


/***************************************************************
*** Preparation of a generator x_d * x_pi 
***************************************************************/

/// @cond DO_NOT_DOCUMENT 



/** @struct mmv_fast_op_pi_type

 @brief Structure used for preparing an operation \f$x_\epsilon  x_\pi\f$

 Function ``mm_sub_prep_pi`` computes some tables required for the operation
 of \f$x_\epsilon  x_\pi\f$ on the representation of the monster group, and
 stores these tables in a structure of type ``mm_sub_op_pi_type``.

 The structure of type ``mm_sub_op_pi_type`` has the following members:
*/
typedef ALIGNED(64) struct {
    /**
      Structure for accelerating a permutation of 24 bytes.
    */
    mmv_fast_shuffle24_type perm24;
    /**
      The inverse of the permutation ``perm``.
    */
    mmv_fast_row32_type inv_perm;

    /**
       A 12-bit integer describing an element  \f$\epsilon\f$ of
       the Golay cocode.
    */
    uint32_t eps; 
    /**
      An integer describing the element \f$\pi\f$ of the Mathieu
      group \f$M_{24}\f$ as in module ``mat24_functions.c``.
    */    
    uint32_t pi;
    /**
      The permutation ``0...23 -> 0...23`` given by the
      element \f$\pi\f$ of \f$M_{24}\f$.
    */
    // uint8_t perm[24];

    
    /**
      For tags ``X, Y, Z``, an entry ``(tag, i, j)`` of the
      representation of the monster is mapped to entry ``(tag1, i1, j1)``,
      with ``i1`` depending on ``i`` (and the tag), and ``j1``
      depending on ``j`` only.

      If ``tbl_perm24_big[i1] & 0x7ff = i`` for ``0 <= i1 < 2048``
      then ``(tag, i, j)`` ia mapped to ``(Tag, i1, perm[j])``, up to sign,
      for tags ``X``, ``Y`` and ``Z``. In case of an odd \f$\epsilon\f$,
      tags ``Y`` and ``Z`` have to be exchanged. The
      value ``tbl_perm24_big[2048 + 24*k + i1] & 0x7ff`` describes the
      preimage of ``(tag, i1, j1)`` in a similar way,
      where ``tag = A, B, C``, for ``k = 0, 1, 2``.

      Bits 12,...,15 of ``tbl_perm24_big[i1]`` encode the signs of the
      preimages of the corresponding entry of the rep. Bits 12, 13, and 14
      refer to the signs for the preimages for the tags ``X``, ``Z``
      and ``Y``, respectively. Bit 15 refers to the signs for the preimages
      for tags ``A``, ``B`` and ``C``. If the corresponding bit is set,
      the preimage has to be negated.

      Note that function ``mat24_op_all_autpl`` in
      module ``mat24_functions.c computes``the first 2048 entries of
      the table.
    */


    uint16_t *p_autpl;
    uint16_t *p_autpl_next;
	
    uint16_t p_autpl_ABC[24];
    
} mmv_fast_op_pi_type;




/** @struct mm_fast_sub_op_pi64_type from "mm_basics.h"

 @brief Auxiliary structure for the structure ``mm_sub_op_pi_type``


 An array of type ``mm_sub_op_pi64_type[759]`` encodes the operation
 of  \f$x_\epsilon x_\pi\f$ on the representation of the monster group
 for entries with tag ``T``. Assume that entry ``(T, i, j)`` is mapped
 to entry ``+-(T, i1, j1)``. Then ``i1`` depends on ``i`` only, and ``j1``
 depends on ``i`` and ``j``. For fixed ``i`` the mapping ``j -> j1`` is
 linear if we consider the binary numbers ``j`` and ``j1`` as bit vectors.

 Entry ``i1`` of the array of type ``mm_sub_op_pi64_type[759]``
 describes the preimage of ``(T, i1, j1)`` for all ``0 <= j1 < 64``
 as documented in the description of the members ``preimage``
 and  ``perm``.
 
 Note that the values 1, 3, 7, 15, 31, 63 occur as
 differences `` j1 ^ (j1 - 1)`` when counting ``j1`` from 0 up to 63. So the
 preimage of ``(T, i1, j1)`` can be computed from the preimage
 of ``(T, i1, j1 - 1)`` using linearity and the approprate entry in
 member perm.

 We remark that in case of an odd value epsilon the mapping for tag ``T``
 requires a postprocessing step that cannot be derived from the
 infomration in this structure. Then entry ``(T, i, j)`` has to be negated
 if the bit weight of the subset of octade ``i`` corresponding to
 index ``j`` has bit weight 2 modulo 4.
 
 In the sequel we describe the meaning of entry ``i1`` an an array of 
 elements of type ``mm_sub_op_pi64_type``.
*/
typedef struct {
   /**
   Bits 9...0 : preimage ``i`` such that ``(T, i, .)`` maps to ``+-(T, i1, .)``

   Bit 12: sign bit: ``(T, i, .)`` maps to ``-(T, i1, .)`` if bit 12 is set
   */
   uint16_t preimage;
   /**
   Member ``perm[k]`` is a value ``v ``such that ``(T, i, v)`` maps 
   to ``+-(T, i1, 2 * 2**k - 1)``
   */
   uint8_t perm[6];
} mm_fast_sub_op_pi64_type;

/// @endcond


/***************************************************************
*** Preparation of a generator x_f * x_e * x_d 
***************************************************************/

// Yet to be done



/** @struct mm_sub_op_xy_type "mm_basics.h"

  @brief Structure used for preparing an operation \f$y_f x_e x_\epsilon\f$
  
  The operation of \f$g = y_f x_e x_\epsilon\f$, (or, more precisely, of its 
  inverse \f$g^{-1}\f$) on the representation of the monster group is 
  described in section **Implementing generators of the Monster group** in 
  the **The mmgroup guide for developers**. 
  
  Function ``mm_sub_prep_xy`` in file ``mm_tables.c`` collects the data
  required for this operation in a structure of type ``mm_sub_op_xy_type``.

*/
typedef ALIGNED(32) struct {
    /**
       Byte \f$i\f$ of member ``e_i`` is the scalar product of \f$e\f$ and
       the singleton cocode word  \f$(i)\f$.
    */
    mmv_fast_row32_type e_i;
    /**
       Byte \f$i\f$ of member ``f_i`` is the scalar product of \f$f\f$ and
       the singleton cocode word  \f$(i)\f$.
    */
    mmv_fast_row32_type f_i;
    /**
       Bit \f$i\f$ of member ``ef_i`` is the scalar product of \f$ef\f$ and
       the singleton cocode word  \f$(i)\f$.
    */
    mmv_fast_row32_type ef_i;
    /**
       See documentation of component ``lin_d``. Here ``p_lin_i[i]``
       is ``ef_i, f_i, f_i, e_i`` for ``i = 0, 1, 2, 3``, respectively.
    */
    mmv_fast_row32_type *p_lin_i[4];

    /**
       A 13-bit integer describing an element  \f$f\f$ of the Parker loop.
    */
    uint32_t f;            
    /**
       A 13-bit integer describing an element  \f$e\f$ of the Parker loop.
    */
    uint32_t e;
    /**
       A 12-bit integer describing an element  \f$\epsilon\f$ of
       the Golay cocode.
    */
    uint32_t eps;
    /**
        Bit \f$i\f$ of member `vf`` is byte \f$i\f$ of member ``f_i``. 
    */
    uint32_t vf;
    /**
        Bit \f$i\f$ of member `vef`` is byte \f$i\f$ of member ``ef_i``. 
    */
    uint32_t vef;
    /**
       Let ``U_k = X, Z, Y`` for ``k = 0, 1, 2``. If the cocode
       element \f$\epsilon\f$ is even then we put ``U'_k = U_k``, otherwise 
       we put  ``U'_k = X, Y, Z``   for ``k = 0, 1, 2``. The
       operation \f$g^{-1}\f$ maps the vector with tag ``(U_k, d, i)`` 
       to ``(-1)**s`` times the vector with tag ``(U'_k, d ^ lin[d], i)``. 
       Here ``**`` denotes exponentiation and we have
       
       ``s`` =  ``s(k, d, i)`` = ``(p_lin_i[k] >> i) + (sign_XYZ[d] >> k)``.

       If ``k = 0`` and \f$\epsilon\f$  is odd then we have to 
       correct ``s(k, d, i)``  by a  term ``<d, i>``.
    */
    uint32_t lin_d[3];
    /**
       Pointer ``p_signs`` refers to an array of length 2048. This is
       used for calculations of signs as described above. Here we use the
       formula in section **Implementing generators of the Monster group**
       of the  **mmgroup guide for developers**, dropping all terms
       depending on ``i``.

       Signs for tags 'X', 'Z', and 'Y' are stored in bit 0, 1, and 2,
       respectively. For te cases ``d < 2048`` the sign for tag 'T' is
       stored in bit 6.
    */
    uint16_t *p_signs;

    /**
       The following stuff is deprecated ans subject to change!!!!

       Pointer ``s_T`` refers to an array of length 759.  Entry ``d`` 
       of this array refers to the octad ``o(d)``  with number ``d``. 
       The bits of entry ``d`` are interpreted as follows: 
    
       Bits 5,...,0: The asscociator ``delta' = A(o(d), f)`` encoded
       as a suboctad of octad ``o(d))``.
       
       Bits 13,...,8: The asscociator ``alpha = A(o(d), ef)`` encoded
       as a suboctad of octad ``o(d))``. From his information we can
       compute the scalar product ``<ef, \delta>`` for each suboctad 
       ``delta`` of ``o(d)`` as an  intersection of tow suboctads.
       Here we assume that ``delta`` is represented as such a suboctad.
       
       Bit 14: The sign bit ``s(d) = P(d) + P(de) + <d, eps>``, where
       ``P(.)`` is the squaring map in the Parker loop.
       
       Bit 15: Parity bit ``|eps|`` of the cocode word ``eps``. 
       
       Then \f$g^{-1}\f$ maps the vector with tag ``(T, d, delta)`` 
       to ``(-1)**s'`` times  the vector with 
       tag ``(T, d, \delta ^ delta')``. 
       Here ``**`` denotes exponentiation and we have
       
       ``s'`` = ``s'(T, d, delta)`` 
       = ``s(d)`` + ``<\alpha, \delta>`` + ``|delta| * |eps| / 2``. 
       
       Here the product ``<\alpha, \delta>`` must be computed as the
       bit length of an intersection of two suboctads.        
    */
    uint16_t *s_T;
} mmv_fast_op_xy_type;




/***************************************************************
*** Exported functions
***************************************************************/


// %%INCLUDE_HEADERS





/***************************************************************
*** Include macros and inline functions
***************************************************************/

#if defined(MM_OP_FAST_ALLOC) || \
   defined(MM_OP_FAST_PERMUTATIONS) || defined(MM_OP_FAST_HADAMARD)
#include "mm_op_fast_intern.h"
#endif


/***************************************************************
*** End of file
***************************************************************/



// %%GEN h
#endif  //  ifndef MM_OP_FAST_H
// %%GEN c




