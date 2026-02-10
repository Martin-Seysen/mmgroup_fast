
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup_fast.mm_op_fast import mm_op_fast_test_op_t_tag_T
from mmgroup.bitfunctions import bitparity, bitweight




MATRICES = [None] * 16

def op_matrix(mode):
    global MATRICES
    mode &= len(MATRICES) - 1
    if MATRICES[mode] is not None:
        return MATRICES[mode]
    if MATRICES[0] is None:
        MATRICES[0] = np.eye(64, dtype = np.int32)        
    if MATRICES[4] is None:
        MATRICES[4] = np.zeros((64, 64), dtype = np.int32)
        for i in range(64):
            for j in range(64):
                MATRICES[4][i, j] = 1 - 2 * bitparity(i & j)
    if MATRICES[1] is None:
        MATRICES[1] = np.zeros((64, 64), dtype=np.int32)
        for i in range(64):
            MATRICES[1][i, i] = 1 if bitweight(i) in [0, 3, 4] else -1
        MATRICES[8] =  MATRICES[1]
    if MATRICES[2] is None:
        MATRICES[2] = np.zeros((64, 64), dtype=np.int32)
        for i in range(64):
            if bitparity(i):
                MATRICES[2][i, 63 - i] = 1
            else:
                MATRICES[2][i, i] = 1
        MATRICES[8] =  MATRICES[1]
    if MATRICES[mode] is None:
        j = mode & -mode
        M = np.copy(MATRICES[j])
        j += j
        while j < len(MATRICES):
           if j & mode:
               M = M @ MATRICES[j]
           j += j
        MATRICES[mode] = M
    return MATRICES[mode]
    


MATRICES_MOD_P = {}
 
def op_matrix_mod_p(mode, p):
    if (mode, p) in MATRICES_MOD_P:
        return MATRICES_MOD_P[(mode, p)]
    if (mode & 4) == 0:
        MATRICES_MOD_P[(mode, p)] = op_matrix(mode)
        return MATRICES_MOD_P[(mode, p)] 
    M = op_matrix(mode)
    if p >= 7:
        assert (p + 1) & 7 == 0
        q = (p + 1) >> 3
        MATRICES_MOD_P[(mode, p)] = (M * q) % p
        return MATRICES_MOD_P[(mode, p)]
    assert p == 3 
    MATRICES_MOD_P[(mode, 3)] = (M * 2) % 3
    return MATRICES_MOD_P[(mode, 3)]





def one_test_tag_T_mod3(mode, data = None, verbose = 0):
    if not data:
        data = [randint(0, 255) for i in range(64)]
    else:
        assert len(data) == 64
    a = np.array(data, dtype = np.uint8)
    a_in = np.zeros((4, 64), dtype = np.int32)
    a_ref = np.zeros((4, 64), dtype = np.int32)
    M = op_matrix_mod_p(mode, 3)
    for i in range(4):
        a_in[i,:64] = np.array((a >> (2*i)) & 3, dtype = np.int32)
        a_ref[i, :64] = np.array((a_in[i] @ M) % 3, dtype = np.uint8)
    if verbose:
        print("Testing mode ", mode)
    res = mm_op_fast_test_op_t_tag_T(3, mode, a)
    if res < 0:
        print("Testing operation of t on tag T, mode =", mode)
        err = "Operation t on tag 'T' failed with result %d" % res
        raise ValueError(err)
    a_out = np.zeros((4, 64), dtype = np.uint8)
    # a[17] ^= 4 # Produce an artificial error for checking test
    for i in range(4):
        a_out[i, :64] = np.array(((a >> (2*i)) & 3) % 3, 
             dtype = np.uint8)
        error = not (a_ref[i] == a_out[i]).all()
        if error or verbose:
            for k in range(0, 64, 16):
                print("input %d: " % i, a_in[i][k:k+16] % 3)
                print("expected:", a_ref[i][k:k+16])
                print("obtained:", a_out[i][k:k+16])
        if error:
            for j in range(64):
                if a_ref[i, j] != a_out[i, j]:
                    break 
            print("Testing operation of t on tag T, mode =", mode)
            print("Error at vector", i, ", index ", j)
            err = "Operation t on tag 'T' failed"
            raise ValueError(err)



def tag_T_mod3_testdata():
    yield 4, [1] + [0] * 63
    for mode in [1, 2, 4, 7, 14]:
        for i in range(10):
            yield mode, None


@pytest.mark.mm_op
def test_tag_T_mod3(verbose=0):
    print("\nTesting operation 't' on tag 'T' (modulo 3)")
    for mode, data in tag_T_mod3_testdata():
        one_test_tag_T_mod3(mode, data, verbose)







    