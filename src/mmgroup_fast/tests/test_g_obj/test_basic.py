
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup.clifford12 import uint64_bit_len
from mmgroup_fast.mm_op_fast import MMOpFastG
from mmgroup import MM0, MMV, MM
from mmgroup_fast.mm_op_fast import MMOpFastMatrix


MMV3 = MMV(3)


def hex_array(text, arr):
    lines = [text + " = ["]

    for i in range(0, len(arr), 6):
        chunk = arr[i:i+6]
        line = ", ".join(f"0x{v & 0xFFFFFFFF:08X}" for v in chunk)
        if i + 6 < len(arr):
            line += ","
        lines.append(line)

    lines.append("]")
    return "\n".join(lines)



def chk_MMOpFastG_equ_mm(a, g, message = "", do_raise = True):
    if not isinstance(a, MMOpFastG):
        ERR = "Element should be %s, but is %d"
        raise ValueError(ERR % (type(MMOpFastG()), type(a)))
    mm = MM('a', np.concatenate((a.copy().inverse_mmdata, g.mmdata)))
    if mm == MM():
        return
    print("\n")
    if (message):
        print(message)
    print("MMOpFastG element is:\n  %s" % str(a))
    a._display_flags()
    print("But it should be equal to:\n  %s" % str(g))
    ERR = "Monster group elements are different"
    if do_raise:
        raise ValueError(ERR)
    else:
        print("\nError: %s!!!\n" % ERR)

@pytest.mark.mm_op
def test_basics(verbose = 0):
    data = [
        (str(MM0('r', 14)),),
        (),  
        ('x',2),
       # ([('t',1), ('y',2), ('t',1)],),
       # ([('t',1), ('l',2), ('t',1)],),
        (MM0('r', 3),),
        ([('t',1), ('l',2), ('p',172227289)],),
        (str(MM0('r', 7)),),
    ]
    for n, d in enumerate(data):
        a0 = MM0(*d).reduce()
        a = MMOpFastG(*d)
        if verbose:
            print("\nTest", n+1)
            print("Test element is: ", a0)
            print("Test element has length", len(a0.mmdata))
            a._display_flags()
        assert MM('a', a.copy().g(0)) == MM(a0)
        mat_g = a.copy().mat()
        assert isinstance(mat_g, MMOpFastMatrix)
        if verbose:
            # a._display_flags()
            mat_g._display_status()
        mat_g1 = MMOpFastMatrix(3,4,1)
        mat_g1.set_vstd(1)
        mat_g1.mul_exp(a0)
        g1i_red = mat_g1.copy().reduce_v_g()
        g1_neutral = MM('a', np.concatenate((a0.mmdata, g1i_red)))
        assert g1_neutral == MM()
        if verbose:
            a._display_flags()
            print("g1_inv =", MM0('a',g1i_red)) 
            print("g_inv =", MM0('a', mat_g.copy().reduce_v_g())) 
        
        for i in range(3):
            if mat_g.row_as_mmv(i) != mat_g1.row_as_mmv(i):
                print(mat_g.row_as_mmv(i)['A'])
                print(mat_g1.row_as_mmv(i)['A'])
                ERR = "Error in comparing matrix row %d"
                raise ValueError(ERR % i)
        gi_red = mat_g1.reduce_v_g()
        g_neutral = MM('a', np.concatenate((a0.mmdata, gi_red)))
        assert g_neutral == MM()
        if verbose:
            print(hex_array("a0", a0.mmdata))
            print(hex_array("a", a.mmdata))
        chk_MMOpFastG_equ_mm(a, a0)
        assert MM(a0) == MM('a', a.mmdata) # This fails!!!
        del a



def do_test_exp(max_exp, factor, verbose):
    b = MM('r')
    bf = MMOpFastG(b)
    bpwr = b**factor
    chk_MMOpFastG_equ_mm(bf, b)
    b_exp = MM()
    for e0 in range(max_exp+1):
        e = e0 * factor
        if verbose:
            print("exponent", e, ", bitlen(e) =", uint64_bit_len(abs(e)))
        bf_exp = bf ** e
        msg = "Error in mulexp, e = %d" % e
        chk_MMOpFastG_equ_mm(bf_exp, b_exp, msg, False)
        if e0 <= max_exp:
           b_exp *= bpwr


@pytest.mark.mm_op
def test_exp(verbose=0):
    do_test_exp(7, 1, verbose)
    do_test_exp(7, -1, verbose)
    do_test_exp(5, 5, verbose)
    do_test_exp(5, -5, verbose)



