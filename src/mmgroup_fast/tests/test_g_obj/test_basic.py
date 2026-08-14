
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup.clifford12 import uint64_bit_len
from mmgroup_fast.mm_op_fast import MMOpFastG
from mmgroup import MM0, MMV, MM, MM_from_int
from mmgroup_fast.mm_op_fast import MMOpFastMatrix
from mmgroup_fast.mm_op_fast import fast_g_obj_subgroup_mul_e

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


#####################################################################
# Test multiplication of Monster elements in classes
# MMOpFastG and reduction in class MMOpFastMatrix 
#####################################################################

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
        g1i_red = mat_g1.copy().reduce_v_g(mode=1)
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
        gi_red = mat_g1.reduce_v_g(mode=1)
        g_neutral = MM('a', np.concatenate((a0.mmdata, gi_red)))
        assert g_neutral == MM()
        if verbose:
            print(hex_array("a0", a0.mmdata))
            print(hex_array("a", a.mmdata))
        chk_MMOpFastG_equ_mm(a, a0)
        assert MM(a0) == MM('a', a.mmdata)
        del a


#####################################################################
# Test exponentiation of Monster elements in class MMOpFastG 
#####################################################################



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



#####################################################################
# Test conversion of Monster elements to integers in class MMOpFastG 
#####################################################################


@pytest.mark.mm_op
def test_reduce_v_g():
    for i in range(3):
        m = MM('r')
        print(m)
        mm = MMOpFastMatrix(3,4,1)
        mm.set_vstd(hash = 1)
        mm.mul_exp(m)
        mm_g = MM('a', mm.reduce_v_g(mode = 0x17))
        assert mm_g == m
        ii = mm.reduce_v_g_as_int()
        assert MM_from_int(ii) == m, ii
 
        a = MMOpFastG(m)
        assert MM('a', a.mmdata) == m
        n = a.as_int()
        assert MM_from_int(n) == m



#####################################################################
# Test elements of subgoups of Monster in class MMOpFastG  
#####################################################################


def try_subgroup(g, e, h, f):
    a = np.zeros(4, dtype = np.uint64)
    gout = np.zeros(10, dtype = np.uint32)
    res = fast_g_obj_subgroup_mul_e(
        g.mmdata, len(g.mmdata), e,
        h.mmdata, len(h.mmdata), f,
        gout, a
    )
    if res < 0:
        return 0, None
    n = (int(a[0]) + (int(a[1]) << 64) + (int(a[2]) << 128) 
        + (int(a[3]) << 196))
    return n, MM('a', gout[:res])

def subgroup_samples():
    """Yield tuples g, e, h, f, b with g, h of class MM integers e,f

    Here g, h are elements of the Monster M, that may be in one of the
    subgroups G_x0 or N_0 of M. Then a test program should that check
    that  g**e  * h**f  is computed correctly in  class MMOpFastG.

    b is 1 if try_subgroup(g, e, h, f) is expected to succeed and 0
    otherwise
    """
    data = [(MM(), 1, MM(), 1, 1),
       (MM('r', 'G_x0'), 3, MM('r', 'G_x0'), -1, 1),
       (MM('r'), 1, MM('r'), 1, 0),
       (MM('r','N_0'), 1, MM('r'), 1, 0),
       (MM('r'), 1, MM('r','N_0'), 1, 0),
       (MM('r', 'G_x0'), 1, MM('r','N_0'), 1, 0),
       (MM('r', 'N_0'), 1, MM('r','G_x0'), -1, 0),
       (MM('r', 'N_0'), 3, MM('r','N_x0'), 0, 1),
       (MM('r', 'N_0'), -4, MM('r', 'N_0'), 2, 1),
    ]
    for i in range(2):
        for x in data:
            yield x

def one_test_subgroup(g, e, h, f, s, verbose=0):
    r = MMOpFastG(g) ** e
    r.mulexp(h, f)
    i1 = r.as_int()
    r_ref = MM(g)**e * MM(h)**f
    mm = MMOpFastMatrix(3,4,1)
    mm.set_vstd(hash = 1)
    mm.mul_exp(r_ref)
    mm_r = MM('a', mm.reduce_v_g(mode = 0x17))
    ok = mm_r == r_ref
    i1_ref = mm.reduce_v_g_as_int()
    ok &= i1 == i1_ref
    ti, tr = try_subgroup(g, e, h, f)
    if s:
        ok &= tr == r_ref
        ok &= ti ==i1
    if not ok or verbose:
        print("g=", g)
        print("e=", e)
        print("h=", h)
        print("f=", f)
        print("s=", s)
        if (s and tr != r_ref):
            print("r obtained directly=\n  ",tr)
        if (mm_r != r_ref):
            print("r obtained=\n  ", mm_r)
        print("r=", r_ref)
        print("i expected", hex(i1_ref))
        print(" ", MM_from_int(i1_ref))
        print("i obtained", hex(i1))
        print(" ", MM_from_int(i1))
        if s:
            print("i subgroup", hex(ti))
            print(" ", MM_from_int(ti))
        ERR = "Integer compression of Monster element failed"
        if not ok:
            raise ValueError(ERR)


@pytest.mark.mm_op
def test_MM_subgroup(verbose = 0):
    for i, (g, e, h, f, s) in enumerate(subgroup_samples()):
        if verbose:
            print("\nTest", i)
        one_test_subgroup(g, e, h, f, s,verbose)
