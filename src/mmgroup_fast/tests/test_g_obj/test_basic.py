
import numpy as np
from random import randint, shuffle

import pytest

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


@pytest.mark.mm_op
def test_basis(verbose = 0):
    data = [
        (),  
       # ('x',2),
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
        assert MM('a', a.g(0)) == MM(a0)
        mat_g = a.mat()
        assert isinstance(mat_g, MMOpFastMatrix)
        if verbose:
            mat_g._display_status()
        mat_g1 = MMOpFastMatrix(3,4,1)
        mat_g1.set_vstd(1)
        mat_g1.mul_exp(a0)
        g1i_red = mat_g1.copy().reduce_v_g()
        g1_neutral = MM('a', np.concatenate((a0.mmdata, g1i_red)))
        assert g1_neutral == MM()
        if verbose:
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
        assert MM(a0) == MM('a', a.mmdata) # This fails!!!
        del a

