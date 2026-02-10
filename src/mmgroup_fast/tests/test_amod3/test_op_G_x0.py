from collections import OrderedDict
import numpy as np
from random import randint, shuffle, sample

import pytest

from mmgroup.generators import mm_group_invert_word
from mmgroup_fast.mm_op_fast import MMOpFastMatrix, MMOpFastAmod3
from mmgroup import MM0, MMV, MM, XLeech2, Cocode, Xsp2_Co1
from mmgroup.axes import Axis



def get_test_axes():
    for ax in Axis.representatives().values():
        yield ax * Xsp2_Co1('r')



def one_test_FastAmod3_op_Gx0(axis, verbose = 0):
    a3 = MMOpFastAmod3(axis)
    g = Xsp2_Co1('r')
    A = a3.op_Gx0(g).data[1]
    A_ref = ((axis * g).v15 % 3)['A']
    if verbose:
        print("Test method MMOpFastAmod3.op_Gx0")
        print(A)
        print(A_ref)
        print("")
    assert (A == A_ref).all(), (A - A_ref) & 7
    gi = (g**-1)
    a3i = MMOpFastAmod3(axis)
    mi = gi.mmdata
    mm_group_invert_word(mi, len(mi))
    #print([hex(x) for x in mi])
    A1 = a3i.op_Gx0(mi).data[1]
    assert (A1 == A_ref).all(), (A1 - A_ref) & 7
   
    
   



@pytest.mark.mm_amod3
def test_MMOpFastAmod3_op_Gx0():
    for axis in get_test_axes():
        one_test_FastAmod3_op_Gx0(axis, verbose = 0)


