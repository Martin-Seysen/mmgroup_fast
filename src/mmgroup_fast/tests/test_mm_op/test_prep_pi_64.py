from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import numpy as np
from random import randint

import pytest

from mmgroup.mm_op import mm_sub_test_prep_pi_64
from mmgroup import mat24
from mmgroup_fast.mm_op_fast import mm_op_fast_op_all_autpl


def make_m1(delta, pi):
    perm = mat24.m24num_to_perm(pi)
    m1 = np.array(mat24.perm_to_autpl(delta, perm), dtype = np.uint32)
    return m1


def get_ref_op_autpl(m1):
    return mat24.op_all_autpl(m1)

def get_op_autpl(m1):
    a_out = np.zeros(2048, dtype = np.uint16)
    mm_op_fast_op_all_autpl(m1, a_out)
    return list(a_out)


def pi_testcases():
    testdata = [
        (0, 0),
    ] 
    for x in testdata: yield x  
    for i in range(80):
        yield randint(0, 0xfff), randint(0, mat24.MAT24_ORDER - 1)


@pytest.mark.mm_op
def test_prep_pi():
    print("Test function mm_sub_prep_pi... ")
    for delta, pi in pi_testcases():
        m1 = make_m1(delta, pi)
        a_ref = [x & 0x7fff for x in get_ref_op_autpl(m1)]
        a = [x & 0x7fff for x in get_op_autpl(m1)]
        ok = a_ref == a
        if not ok:
           for i in range(2048):
               if a_ref[i] != a[i]:
                 print("Error at index", i, hex(a_ref[i]), hex(a[i]))
                 break
           raise ValueError("Function vmat24_op_all_autpl failed")
    print("passed")



    