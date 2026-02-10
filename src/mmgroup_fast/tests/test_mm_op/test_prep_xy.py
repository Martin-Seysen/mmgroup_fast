from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import numpy as np
from random import randint

import pytest

from mmgroup.mm_op import mm_sub_test_prep_xy

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




@pytest.mark.mm_op
def test_bench_prep_xy(n = 10000):
    print("Benchmark function mm_sub_prep_xy... ")
    f, e = randint(0, 0x1fff), randint(0, 0x1fff)
    eps = randint(0x800, 0xfff)

    import time
    t = time.time()
    tbl = np.zeros(2048, dtype = np.uint32)
    x = 0
    for i in range(n):
        mm_sub_test_prep_xy(f, e, eps, 0, tbl)
        #x += 1
    t =  time.time() - t
    t1 = t * 10e6 / n
    print("Runtime of function mm_sub_test_prep_xy: %.3f us" % t1)


    