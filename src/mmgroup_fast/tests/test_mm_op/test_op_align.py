
import sys
import numpy as np
from random import randint, shuffle, sample

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastMatrix
from mmgroup import MM0, MMV, MM


MMV3 = MMV(3)


def shuffled_range(n):
    l = list(range(n))
    shuffle(l)
    return l


def g_testdata():
    TAGS = "dxyptl"
    for i in range(20):
        yield MM0('d', 'o')
    for i in range(20):
        tag = sample(TAGS, 1)[0]
        yield MM0(tag, 'r')

@pytest.mark.alignment
def test_matrix_op(verbose = 0):
    for i, pi in enumerate(g_testdata()):
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in shuffled_range(4):
            v = MMV3('R')
            vectors[j] = v
            a.set_row(j, v)
        # a_in_dump = a.dump()
        if verbose:
            print("\nTest %d: op %s (mod 3) on vector" % (i,str(pi)))
            sys.stdout.flush()
        a.mul_exp(pi)
        if verbose > 1:
            print("multiplication done")
            sys.stdout.flush()
        continue
        # a_dump = a.dump()
        for j in shuffled_range(4):
            w = a.row_as_mmv(j)
            w.check()
            w1 = vectors[j] * pi
            assert w == w1
 


    