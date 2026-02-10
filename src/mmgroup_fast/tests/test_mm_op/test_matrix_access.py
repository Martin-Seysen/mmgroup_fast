
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastMatrix
from mmgroup import MM0, MMV, MM


MMV3 = MMV(3)


def shuffled_range(n):
    l = list(range(n))
    shuffle(l)
    return l

@pytest.mark.mm_op
def test_matrix_access():
    for i in range(100):
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in shuffled_range(4):
            v = MMV3('R')
            vectors[j] = v
            a.set_row(j, v)
            # v['T', 1, 3] = 2 # This creates an error here
        for j in shuffled_range(4):
            w = a.row_as_mmv(j)
            assert w == vectors[j]

MMV_OFS_X =  50880
MMV_OFS_Z = 116416

def g_testdata():
    yield MM0('y_6f0h*x_1a7dh*d_9b9h')
    yield MM0('x', 'r') * MM0('y', 'r') * MM0('d', 'o') 
    for exp in (0, 1, 2):    
        yield MM0('l', exp)
    for exp in range(3):    
        yield MM0('t', exp)
    for e in range(0, 0x2000, 0x800):    
        for f in range(0, 0x2000, 0x800):    
            yield MM0([('y', f), ('x', e), ('d', 0)])
    for i in range(50):
        eps = randint(0, 0xfff)
        e = randint(0, 3) << 11
        f = randint(0, 3) << 11
        yield MM0([('y', f), ('x', e), ('d', eps)])
    for i in range(100):
        eps = randint(0, 0xfff)
        e = randint(0, 0x1fff)
        f = randint(0, 0x1fff)
        yield MM0([('y', f), ('x', e), ('d', eps)])
    for i in range(100):
        d = randint(0, 0xfff)
        yield MM0([('d', d), ('p', 'r')])



def analyze_vector_diff(w, w_ref, test_num = 0, test_row = 0, text = ""):
    for tag in  "ABCXZYT":
        ok =  (w_ref[tag] == w[tag]).all()
        if not ok:
            row = None
            for k0 in range(len(w[tag])):
                if not (w_ref[tag][k0] == w[tag][k0]).all():
                   row_ref, row = w_ref[tag][k0], w[tag][k0]
                   for k1, r0 in enumerate(row):
                       if r0 != row_ref[k1]:
                           break
                   break
            print("\nTest %d, row %d, entry %s,%d,%d is bad" % 
                (test_num, test_row, tag, k0, k1))
            if (text):
                print(text)
            if row is not None:
                print("obtained:")
                print(row)
                print("expected:")
                print(row_ref)
            return False
    return True


@pytest.mark.mm_op
def test_matrix_op_pi(verbose = 0):
    OFS = MMV_OFS_Z
    for i, pi in enumerate(g_testdata()):
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in shuffled_range(4):
            v = MMV3('R')
            vectors[j] = v
            a.set_row(j, v)
        a_in_dump = a.dump()
        if verbose:
            print("Test op %s (mod 3) on vector" % str(pi))
        a.mul_exp(pi)
        a_dump = a.dump()
        for j in shuffled_range(4):
            w = a.row_as_mmv(j)
            w.check()
            w_ref = vectors[j] * pi
            text = "Op = %s , current = %s" % (pi, a_dump.current)
            ok = analyze_vector_diff(w, w_ref, i+1, j, text)
            if not ok:
                raise ValueError("Test failed")


@pytest.mark.mm_op
def test_matrix_op_any(verbose = 0):
    for i in range(10):
        g = MM('r')
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in shuffled_range(4):
            v = MMV3('R')
            vectors[j] = v
            a.set_row(j, v)
        a.mul_exp(g)
        a_dump = a.dump()
        for j in shuffled_range(4):
            w = a.row_as_mmv(j)
            w1 = vectors[j]  * g
            w.check()
            w1.check()
            text = "Op = %s , current = %s" % (g, a_dump.current)
            ok = analyze_vector_diff(w, w1, i+1, j, text)
            if not ok:
                raise ValueError("Test failed")





@pytest.mark.mm_op
def test_bench_matrix_op_pi(ntests = 5000):
    d = randint(1, 0x7ff)
    g = MM0([('d', d), ('p', 'r')])
    a = MMOpFastMatrix(3,4)
    for j in range(4):
        v = MMV3('R')
        a.set_row(j, v)
    t = a.mul_exp_bench(g, 1, ntests)
    t1 = (1.0e6 * t) / ntests
    print("\nRuntime for generator x_pi: %.3f us" % t1)





    