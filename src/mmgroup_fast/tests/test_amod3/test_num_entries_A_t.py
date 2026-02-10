import os
from collections import defaultdict
import numpy as np

import pytest

from mmgroup import MMV, MM0, MM, MMSpace, Xsp2_Co1, XLeech2
from mmgroup.generators import gen_leech2_reduce_type4
from mmgroup.axes import Axis, BabyAxis
from mmgroup_fast.mm_op_fast import MMOpFastMatrix, MMOpFastAmod3
from mmgroup_fast.mm_op_fast import mm_axis3_fast_orbit_dict
from mmgroup_fast.dev.mm_op_axis_mod3.order_vector import std_matrix

MMV3 = MMV(3)


GOOD_AXIS_TYPES = [
      [], [], [28], [253],         [147], [136], [231], [171],
      [78], [8], [181, 1], [146,26],  [145]
]

ORBIT_DICT = mm_axis3_fast_orbit_dict()


def vector_count_BpmC(v):
    B = v['B'] % 3
    C = v['C'] % 3
    p = np.count_nonzero((B+C) % 3)
    m = np.count_nonzero((B+15-C) % 3)
    return int(p) >> 1, int(m) >> 1



@pytest.mark.mm_amod3
def test_num_entries_A_t():
    for i in range(10):
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in range(4):
            v = MMV3('R')
            #vectors[j] = v
            a.set_row(j, v)
            t1, t2 = a.num_entries_A_t(j)
            b, c = v['B'], v['C']
            t1_ref, t2_ref = vector_count_BpmC(v)
            assert t1 == t1_ref 
            assert t2 == t2_ref 
    


def hexx(lst):
    return "[" + " ".join(["%08x" % x for x in lst]) + "]"


def mm_c_data(v4):
    a = np.zeros(6, dtype = np.uint32)
    status = gen_leech2_reduce_type4(v4, a);
    assert status >= 0
    return a[:status], MM0('a', a[:status])

def mm_t_data(e):
    assert 1 <= e <= 2
    a = np.zeros(1, dtype = np.uint32)
    a[0] = 0xD0000003 - e
    return a, MM0('a', a)


def reduce_mm(test, g, baby=True, check_C = True, verbose = 0):
    m = std_matrix()
    ax = Axis(g)
    if baby:
        baby = Axis(BabyAxis()) * g
    m.mul_exp(g)
    result = np.zeros(0, dtype = np.uint32)
    assert m.row_as_mmv(0) == ax.v15 % 3
    round = 1
    while True:
        if verbose:
            print("test", test, ", round", round)
        a = MMOpFastAmod3(m, 0, 1)
        ax_type, c =  a.find_v4()
        if verbose > 1:
            print("ax_type", ax_type, ORBIT_DICT[ax_type])
        if ax_type == '2A':
            v2 = c
            break
        h_data, h = mm_c_data(c)
        result = np.append(result, h_data)
        m = m.mul_exp(h)
        ax *= h
        if baby:
            baby *= h
        # assert m.row_as_mmv(0) == ax.v15 % 3
        if verbose > 1:
            print([ax.axis_type(e) for e in range(3)])
        e = m.find_exp_t(0, ax_type)
        if verbose > 1:
            print("e=", e)
        h_data, h = mm_t_data(e)
        result = np.append(result, h_data)
        m = m.mul_exp(h)
        ax *= h
        if baby:
            baby *= h
        round += 1
        if (round > 6):
            raise valueError("WTF")
    assert m.row_as_mmv(0) == ax.v15 % 3
    assert ax.axis_type() == '2A'
    assert XLeech2(ax.g_axis) == XLeech2(v2)
    if not baby:
        return result
    if verbose > 1:
        print(["%08x" % x for x in result])
    round = 11
    while True:
        if verbose:
            print("test", test, ", round", round, ", v2 =", hex(v2))
        a = MMOpFastAmod3(m, 1)
        ax_type, c =  a.find_v4(v2)
        if verbose > 1:
            print("ax_type", ax_type, ORBIT_DICT[ax_type], hex(v2))
        if ax_type == '2A':
            v2a = c
            break
        h_data, h = mm_c_data(c)
        result = np.append(result, h_data)
        m = m.mul_exp(h)
        baby *= h
        v2 = (XLeech2(v2) * h).ord
        # assert m.row_as_mmv(0) == ax.v15 % 3
        if verbose > 1:
            print([baby.axis_type(e) for e in range(3)])
        e = m.find_exp_t(1, ax_type)
        if verbose > 1:
            print("e=", e)
        h_data, h = mm_t_data(e)
        result = np.append(result, h_data)
        m = m.mul_exp(h)
        baby *= h
        v2 = (XLeech2(v2) * h).ord
        round += 1
        if (round > 16):
            raise valueError("WTF")
    v4 = (XLeech2(v2) * XLeech2(v2a)).ord
    if v4 & 0xffffff:
        h_data, h = mm_c_data(v4)
        result = np.append(result, h_data)
        v4 = (XLeech2(v4) * h).ord
        h_data, h = mm_t_data(1 + ((v4 >> 24) & 1))
        result = np.append(result, h_data)
        v4 = (XLeech2(v4) * h).ord
    assert v4 == 0x1000000
    assert (g * MM('a', result)).in_G_x0()

    if check_C:
        m_c = std_matrix()
        m_c.mul_exp(g)
        c_result = m_c.reduce_axes()
        if verbose > 2:
            print(hexx(c_result))
            print(hexx(result))
        assert (result == c_result).all()
    return result

def iter_test_elementes(ncases = 10):
    types = Axis.representatives().keys()
    for n in range(ncases):
        for t in types:
            g0 = Axis.representatives()[t].g
            yield g0 * MM(Xsp2_Co1('r'))
        yield MM('r', 'M')

def do_test_std_axes():
    from mmgroup.axes import Axis, BabyAxis
    m = std_matrix()
    assert m.row_as_mmv(0) == Axis().v15 % 3
    assert m.row_as_mmv(1) == BabyAxis().v15 % 3



@pytest.mark.mm_amod3
def test_reduce_mm(ncases = 1, processes = 16, verbose = 0):
    do_test_std_axes()
    n_cpu = max(1, min(16, processes, os.cpu_count()))
    if n_cpu == 1:
        for n, g in enumerate(iter_test_elementes(ncases)):
            if verbose:
                print("Test", n+1)
            a1 = reduce_mm(n+1, g, 1, 1, verbose) 
    else:
        #print("\nn_cpu=", n_cpu)
        from multiprocessing import Pool             
        with Pool(processes = n_cpu) as pool:
            results = pool.starmap(reduce_mm,
                enumerate(iter_test_elementes(ncases)))



def fast_test_reduce(g):
    m = std_matrix()
    m.mul_exp(g)
    c_result = m.reduce_axes()
    h = g * MM('a', c_result)
    assert h.in_G_x0()



def count_tau(a):
    b = (a >> 28) & 7 == 5
    return np.count_nonzero(b)

@pytest.mark.bench
@pytest.mark.mm_amod3
def test_bench_fast_reduce_mm(ncases = 10):
    import time
    Mat0 = std_matrix()
    NDATA = 100
    NTESTS = 1000
    matrices = []
    max_tau, sum_tau = 0, 0.0
    for i in range(NDATA):
        matrices.append(Mat0.copy().mul_exp(MM0('r',8)))
        a = matrices[-1].copy().reduce_axes()
        n_tau = count_tau(a)
        max_tau = max(n_tau, max_tau)
        sum_tau += n_tau
    print("\nBenchmark for reduction of an axis matrix")
    t = time.time()
    for j in range(NTESTS):
        m = matrices[i % NDATA].copy()
        m.reduce_axes()
    t =  time.time() - t
    print("Average run time is %.3f ms" % (1000*t/NTESTS))
    print("Number of triality elements: max = %d, ave = %.2f" 
            % (max_tau, sum_tau / NDATA))





