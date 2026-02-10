
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastMatrix
from mmgroup import MM0, MMV, MM
from mmgroup_fast.display import gcc_capabilities


MMV3 = MMV(3)
MMV15 = MMV(15)
ITERATIONS = 20000


try:
    from mmgroup.tests.test_mm_op.test_benchmark import MM_REDUCE_OPS
    from mmgroup.tests.test_mm_op.test_benchmark import FREQ
except:
    # Use your own lies in statistics if you can't cite an authority
    MM_REDUCE_OPS = {3:3.0, 15:(3.0 + 2 + 3.0/7.0)}
    FREQ = {'xy':1, 'p':16, 'l':17, 't':6.25}



def make_matrix():
    m = MMOpFastMatrix(3,4)
    for j in range(4):
        m.set_row(j, MMV3('R'))
    return m

def make_group_elements(length = 64):
    return [MM('r') for i in range(length)]



def bench(m,  operation = [], iterations = ITERATIONS):
    g = MM0(operation)
    t = m.mul_exp_bench(g, 1, iterations)
    return t / iterations

def bench_weights(frequencies, timings):
    t = 0.0
    for tag, f in frequencies.items():
        t += f * timings[tag] 
    return t   

def quot_ms(f, *args):
    t = f(*args)
    return t, "%9.6f" % (1000 * t)



@pytest.mark.bench
@pytest.mark.mm_op
def test_vector_op_bench():
    cap = gcc_capabilities()
    if not cap:
        cap = 'None'
    print("""
Benchmarking monster operations with GCC capabilities:
%s
All times are given in milliseconds.
""" % cap
    )
    runtimes = {}
    m = make_matrix()

    op = [('p', 23), ('d', 12745645)]        
    runtimes['p'], msg = quot_ms(bench, m, op)
    print ("p odd", msg)

    op = [('x', 1237), ('y', 567),]
    runtimes['xy'], msg = quot_ms(bench, m, op)
    print ("xy   ", msg)

    op = [('l', 2)]
    runtimes['l'], msg = quot_ms(bench, m, op)
    print ("l    ", msg)

    op = [('t', 2)]
    runtimes['t'], msg = quot_ms(bench, m, op)
    print ("t    ", msg)

    _ , msg = quot_ms(bench_weights, FREQ, runtimes)
    print ("MM   ", msg)







@pytest.mark.bench
@pytest.mark.mm_op
def test_vector_mul_bench(ntests = 50, length = 20):

    m = make_matrix()
    a = make_group_elements(length)
    t = sum([m.mul_exp_bench(g, 1, ntests) for g in a])
    t1 = 1.0e3 * t / ntests / len(a)
    S = "Runtime mmgroup_fast vector operation"
    print("\n%s: %.3f ms" % (S,t1))
    S = "Runtime mmgroup_fast group operation"
    print("%s: %.3f ms" % (S, 3 * t1))

    ntests = ntests // 2
    v3 = MMV3('R')
    v15 = MMV15('R')
    t3 = sum([v3.mul_exp(g, ntests, True).last_timing for g in a])
    t3 = 1.0e3 * t3 / ntests / len(a)
    t15 = sum([v15.mul_exp(g, ntests, True).last_timing for g in a])
    t15 = 1.0e3 * t15 / ntests / len(a)
    S = "Runtime mmgroup vector operation"
    print("%s mod 3: %.3f ms, mod 15: %.3f ms" % (S,t3,t15))
    tg = MM_REDUCE_OPS[3] * t3 + MM_REDUCE_OPS[15] * t15
    S = "Runtime mmgroup group operation"
    print("%s: %.3f ms" % (S,tg))
    print("Estimated speedup factor = %.2f" % (tg/(3 * t1)))








    