
import time
import pytest

import numpy as np

from mmgroup.mm_reduce import mm_compress_pc_expand_int
from mmgroup_fast.mm_op_fast import MMOpFastG


N = 32
MASK = N-1
SUBGROUPS = [
   (8,50), ("G_x0", 5000), 
   ("N_x0", 50000), ("N_0", 50000), ("G_3", 100)
]

def timing_fast_g(s, ntests):
    lg = []
    for i in range(N+1):
         lg.append(MMOpFastG('r', s)) 
         lg[-1].as_int()   
    t = time.process_time()
    for i in range(ntests):
        j = i & MASK
        a = (lg[j] * lg[j+1]).as_int()
    dt = time.process_time() - t
    return dt / ntests


def timing_expand_int(ntests):
    a = np.zeros((N,4), dtype = np.uint64)
    for i in range(N):
         a[i] = MMOpFastG('r').as_int_array()
         #print([hex(x) for x in (a[i])])
    mm = np.zeros(80, dtype = np.uint32)
    t = time.process_time()
    for i in range(ntests):
        mm_compress_pc_expand_int(a[i & MASK], mm, 80)
    dt = time.process_time() - t
    return dt / ntests




@pytest.mark.bench
@pytest.mark.mm_op
def test_MM_subgroup(verbose = 0):
    print("\nTest fast multiplication of temp. variables in subgroups of MM")
    for s, n in SUBGROUPS:
        tt = 1000 * timing_fast_g(s, n)
        s1 = s if isinstance(s, str) else "MM"
        print("subgroup %4s: %7.4f ms" % (s1, tt))
    tt = 1000 * timing_expand_int(5000)
    print("Expand int to MM: %7.4f ms" % tt)

