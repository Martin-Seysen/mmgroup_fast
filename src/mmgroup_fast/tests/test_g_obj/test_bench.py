
import time
import pytest

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



@pytest.mark.bench
@pytest.mark.mm_op
def test_MM_subgroup(verbose = 0):
    print("\nTest fast multiplication of temp. variables in subgroups of MM")
    for s, n in SUBGROUPS:
        tt = 1000 * timing_fast_g(s, n)
        s1 = s if isinstance(s, str) else "MM"
        print("subgroup %4s: %7.4f ms" % (s1, tt))


