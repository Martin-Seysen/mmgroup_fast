from time import time
import pytest
from mmgroup_fast import FastBuffer
import numpy as np

@pytest.mark.mmm
def test_fast_buffer1(verbose = 1):
    if verbose:
        print("\nStart test buffer 1")
    BUFSIZE = 32000
    NBUFFERS = 20000
    FastBuffer.gc()
    data = [FastBuffer(BUFSIZE) for i in range(NBUFFERS)]
    for buf in data:
        for i in range(0, BUFSIZE, 20):
            buf[i] = 1
        buf[BUFSIZE-1] = 2
    del buf
    i = 0
    while len(data):
        data.pop()
        if i % 1000 == 0 and verbose:
            FastBuffer.statistics()
            pass
        i += 1
    if verbose:
        FastBuffer.statistics()
    FastBuffer.gc()
    if verbose:
        FastBuffer.statistics()
        print("test passed")



@pytest.mark.mmm
def test_fast_buffer2(verbose = 1):
    if verbose:
        print("\nStart test buffer 2")
    NTESTS = 10000
    BUFSIZE = 250000
    t  = time()
    for i in range(NTESTS):
        
        buf = FastBuffer(BUFSIZE)
        #buf[0] = 1
        #buf[BUFSIZE-1] = 2
        del buf
    dt  = time() - t
    print("\nTime for %d fast buffer allocations: %.3f ms" %
         (NTESTS, 1000*dt))
    t  = time()
    for i in range(NTESTS):
        buf = np.empty(BUFSIZE, dtype = np.uint8)
        #buf[0] = 1
        #buf[BUFSIZE-1] = 2
        del buf
    dt  = time() - t
    print("Time for %d numpy array allocations: %.3f ms" %
         (NTESTS, 1000*dt))
    FastBuffer.gc()
