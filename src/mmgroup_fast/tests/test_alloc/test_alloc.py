import os
from time import time
from random import randint, shuffle
import pytest
from mmgroup_fast import FastBuffer
import numpy as np

@pytest.mark.mm_op
def test_fast_buffer1(verbose = 0):
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


@pytest.mark.mm_op
@pytest.mark.bench
def test_fast_buffer2(verbose = 0):
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




def alloc_thread(nbuffers, nsteps):
    def alloc_and_use_buffer():
        size = randint(5,7) << randint(12, 13)
        b = FastBuffer(size)
        assert len(b) == size
        b[0] = b[len(b)-1] = 3
        index = randint(0, size-1)
        value =  randint(0, 0xff)
        b[index] = value
        buffer_entry =  b, index, value
        return buffer_entry
    def check_buffer(buffer_entry):
        b, index, value = buffer_entry
        assert b[index] == value
    data = []
    for i in range(nbuffers):
        data.append(alloc_and_use_buffer())
    for i in range(nsteps):
        data_new = []
        shuffle(data)
        for i in range(nbuffers//2+1):
            check_buffer(data.pop())
            data_new.append(alloc_and_use_buffer())
        data += data_new

    while len(data):
        data.pop()


def do_test_fast_buffer_threads(max_threads, nbuffers, nsteps, verbose = 1):
    """Stress test for memory allocator

    Depending on the OS and the hardware, we use up to ``max_threads``,
    threads; and in each thread we allocate upto ``nbuffers`` buffers
    concurrently. After allocaing the buffers, we free half of them
    and allocacte the same numbers buffers again. In each thread, this
    procedure is repeated ``nsteps`` times. The buffers to be freed are
    selected  at random. An allocated buffer may have size up
    to 65536 bytes.

    We expect an actual parallel execution of these threads
    in Python >= 3.14 only.
    """
    nthreads = max(2, min(2 * os.cpu_count() // 3 + 1, max_threads))
    if verbose:
        print("\nTest allocator. Number of threads: %d" % nthreads)
    try:
        import threading
    except:
        print("\nThreading not supported")
        return
    FastBuffer.gc()
    threads = []
    for i in range(nthreads):
        t = threading.Thread(target=alloc_thread, args = (nbuffers, nsteps))
        threads.append(t)

    GRAN = max(1, nthreads//3)
    for i, t in enumerate(threads):
        t.start()
        if i % GRAN == 0 and verbose:
            FastBuffer.statistics()
    if verbose:
        FastBuffer.statistics()

    # Wait for all threads to finish
    for t in threads:
        t.join()
    if verbose:
        FastBuffer.statistics()
    FastBuffer.gc()
    if verbose:
        FastBuffer.statistics()



def test_fast_buffer_threads(verbose = 0):
    do_test_fast_buffer_threads(4, 100, 50, verbose)



MAX_THREADS = 30
NBUFFERS = 1024
NSTEPS = 100

@pytest.mark.slow
@pytest.mark.mm_op
def test_fast_buffer_threads(verbose = 0):
    """This test is slow and strains the memory considerably.

    Here the number 2**17 * MAX_THREADS * NSTEPS should be smaller
    than the amount of physical RAM on the user's computer.
    """
    do_test_fast_buffer_threads(MAX_THREADS, NBUFFERS, NSTEPS, verbose)
