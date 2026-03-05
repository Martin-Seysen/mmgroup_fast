import os
from collections import defaultdict
import random
import time

import numpy as np
import pytest

from mmgroup import MMV, MM0, MM, MMSpace, Xsp2_Co1, XLeech2
from mmgroup.mm_reduce import mm_reduce_M
from mmgroup_fast.mm_op_fast import MMOpFastMatrix

MMV3 = MMV(3)

def hexx(lst):
    return "[" + " ".join(["%08x" % x for x in lst]) + "]"



def std_matrix():
    m = MMOpFastMatrix(3, 4, 1)
    m.set_vstd()
    return m


class ReduceSamples():
    N_SAMPLES = 64
    _fast_samples = None
    _std_samples = None

    @classmethod
    def fast_samples(cls):
        if cls._fast_samples is not None:
            return cls._fast_samples
        cls._fast_samples = []
        for i in range(cls.N_SAMPLES):
            g = MM0('r')
            m = std_matrix()
            m.mul_exp(g)
            reduced = m.reduce_v_g(check=1)
            cls._fast_samples.append(reduced)
        return cls._fast_samples

    @classmethod
    def std_samples(cls):
        if cls._std_samples is not None:
            return cls._std_samples
        cls._std_samples = []
        for a in cls.fast_samples():
            cls._std_samples.append(MM('a', a).reduce().mmdata)  
        return cls._std_samples

    def sample_fast_elements(self, n_elements):
        data = []
        while len(data) < n_elements:
            data += self.fast_samples()[:]
        return data[:n_elements]

    def sample_fast_pairs(self, n_pairs):
        pairs = [(i,j) for i in range(self.N_SAMPLES)
            for j in range(self.N_SAMPLES) if i != j] 
        p =  random.sample(pairs, n_pairs)
        fs = self.fast_samples()
        return [(fs[i], fs[j]) for i, j in p]

    def sample_std_pairs(self, n_pairs):
        pairs = [(i,j) for i in range(self.N_SAMPLES)
            for j in range(self.N_SAMPLES) if i != j] 
        p =  random.sample(pairs, n_pairs)
        fs = self.std_samples()
        return [(fs[i], fs[j]) for i, j in p]
  

def count_tau(a):
    b = (a >> 28) & 7 == 5
    return np.count_nonzero(b)

def timings_fast_mul_reduce_mm(ncases = 100):
    ncases = max(10, min(ncases,2000))
    samples = ReduceSamples()
    timings = []
    n_tags = [0.0] * 8
    pairs = samples.sample_fast_pairs(ncases)
    for g1, g2 in pairs:
        t0 = time.time()
        m = std_matrix()
        m.mul_exp( g1)
        m.mul_exp( g2)
        g = m.reduce_v_g()
        t1 = time.time()
        timings.append(t1 - t0)
        for x in g:
            n_tags[(x >> 28) & 7] += 1
    return timings, n_tags

def timings_fast_mul_g_mm(ncases = 100):
    ncases = max(10, min(ncases,2000))
    samples = ReduceSamples()
    timings = []
    elements = samples.sample_fast_elements(ncases)
    for g1 in elements:
        t0 = time.time()
        m = std_matrix()
        m.mul_exp( g1)
        t1 = time.time()
        timings.append(t1 - t0)
    return timings




def timings_std_mul_reduce_mm(ncases = 100):
    ncases = max(10, min(ncases,2000))
    samples = ReduceSamples()
    timings = []
    n_tags = [0.0] * 8
    pairs = samples.sample_std_pairs(ncases)
    g = np.zeros(128, dtype = np.uint32)
    for g1, g2 in pairs:
        t0 = time.time()
        a = np.concatenate((g1, g2), dtype = np.uint32)
        len_g = mm_reduce_M(a, len(a), 0, g)
        t1 = time.time()
        timings.append(t1 - t0)
        for x in g[:len_g]:
            n_tags[(x >> 28) & 7] += 1
    return timings, n_tags


def stat(t):
    assert isinstance(t,list)
    assert isinstance(t[0], float)
    min_t = min(t)
    max_t = max(t)
    n = len(t)
    avg = sum(t) / n
    var = sum([(t_i - avg)**2 for t_i in t]) / (n - 1)
    sigma = var**0.5
    return n, avg, sigma, min_t, max_t


def display_tags(n, n_tags):
    MM_TAGS = " dpxyTl?"
    data = zip(MM_TAGS, [x / n for x in n_tags])
    assert n_tags[7] == 0, n_tags
    freq = ["%s:%.2f" % (tag, i) for tag, i in data]
    print("Tags per element:",  ", ".join(freq[1:7]))  




@pytest.mark.bench
@pytest.mark.mm_amod3
def test_bench_timings_fast_reduce_mm(ncases = 100):
    t, n_tags = timings_fast_mul_reduce_mm(ncases)
    n, avg, sigma, min_t, max_t = stat(t)
    print("\nRuntime for fast multipliction: %.3f ms +- %.3f, max = %.3f ms"
        ", %d tests" %
        (avg * 1000, sigma * 1000, max_t * 1000, len(t)))
    display_tags(n, n_tags)
    tm = timings_fast_mul_g_mm(ncases)
    n_m, avg_m, sigma_m, _1, _2 = stat(tm)
    print("Runtime for fast vector multipliction: %.3f ms +- %.3f" %
        (avg_m * 1000, sigma_m * 1000))

    t1, n_tags1  = timings_std_mul_reduce_mm(ncases // 3 + 1)
    n1, avg1, sigma1, min_t1, max_t1= stat(t1)
    print("Runtime for  std multipliction: %.3f ms +- %.3f, max = %.3f ms"
        ", %d tests" %
        (avg1 * 1000, sigma1 * 1000, max_t1 * 1000, len(t1)))
    display_tags(n1, n_tags1)
    print("Fast reduction is %.3f times faster than standard reduction"
         % (avg1 / avg))

