import numpy as np
from mmgroup.mat24 import mul_perm, inv_perm
from random import shuffle, randint
from mmgroup_fast.mm_op_fast import mmgroup_fast_test_prep_shuffle24
from mmgroup_fast.mm_op_fast import mmgroup_fast_test_shuffle24
from mmgroup_fast.display import gcc_capabilities



import pytest



def perm_to_shuffle16(pi):
    pi_out = list(range(24))
    i0, i1 = 0, 15
    while i0 < 8:
        if pi[i0] >= 16:
            while pi[i1] >= 16:
                i1 -= 1
                if i1 < 7: raise ValueError("WTF1")
            pi_out[i0], pi_out[i1] = pi_out[i1], pi_out[i0]
            i1 -= 1
        i0 += 1
        if i1 < 7: raise ValueError("WTF2")
    return pi_out 


def lrange(*args):
    return list(range(*args))    


def mul_perms(*p):
    if len(p) > 1:
        return mul_perm(p[0], mul_perms(*p[1:]))
    elif len(p) == 1:
        return p[0][:]
    else:
        return lrange(24)


def perm_to_xch_big(pi):
    pi = pi[:]
    pi_out = list(range(24))
    ok = False
    while not ok:
        for i in range(16):
            if pi[i] >= 16:
                j = pi[i] - 16
                pi[i], pi[j] = pi[j], pi[i]
                pi_out[i], pi_out[j] = pi_out[j], pi_out[i]
        ok = True
        for i in range(8):
            ok = ok and  pi[i] == i + 16 
        # if not ok: print("!", end = "") 
    return pi_out 



pi_fix = lrange(16, 24) + lrange(8,16) + lrange(0, 8) 

def rand_pi():
    pi = lrange(24)
    shuffle(pi)
    return pi



def split_pi(pi):
    pi0 = perm_to_shuffle16(pi)
    assert pi0[16:] == lrange(16,24)
    pi_strip = mul_perm(pi0, pi)
    #print(pi0)
    #print(pi_strip)
    pi_strip = mul_perm(pi_fix, pi_strip)
    #print(pi_strip)
    pi1 = perm_to_xch_big(pi_strip)
    assert pi1[16:] == lrange(16,24)
    pi_strip = mul_perm(pi1, pi_strip)
    #print(pi1)
    #print(pi_strip)
    pi_strip = mul_perm(pi_fix, pi_strip)
    #print(pi_strip)
    pi2 = inv_perm(pi_strip)
    pi_strip = mul_perm(pi2, pi_strip)
    assert pi_strip[16:] == lrange(16,24)
    #print(pi_strip)
    return pi0, pi1, pi2


def ref_prep_shuffle24(pi, mode):
    assert mode in [0,1]
    if mode == 0:
        return list(inv_perm(pi[:24])) + lrange(24,32)
    if mode == 1:
        pi0, pi1, pi2 = split_pi(pi)
        slack = lrange(24, 32)
        return pi0 + slack + pi1 + slack + pi2 + slack


def shuffle24(data, p1, p2, p3):
    def builtin_shuffle(data, pi):
        return [data[pi[i]] for i in range(16)] + data[16:]
    data = builtin_shuffle(data, p1)
    data = data[16:24] + data[8:16] + data[0:8]
    data = builtin_shuffle(data, p2)
    data = data[16:24] + data[8:16] + data[0:8]
    data = builtin_shuffle(data, p3)
    return data

def permute24(data, pi):
    l = [None] * 24
    for i in range(24):
        l[pi[i]] = data[i]
    return l


def display_mode():
    capa = gcc_capabilities()
    print("\nGCC capablities:", capa if capa else "None")
    pi = np.array(range(24), dtype = np.uint8)
    prep = np.zeros(128, dtype = np.uint8)
    res = mmgroup_fast_test_prep_shuffle24(pi, prep)
    assert res in [0, 1]
    print("MM_OP_FAST_SPLIT_SHUFFLE24 =", res)


@pytest.mark.mm_op
def test_prep_shuffle_pi24(verbose = 0):
    display_mode()
    for i in range(100):
        pi = rand_pi()
        #print(pi)
        p0, p1, p2 = split_pi(pi)
        data = [randint(0, 255) for j in range(24)]
        if verbose:
            print("data=", data)
        np_data = np.array(data, dtype = np.uint8)
        assert pi == inv_perm(mul_perms(p2, pi_fix, p1, pi_fix, p0))
        assert permute24(data[:], pi) == shuffle24(data[:], p0, p1, p2)
        np_pi = np.array(pi, dtype = np.uint8)
        np_prep_obt = np.zeros(96, dtype = np.uint8)
        res = mmgroup_fast_test_prep_shuffle24(np_pi, np_prep_obt)
        assert res in [0,1]
        prep_exp = ref_prep_shuffle24(pi, res)
        prep_obt = [x for x in np_prep_obt[:len(prep_exp)]]
        assert prep_obt ==  prep_exp

        
        mmgroup_fast_test_shuffle24(np_data, np_prep_obt)
        c_permuted_data = [x for x in np_data]
        assert c_permuted_data == permute24(data, pi)
        
       
         





