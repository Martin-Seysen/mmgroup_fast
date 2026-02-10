
import numpy as np
from random import randint

import pytest

import mmgroup
from mmgroup import MM0, MMV
from mmgroup_fast.mm_op_fast import mmv_fast_test_prep_pi_64
from mmgroup_fast.mm_op_fast import mmv_fast_test_debug_pi_64
from mmgroup.mm_op import mm_sub_test_prep_pi_64




V255 = MMV(255)
MMV_TAG_T = 0x8000000
def ref_preimage_tag_T_op_pi(row, d, pi, doublecheck = True):
    assert 0 <= row < 759
    assert 0 <= d < 2048 and 0 <= pi < mmgroup.MAT24_ORDER
    vdata = [MMV_TAG_T + (row << 14) + 0x101 * i + 1 for i in range(64)]
    g = MM0([('d', d), ('p', pi)])
    v = V255('S', vdata).mul_exp(g, -1).as_sparse()
    v = np.sort(v)
    #print([hex(x) for x in v])
    sign = (v[0] >> 7) & 1
    preimage = (v[0] >> 14) & 0x7ff
    v &= 0xff
    vsign = v & 0x80
    v ^= (vsign << 1) - (vsign >> 7)
    v = v - 1
    assert set(v) == set(range(64))
    v_inv = [None] * 64
    for i in range(64):
        v_inv[v[i]] = i
    if doublecheck:
        a = np.zeros(759 * 7, dtype = np.uint32)
        mm_sub_test_prep_pi_64(d, pi, a)
        a0 = a[row * 7 : row * 7 + 7]
        assert a0[0] == (sign << 12) + preimage
        for i in range(1, 7):
            assert a0[i] ==  v_inv[(1 << i) - 1]
    return sign, preimage, v_inv    


def preimage_tag_T_op_pi(row, d, pi):
    assert 0 <= row < 759
    assert 0 <= d < 2048 and 0 <= pi < mmgroup.MAT24_ORDER
    a = np.zeros(64, dtype = np.uint8)
    res =  mmv_fast_test_prep_pi_64(d, pi, row, a)
    if res < 0:
        ERR = "Function test_prep_op_pi_row64 returns %d"
        raise ValueError(ERR % res)
    return res >> 12, res & 0xfff, [int(x) for x in list(a)]
     


def one_test_preimage_tag_T_op_pi(row, d, pi):
    ref_result = ref_preimage_tag_T_op_pi(row, d, pi, doublecheck=True)
    result = preimage_tag_T_op_pi(row, d, pi)
    ok = ref_result == result
    if not ok:
        print("row = %d, d = 0x%03x, pi = %d" % (row, d, pi))
        r_sign, r_preimage, r_data = ref_result
        sign, preimage, data = result
        print("sign, preimage expected:", r_sign, r_preimage)
        print("sign, preimage obtained:", sign, preimage)
        print("shuffle, expected and obtained")
        for i in range(0, 64, 16):
           print("exp:", np.array(r_data[i:i+16], dtype = np.uint8))
           print("obt:", np.array(data[i:i+16], dtype = np.uint8))
        raise ValueError("Test of preimage_tag_T_op_pi failed")       

def preimage_tag_T_op_pi_testdata():
    for i in range(100):
        row = randint(0, 758)
        d = randint(0, 2047)
        pi = randint(0, mmgroup.MAT24_ORDER - 1)
        yield row, d, pi



def purged(a):
    # return a ^ (((a & 0x80) << 1) - ((a & 0x80) >> 7))
    return a

@pytest.mark.mm_op
def test_test_preimage_tag_T_op_pi():
    print("Testing preimage_tag_T_op_pi")
    for row, d, pi in preimage_tag_T_op_pi_testdata():
        #one_test_preimage_tag_T_op_pi(row, d, pi)
        ref_a = np.zeros(759 * 7, dtype = np.uint32)
        mm_sub_test_prep_pi_64(d, pi, ref_a)
        a = np.zeros(759 * 7, dtype = np.uint32)
        mmv_fast_test_prep_pi_64(d, pi, a)
        ok =  (a == ref_a).all()
        if not ok:
           for i in range(759):
               ref_b, b = ref_a[i*7:i*7+7] , a[i*7:i*7+7]
               if not (ref_b == b).all():
                   print("Error in entry", i)
                   print(ref_b)
                   print(b)
                   break;
           raise ValueError("Test of preimage_tag_T_op_pi failed")
       
        dbg = np.zeros(128, dtype = np.uint8)
        result = mmv_fast_test_debug_pi_64(row, d, pi, dbg)
        if result < 0:
            continue
        dbg_obt, dbg_exp = dbg[:64], dbg[64:]
        if not (dbg_obt == dbg_exp).all():
            for i in range(0, 64, 16):
                print("obt:", purged(dbg[i:i+16]))
                print("exp:", purged(dbg[64+i:64+i+16]))
            err = "Function mmv_fast_test_debug_pi_64 failed"
            raise ValueError(err)
    if result < 0:
        print("Function mmv_fast_test_debug_pi_64 is not implemented")


