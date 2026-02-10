
from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import sys
import os
import collections
import re
import warnings
from random import randint
import pytest

import numpy as np

from mmgroup import mat24 as m24
from mmgroup.tests.spaces.sparse_mm_space import SparseMmV
from mmgroup.tests.groups.mgroup_n import MGroupNWord
from mmgroup_fast.mm_op_fast import mmv_fast_test_prep_xy
from mmgroup_fast.mm_op_fast import mmv_fast_read_table_perm64_xy





p = 241
rep_mm = SparseMmV(p)
grp = MGroupNWord


def vector_data(v):
    data = v.as_tuples()
    assert len(data) == 1
    return data[0]


def as_suboctad(v1, d):
    c = m24.ploop_cap(v1, d)
    return m24.cocode_to_suboctad(c, d) & 0x3f


#234567890123456789012345678901234567890123456789012345678901234567890
def op_xy(v, eps, e, f):
    """Multiply unit vector v with group element

    This function multplies a (multiple of a) unit vector v
    with the group element

        g  =  d_<eps> * (x_<e>)**(-1)  * (y_<f>)**(-1) .
   
    It returns the vector w = v * g. This function uses the same 
    formula for calculating  w = v * g,  which is used in the 
    implementation of the monster group for computing
    v = w * g ** (-1).

    Input vector v  must be given as a tuple as described in class
    mmgroup.structures.abstract_mm_rep_space.AbstractMmRepSpace.
    Output vector is returned as a tuple of the same shape.

    This function is equivalent to the corresponding function in
    module mmgroup.tests.test_space. But here we use a method
    that is more suitable for vectorization. 
    """
    assert len(v) == 4
    value, tag, d, j = v
    parity_eps = (eps >> 11) & 1  # parity of eps

    ker_table_yx = [0, 0x1000, 0x1800, 0x800];
    e ^= ker_table_yx[(f >> 11) & 3]
    e &= 0x1fff
    f &= 0x7ff
    eps &= 0xfff

    if tag == 'X':
        assert 0 <= d < 0x800
        sign = f >> 12
        i =  m24.vect_to_cocode(1 << j)
        w = d ^ (f & 0x7ff)
        # sign ^= d >> 12
        # d &= 0x7ff
        sign += parity_eps * m24.scalar_prod(d, i)
        sign += m24.scalar_prod(e, i)
        sign += parity_eps * m24.gcode_weight(d)
        sign += m24.scalar_prod(e ^ f, m24.ploop_theta(d))
        sign += m24.scalar_prod(d, eps ^ m24.ploop_theta(e))
        sign += m24.gcode_weight(f) 
        return (-1)**sign * value, 'X', w, j
    elif tag in 'YZ':
        assert 0 <= d < 0x800
        tau = tag == 'Y'
        sigma = (tau + parity_eps) & 1
        w_tag = "ZY"[sigma]
        sign = (e >> 12) + (f >> 12) * (sigma + 1)
        sign += ((e >> 11) & sigma)
        # sign += (d >> 12) + ((d >> 11) & tau)
        # d &= 0x7ff
        i =  m24.vect_to_cocode(1 << j)
        w = (d ^ e ^ (f & ~-sigma)) & 0x7ff
        sign += m24.scalar_prod(f, i)
        sign += m24.scalar_prod(f, m24.ploop_theta(d))
        sign += m24.scalar_prod(d, eps ^ m24.ploop_theta(e ^ f)
              ^ (sigma ^ 1) * m24.ploop_theta(f))
        sign += m24.scalar_prod(f, m24.ploop_theta(e))
        sign += sigma * m24.scalar_prod(e, m24.ploop_theta(f))
        sign += (sigma ^ 1) * m24.gcode_weight(f)
        return (-1)**sign * value, w_tag, w, j
    elif tag == 'T':
        o = d
        assert 0 <= o < 759
        d = m24.octad_to_gcode(o)
        w_j = j ^ as_suboctad(f, d)
        sign = m24.suboctad_weight(j) * parity_eps
        sign += m24.suboctad_scalar_prod(as_suboctad(e ^ f, d), j)
        sign +=  m24.scalar_prod(e, m24.ploop_theta(d))
        sign +=  m24.scalar_prod(d, eps ^ m24.ploop_theta(e))
        return (-1)**sign * value, 'T', o, w_j
    elif tag in 'BC':
        m = tag == 'C'
        i = d
        c = m24.vect_to_cocode((1 << i) ^ (1 << j))
        n = m ^ m24.scalar_prod(f, c)
        w_tag = "BC"[n]
        i, j = max(i, j), min(i, j)
        sign = m * parity_eps + m24.scalar_prod(e ^ f, c)
        return (-1)**sign * value, w_tag, i, j
    elif tag == 'A':
        i = d
        i, j = max(i, j), min(i, j)
        c = m24.vect_to_cocode((1 << i) ^ (1 << j))
        sign = m24.scalar_prod(f, c)
        return (-1)**sign * value, 'A', i, j
    else:
        raise ValueError("Bad tag " + tag)




def op_xy_prep(v, eps, e, f):
    """Multiply unit vector v with group element

    Same functionality as function ``op_xy`` implemented above, but
    we use the C function ``mmv_fast_op_xy_type mmv_fast_prep_xy``. 
    """
    assert len(v) == 4
    value, tag, d, j = v
    op = np.zeros(3 + 4*24 + 2048 + 759*2 , dtype = np.uint32)
    mmv_fast_test_prep_xy(f, e, eps, op)
    op_lin_d, op = op[:3], op[3:]
    op_lin_i, op = op[:96].reshape((4,24)), op[96:]
    op_signs, op_T = op[:2048], op[2048:].reshape((759,2))
    parity_eps = (eps >> 11) & 1  # parity of eps

    ker_table_yx = [0, 0x1000, 0x1800, 0x800];
    e ^= ker_table_yx[(f >> 11) & 3]
    e &= 0x1fff
    f &= 0x7ff
    eps &= 0xfff
    
    if tag == 'X':
        assert 0 <= d < 0x800
        assert op_lin_i[0][j] in [0, 0xff]
        sign = (op_signs[d] >> 0)  + op_lin_i[0][j]
        sign_ref = m24.gcode_weight(f) & 1
        sign_ref ^= m24.scalar_prod(d, eps ^ m24.ploop_theta(e))  
        sign_ref ^= m24.gcode_weight(d) & parity_eps & 1
        sign_ref ^= m24.scalar_prod(e ^ f, m24.ploop_theta(d))
        assert op_signs[d] & 1 == sign_ref, (op_signs[d], sign_ref)
        sign_ref ^= m24.scalar_prod(e, m24.vect_to_cocode(1 << j))
        assert sign & 1 == sign_ref, (sign & 1, sign_ref)
        sign ^= parity_eps * m24.scalar_prod(d, m24.vect_to_cocode(1 << j))
        w = d ^ op_lin_d[0];
        return (-1)**int(sign) * value, 'X', w, j
    if tag == 'Z':
        tag1 = 'ZY'[parity_eps]
        sign = (op_signs[d] >> 1)  + op_lin_i[1][j]
        w = d ^ op_lin_d[1];
        return (-1)**int(sign) * value, tag1, w, j
    if tag == 'Y':
        tag1 = 'YZ'[parity_eps]
        sign = (op_signs[d] >> 2)  + op_lin_i[2][j]
        w = d ^ op_lin_d[2];
        return (-1)**int(sign) * value, tag1, w, j
    if tag == 'T':
        assert 0 <= d < 759
        wj = j ^ op_T[d, 0]
        """
        sign = op_T[d, 1] >> 6
        sign ^= m24.suboctad_scalar_prod(j, op_T[d, 1] & 63)
        sign ^= m24.suboctad_weight(j) & (op_T[d, 1] >> 7)
        """
        sign = mmv_fast_read_table_perm64_xy(op_T[d, 1], j) & 1
        return (-1)**sign * value, 'T', d, wj
    raise ValueError("Unknown tag")



ignored_tags = []

def one_test_op_xy(v, eps, e, f, verbose):
    eps_atom = grp('d', eps)
    x_atom = grp('x', e)**(-1)
    y_atom = grp('y', f)**(-1)
    global ignored_tags
    w = v * eps_atom * x_atom  * y_atom 
    if verbose:
        print("%s * %s * %s * %s = %s" % (v, eps_atom, x_atom, y_atom, w))
    v_tuple = vector_data(v)   
    w_tuple = vector_data(w)
    w1_tuple = op_xy(v_tuple, eps, e, f)  
    w1 = v.space(v.p, [w1_tuple])
    if w != w1:
        print("%s * %s * %s * %s = %s" % (v, eps_atom, x_atom, y_atom, w))
        print("with xy formula:", w1)
        raise ValueError("Calculation with xy formula failed")
    if  not v_tuple[1] in 'XZYT':
        if not v_tuple[1] in ignored_tags:
            print("tag %s ignored" % v_tuple[1])
            ignored_tags.append(v_tuple[1])
        return
    w2_tuple = op_xy_prep(v_tuple, eps, e, f)
    w2 = v.space(v.p, [w2_tuple])
    if w != w2:
        print("%s * %s * %s * %s = %s" % (v, eps_atom, x_atom, y_atom, w))
        print("with function mmv_fast_test_prep_xy():", w2)
        raise ValueError("Calculation with xy formula in C failed")
        #print("WTF! Calculation with xy formula in C failed\n")






def rand_v(tag):
    return rep_mm(tag, "r")
   

def op_xy_testdata():
    data = [
       [ ("Y", 3, 6),  0, 0, 0x1171 ],
       [ ("Y", 3, 6),  12, 0, 0 ],
       [ ("Y", 3, 6),  12, 1111, 0 ],
       [ ("Y", 3, 6),  12, 0, 1111],
       [ ("Z", 3, 6),  0, 0, 0x1171 ],
       [ ("Z", 3, 6),  12, 0, 0 ],
       [ ("Z", 3, 6),  12, 1111, 0 ],
       [ ("Z", 3, 6),  12, 0, 1111],
       [ ("X", 3, 6),  0, 0, 0x1171 ],
       [ ("X", 3, 6),  12, 0, 0 ],
       [ ("X", 3, 6),  12, 1111, 0 ],
       [ ("X", 3, 6),  12, 0, 1111],
    ]
    for v, f, e, eps in data:
        if isinstance(v, str): 
            v1 = rand_v(v)
        else :
            v1=  rep_mm(*v)
        yield v1,  f, e, eps
    for v in "ZYX":
        for i in range(400):
            v1 =  rand_v(v)
            eps = randint(0, 0xfff) 
            e = randint(0, 0x1fff)
            yield v1,  eps, e, 0 
    v_tags = "ABCTYZX"
    for v in v_tags:
        for i in range(100):
            v1 =  rand_v(v)
            eps = randint(0, 0xfff)
            e = randint(0, 0x1fff)
            f = randint(0, 0x1fff)
            yield v1,  eps, e, f
      
       
@pytest.mark.mm_op
def test_op_xy(verbose = 0):
    print("Testing group operations x, y ...")
    for i, (v, eps, e, f) in enumerate(op_xy_testdata()):
        if verbose:
            print("Test", i)
        one_test_op_xy(v, eps, e, f, verbose = verbose)
    print("passed")






