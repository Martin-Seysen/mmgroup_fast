"""Try to beautify an axis orthogonal to the axis AXIS_Y.

Here AXIS_Y is the axis corrsponding to Y = MM('y', o), with
o = PLoop(range(8))), i.e. o correponds to the standard octad.
An axis to be beautified must be an axis orthogonal to axis AXIS_Y.
The main program in this module indicates that there are 38 orbits
of axes orthogonal to axis AXIS_Y under G_x0.

The main function beautify_axis_octad() modifies such an axis so
that the parts axis['A',:8,:8], axis['B',:8,:8], and axis['C',:8,:8]
are equal for each axis in the same G_x0 orbit.

Function test_beautify_axis_octad() tests the function
beautify_axis_octad().

Yet to be updated!!!!!
"""

import sys
import time
import os
from math import floor
from random import randint, shuffle, sample, Random
from collections import defaultdict
from multiprocessing import Pool
from optparse import OptionParser

import numpy as np


from mmgroup.bitfunctions import lin_table, bit_mat_inverse, bit_mat_mul
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_reduce_type4
from mmgroup.generators import gen_ufind_init, gen_ufind_union
from mmgroup.generators import gen_ufind_find_all_min, gen_ufind_make_map 
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_prep_eqn
from mmgroup.clifford12 import leech2matrix_solve_eqn
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup import PLoop, GCode, Cocode, XLeech2, GcVector, Octad
from mmgroup import MM, AutPL, Xsp2_Co1, MMV
from mmgroup.axes import Axis, BabyAxis
from mmgroup.general import Orbit_Lin2, Orbit_Elem2, Random_Subgroup

from axis_class_2B_axes import Y, Y_Gx0, AXIS_Y
from axis_class_2B_axes import CONJ_Y_INV, NEG_AXIS
from axis_class_2B_axes import rand_y
from axis_class_2B_axes import short_E8_vectors, inverse_E8, type2_4_i
from axis_class_2B_axes import map_omega
from axis_class_2B_axes import map_leech2_e8, map_e8_leech2, map_mm_e8

from axis_class_2B_sub import data_type4_large, get_axis_type
from axis_class_2B_sub import data_type4
from axis_class_2B_sub import partition_suboctad
from axis_class_2B_sub import co_affine


from axis_class_2B_beautify import relabel, get_abs_abc
from axis_class_2B_beautify import final_correction_y_octad
from axis_class_2B_beautify import beautify_axis_octad, G_STD_OCTAD
from axis_class_2B_beautify import case_from_c8



################################################################
# Try to disambguate axis orbits
###############################################################



def display_T(axis):
    hx = "0123456789ABCDEF"
    print("T[o] =", "".join([hx[x] for x in axis['T', G_STD_OCTAD]]))



def ABCT_lo_is0(axis):
    is0 = (axis['A',:8,:8] +  axis['A',:8,:8] + axis['C',:8,:8]) == 0
    is0 = is0.all() and (axis['T',G_STD_OCTAD] == 0).all()
    return is0



det_rng = Random(45)
RAND_N64 = np.array([det_rng.randint(1, (1 << 50) - 1) for i in range(64)],
               dtype = np.int64)



def diag_high(axis):
    a = get_abs_abc(axis, high = True, sign = False)
    a = np.array(a, dtype = np.uint32)
    a2 = np.array([axis['A',i,i] for i in range(8,24)], dtype = np.int64)
    for i in range(16):
        for j in range(16):
            a2[i] += RAND_N64[a[0, i, j] + 8]
            a2[i] += RAND_N64[a[1, i, j] + 16]
            a2[i] += RAND_N64[a[2, i, j] + 24]
    relabel(a2)
    #print("dddd", a2)
    return a2



def diag_high_display(axis):
    a2 = diag_high(axis)
    d = defaultdict(list)
    for i, x in enumerate(a2):
        d[x].append(i)
    d = dict(d)
    return d

def map_lo_zero(axis):
    return sorted((int(axis['A',i,i]) == 0, i-8) for i in range(8, 24))


class AffineHiSpace:
    def __init__(self):
        self.a = []
    def add(self, x):
        if x not in self.a:
            if len(self.a) == 0:
                self.a.append(x)
            else:
                d = x ^ self.a[0]
                self.a += [d ^ y for y in self.a]
    def map_hi(self):
        data = [x + 8 for x in self.a]
        pi = AutPL(0, zip(data, list(range(8, 24))), 0)
        return Xsp2_Co1(pi)

def str_coaffine(v):
    s = '*' if len(co_affine(v)) == 0 else ''
    return str(len(v)) + s   



def map_diag_affine(axis):
    a = diag_high(axis)
    i_min, a_min = None, [1000] * 16
    mu = min(a)
    #print(a)
    for i, x in enumerate(a):
        if x == mu:    
            b = [a[i ^ j] for j in range(16)]
            if b < a_min:
                 a_min, i_min = b, i
    assert i_min != None
    pi = AutPL(0, zip([i+8 for i in range(16)], 
        [(i ^ i_min) + 8 for i in range(16)]))   
    return MM(pi)

#################################################################################


def iter_samples(lst, n):
    if n >= len(lst):
        if n == len(lst):
            yield lst[:]
    elif n == 1:
        for x in lst:
            yield [x]
    else:
        for i, x in enumerate(lst):
            for l1 in iter_samples(lst[:i] + lst[i+1:], n-1):
                yield [x] + l1
       
def pi6_cases():
    A = [0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0, 1,0,1,0, 0,1,1,0]
    s = sum(x << i for i, x in enumerate(A[8:]))
    D = {s : AutPL()}
    for a, b, c, d in iter_samples([0,1,4,5,6,7], 4):
        pi =  AutPL(0, zip([2,3,a,b,c,d,8],[2,3,b,a,d,c,8]))
        pi1 = pi.perm
        y = [A[pi1[i]] for i in range(24)]
        s = sum(x << i for i, x in enumerate(y[8:]))
        if not s in D:
            #print(hex(s), pi1)
            D[s] = pi
            if len(D) == 6:
                #print(D)
                return D

_PI6_DICT = None

def pi6_dict():
    global _PI6_DICT
    if _PI6_DICT is None:
        _PI6_DICT = pi6_cases()
    return _PI6_DICT




#######################################################################

CASES_14_15 = {}

MAXLEN_CASES_14_15 = {14:32, 15:32}

HI_OCTADS = [0, GCode(list(range(8, 24))).ord]
for i in range(759):
    o = Octad(i)
    if min(GcVector(o).bit_list) >= 8:
        HI_OCTADS.append(o.ord & 0xfff)
assert len(HI_OCTADS) == 32, len(HI_OCTADS)



def rand_pi_14_15():
    P, Q, R = sample([0,2,4,6], 3)
    p, q, r = [randint(0, 1) for i in range(3)]
    t = randint(8, 15)
    swap = [P^p, P^p^1, Q^q, Q^q^1, R^r, R^r^1, t]
    #print(swap)
    pi = AutPL(0, zip(swap, [0,1,2,3,4,5,8])) 
    x = sample(HI_OCTADS, 1)[0]
    return Xsp2_Co1([('p', pi), ('x', x)])


def hash_case_14_15(axis):
    a = get_abs_abc(axis, sign = True, high = True)[2, 0:8:2, 8:16:2]
    a = a.ravel()
    h = sum(1 << i for i, x in enumerate(a) if x > 7)
    return h 


def make_case_14_15(case, axis):
    global CASES_14_15
    CASES_14_15[case] = d = {}
    for i in range(2000):
        g = rand_pi_14_15()
        h = hash_case_14_15(axis * g)
        if h not in d:
           d[h] = g**-1
           #print(len(d), hex(h))
           if len(d) == MAXLEN_CASES_14_15[case]:
               break
    print("Table for case %d has length %d" % (case, len(d)))
        
def reduce_case_14_15(case, axis):
    if case not in CASES_14_15:
        make_case_14_15(case, axis)    
    axis *= CASES_14_15[case][hash_case_14_15(axis)]



#######################################################################

CASES_3 = {}


def compute_cases_3():
    global CASES_3
    for i in range(1000):
        pi = AutPL(0, 'r ot')
        p = pi.perm
        if p[8] & 0xf8 != 8:
            continue
        t = tuple(p[14:16])
        if t not in CASES_3:
            CASES_3[(t[1], t[0])] = CASES_3[t] = pi**-1 
            if len(CASES_3) >= 56:
                break
         

compute_cases_3()


#################################################################################

def postprocess(axis):
    """Yet to be documented
    """
    cases = case_from_c8(axis)
    #print("c", cases)
    if 28 in cases:
        a = [axis['A', i, i] for i in range(8, 24)]
        s = sum(1 << i for i, x in enumerate(a) if x == 1)
        axis *= MM(pi6_dict()[s])
        return axis
    if 25 in cases and axis['A', 8, 8] == 3:  
        for y in [16, 20]:
            if axis['A', y, y] == 4:
                z1 = [8,9,10,11,0,4,y ]
                z2 = [8,9,10,11,0,4,12 ]
                pi = AutPL(0, zip(z1, z2))
                axis *= MM(pi)
                return axis
    if 22 in cases and axis['A', 8, 8] == 6:  
        for y in [10, 11]:
            if axis['A', y, y] == 3:
                z = 10 + 11 - y
                z1 = [8,9,y,z,0,4,12 ]
                z2 = [8,y,9,z,0,4,12 ]
                pi = AutPL(0, zip(z1, z2))
                axis *= MM(pi)
                return axis
    if 18 in cases: 
        #print("case 18") 
        for y in [10, 11]:
            if axis['A', y, y] == 10:
                z = 10 + 11 - y
                z1 = [8,y,z,9,0,1,4 ]
                z2 = [8,9,y,z,0,1,4 ]
                #print("ccccccccc", 18, y, z, z1, z2)
                pi = AutPL(0, zip(z1, z2))
                axis *= MM(pi)
                return axis
    if 14 in cases or 15 in cases:
        axis *= get_x_equation(axis)
        axis *= get_x_sign_equation(axis)
        a = get_abs_abc(axis, sign = True, high = True)[2, 8:10, :2]
        #print("AAA", get_abs_abc(axis, sign = True, high = True)[2])
        case = 14 if (a == a[0,0]).all() else 15
        reduce_case_14_15(case, axis)
        return axis
    if 3 in cases:
        axis *= get_x_equation(axis)
        a = get_abs_abc(axis, sign = True, high = True)[2, 0, :8]
        al = [i for i, x in enumerate(a) if x != a[0]]
        while len(al) >= 4:
            al1 = al[1:4] + [al[1] ^ al[2] ^ al[3]] 
            #print("rrrrrrr", al, al1)
            al1 += [x+8 for x in al1]
            g = Xsp2_Co1('x', PLoop([x+8 for x in al1]))
            axis *= g
            a = get_abs_abc(axis, sign = True, high = True)[2, 0, :8]
            al = [i for i, x in enumerate(a) if x != a[0]]        
        if len(al):
            pi = CASES_3[(al[0]+8, al[1]+8)]
            axis *= Xsp2_Co1(pi)
            axis *= get_x_equation(axis)
        return axis
    return axis

#################################################################################


_HEX="0123456789abcdef"

def get_abs_abc_mid(axis, sign = True,  display = False):
    """Return array of certain entries of the axis

    The function returns an array A of shape (3, 8, 16) with entries
    of the axis. A[e,:,:] contains entries of part 'A' of the
    axis * T**e, where T is the triality element. Here the entries
    0:8, 8:24 are returned. If sign == False the absolute values
    of the entries  A[e,:,:] are returned instead.

    If display == True then the returned entries are diplayed.
    """
    T = [MM(), MM('t', 1), MM('t', 2)]
    a = np.array([(axis * g)['A', :8, 8:] for g in T], dtype = np.uint8)
    if not sign:
       a = np.where(a > 7, 15 - a, a)
    if display:
        def f(s, a0):
            return (s,) + tuple(_HEX[x] for x in a0)  
        fmt = ["%3s " + "%2s" * 16] * 3
        fmt = "  ".join(fmt)
        s0, s1, s2 = "A0:", "A1:", "A2:"
        for i in range(8):
           data = f(s0, a[0, i]) + f(s1, a[1, i])+ f(s2, a[2, i])
           print(fmt % data)
           s0 = s1 = s2 = ""
        print("")
    return a


#######################################################################

def get_x_equation(axis):
    """Find x_d such that transforming with x_d adjusts the signs

    x_d is retured as an instance of class Xsp2_Co1.
    """
    a = get_abs_abc(axis, sign = True, high = True)[2]
    coeff = 0
    nrows = 0
    m = np.zeros(24, dtype = np.uint64)
    for i in range(1,7):
        row = Cocode([0,i]).ord
        r = leech2matrix_add_eqn(m, nrows, 24, row)
        assert r == 1 
        nrows += 1 
    for i in range(16):
        for j in range(0, i):
            if a[i,j]:
                row = Cocode([i+8,j+8]).ord
                bit = a[i,j] > 7
                r = leech2matrix_add_eqn(m, nrows, 24, row)
                assert r in [0, 1]
                if r == 1:
                    coeff += bit << nrows
                    nrows += 1
    b = np.zeros(24, dtype = np.uint32)
    assert leech2matrix_prep_eqn(m, nrows, 24, b) >= 0
    v = leech2matrix_solve_eqn(b, nrows, coeff)
    assert v >= 0
    result = Xsp2_Co1('x', v & 0xfff)
    #if Y_Gx0 ** result != Y_Gx0:
    #    result *= Xsp2_Co1('d', Cocode([7,8]))
    assert Y_Gx0 ** result == Y_Gx0 
    return result 

AUTPL_STD_OCTAD = PLoop([0,1,2,3,4,5,6,7])

def get_x_sign_equation(axis):
    a = get_abs_abc_mid(axis, sign = True)[2]
    sign = 0
    for i in range(8):
        for j in range(16):
            if a[i,j] != 0:
                sign = a[i,j] > 7
                break
    return Xsp2_Co1('x', AUTPL_STD_OCTAD if sign else 0)
    


#######################################################################


def get_y_equation(axis):
    """Find y_d such that transforming with x_d adjusts the signs

    y_d is retured as an instance of class Xsp2_Co1.
    """
    a = axis['A']
    coeff = 0
    nrows = 0
    m = np.zeros(24, dtype = np.uint64)
    for i in range(24):
        for j in range(0, i):
            if a[i,j]:
                row = Cocode([i,j]).ord
                bit = a[i,j] > 7
                r = leech2matrix_add_eqn(m, nrows, 24, row)
                assert r in [0, 1]
                if r == 1:
                    coeff += bit << nrows
                    nrows += 1
    b = np.zeros(24, dtype = np.uint32)
    assert leech2matrix_prep_eqn(m, nrows, 24, b) >= 0
    v = leech2matrix_solve_eqn(b, nrows, coeff)
    assert v >= 0
    result = Xsp2_Co1('y', v & 0xfff)
    if Y_Gx0 ** result != Y_Gx0:
        result *= Xsp2_Co1('d', Cocode([7,8]))
    assert Y_Gx0 ** result == Y_Gx0 
    return result 


#######################################################################


ORBIT_ABC_DICT = {}
det_rng_abc = Random(45)
RAND_ABC = np.array(
    [det_rng_abc.randint(1, (1 << 40) - 1) for i in range(24**2)],
     dtype = np.int64)


def orbit_from_reduced_ABC(axis, case = None):
    global ORBIT_ABC_DICT
    h = int(sum(axis['C'].ravel() * RAND_ABC))
    if case is not None:
        if h in ORBIT_ABC_DICT:
            assert ORBIT_ABC_DICT[h] == case, (ORBIT_ABC_DICT[h], case)
        else:
            ORBIT_ABC_DICT[h] = case
    return ORBIT_ABC_DICT[h] 



#######################################################################


REF_AXES = {None:None}



def beautify_axis_abc(axis, case = None, check = 1, verbose = 0):
    r"""Beautify an axis orthogonal to axis AXIS_Y

    Let H be the centralizer of axis AXIS_Y in G_x0. The function
    transforms the axis ``axis`` with an element of H so that
    
    for each axis orthogonal to axis AXIS_Y in the same G_x0 orbit.

    The general transformation strategy is:

    - Use final_correction_y_octad() and beautify_axis_octad()
      to get the following parts of the axis right:
      axis['A',:8,:8], axis['B',:8,:8], axis['C',:8,:8], axis['T', o]
      where o is the standard octad

    - The final strategy is yet under development!!!!!!!
    """
    global DISAMBIGUATE, REF_AXES
    axis = beautify_axis_octad(axis, case = case)
    axis = final_correction_y_octad(axis)
    if not ABCT_lo_is0(axis):
        d = diag_high_display(axis)
        ll = ", ".join([str_coaffine(v) for k, v in sorted(d.items())])
        #print("ll", ll)
        axis *= map_diag_affine(axis)
    else:
        #print("case =", case, map_lo_zero(axis))
        lst = map_lo_zero(axis)
        aff = AffineHiSpace()
        for _, i in lst[:9]:
            aff.add(i)
        axis *= aff.map_hi()
    postprocess(axis)
    axis *= get_x_equation(axis)
    axis *= get_x_sign_equation(axis)
    axis *= get_y_equation(axis)
    case = orbit_from_reduced_ABC(axis, case)
    axis *= get_y_equation(axis)

    #assert AXIS_Y * axis.g1 == AXIS_Y
    if verbose:
        print("case =", case)
        get_abs_abc(axis, display = True)
        display_T(axis)
        get_abs_abc_mid(axis, sign = 1, display = True)
        get_abs_abc(axis, high = True, sign = 1,  display = True)

    if check and case is not None:
        assert AXIS_Y * axis.g1 == AXIS_Y
        if case not in REF_AXES:
            REF_AXES[case] = axis
        ref = REF_AXES[case]
        if ref is not None:
            ref_abc = np.array([ref[t] for t in "ABC"], dtype=np.uint8)
            abc = np.array([axis[t] for t in "ABC"], dtype=np.uint8)
            assert (ref_abc == abc).all()
            assert (ref['T', G_STD_OCTAD] == axis['T', G_STD_OCTAD]).all()
            
            """ 
            ref_abc = get_abs_abc(ref, sign = True)
            abc = get_abs_abc(axis, sign = True)
            assert (ref_abc == abc).all()
            #ref_abc_hi = get_abs_abc(ref, sign = False, high = True)
            #abc_hi = get_abs_abc(axis, sign = False, high = True)
            #assert (ref_abc_hi == abc_hi).all()
            assert (ref['T', G_STD_OCTAD] == axis['T', G_STD_OCTAD]).all()
            ref_mid = get_abs_abc_mid(ref, sign = 1)
            mid = get_abs_abc_mid(axis, sign = 1)
            assert (mid[:2] == 0).all()
            assert (ref_mid == mid).all()
            ref_abc_hi_s = get_abs_abc(ref, sign = 1, high = True)
            abc_hi_s = get_abs_abc(axis, sign = 1, high = True)
            assert  (ref_abc_hi_s == abc_hi_s).all()
            """

    return axis, case



     

    