"""Try to beautify an axis ortogonal to the axis AXIS_Y.

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

from axis_class_2B_axes import Y , Y_Gx0, AXIS_Y
from axis_class_2B_axes import CONJ_Y_INV, NEG_AXIS
from axis_class_2B_axes import rand_y
from axis_class_2B_axes import short_E8_vectors, inverse_E8, type2_4_i
from axis_class_2B_axes import map_omega
from axis_class_2B_axes import map_leech2_e8, map_e8_leech2, map_mm_e8

from axis_class_2B_sub import data_type4_large, get_axis_type
from axis_class_2B_sub import data_type4
from axis_class_2B_sub import partition_suboctad


G_STD_OCTAD = GCode(list(range(8))).octad


#######################################################################
# Auxiliary functions for function beautify_axis_octad()
#######################################################################

def find_good_type4(axis):
    """Find good type 4 vector for an axis orthogonal to AXIS_Y

    Here AXIS_Y is the axis corrsponding to Y = MM('y', o), with
    o = PLoop(range(8))), i.e. o correponds to the standard octad.
    Axis ``axis`` must be an axis orthogonal to axis AXIS_Y.

    There are 135 type-4 vectors in the subspace of the Leech
    lattice mod 2 related to Y. These are the vectors in the space
    spanned by Omega, x_o, and the 64 short even vectors in the
    standard suboctad. Here the standard suboctad is the space of
    all even Golay cocode vectors that can be represented as subsets
    of octad o.

    Let C be the centralizer of axis ``axis``.
    The function selects one vector v of these 135 vectors
    according to the following criteria.

    - The orbit of C on the set of these vectors should be small.

    - When mapping v to x_{-1} and then applying a suitable power
      of the triality element, the G_x0 orbit of the resulting axis
      should be 'nice'.

    Supersets of C-orbits are computed with function 
    data_type4_large(); and evidence shows that in makes sense
    to look for vector v in one of the smallest if these supersets.
    The 'nicest' image under the triality element is computed with
    functton get_axis_type().

    The function returns a triple (l, t, v).
    Here v is the selected type-4 vector. l is the length of the
    superset of the C-orbit of v as mentioned above. t is the
    axis type of the 'nicest' image under the triaity element.
    """
    _, TYPE4I = type2_4_i()
    vt = data_type4_large(axis)
    part4 = defaultdict(list)
    for i, x in enumerate(TYPE4I):
        if x:
            ax_t = get_axis_type(x, axis)
            part4[(ax_t, vt[i])].append(x)
    d = {}
    for (ax_t, v1), data in part4.items():
        d[(len(data), int(ax_t[:-1]), ax_t, v1)] = data
    mu = min(d.keys())
    return mu[0], mu[2], d[mu][0]

def find_good_tetrad(axis):
    """Find a good tetrad in the standard suboctad for an axis

    Parameter ``axis`` and terminology is as in function
    find_good_type4(). The function computes a tetrad in the
    standard suboctad that is good for reducing axis ``axis``. The
    tetrad is determined up to complementation in octad o only.

    The function returns triple (pi, l0, l1). Here pi is
    a permutation of the entries of the standard octad that maps
    the selected tetrad to [0,1,2,3] or to its complement [4,5,6,7]
    in o. pi is returned as an inctance of class AutPL.

    We use function std_partition_suboctad() for finding a
    tetrad in a small orbit of the cetralizer of ``axis``.
    l0 is (an upper bound for) the size the orbit of the tetrad
    found. l1 is the number of sets of the same size as the set
    containing v. 
    """
    a = partition_suboctad(axis)
    s = set((4 - y[1], y[2], y[3]) for y in a)
    mu = min(s)
    l_set = len([x for x in s if x[0] == 0 and x[1] == mu[1]])
    #c = "!" if l_set > 1 else ""
    mu_l = [4 - mu[0],  mu[1], mu[2]]
    a4 = [y[0] for y in a if list(y[1:]) == mu_l]
    v_syn = Cocode(a4[0]).syndromes_llist()[:2]
    pi = AutPL(0, zip(v_syn[0] + v_syn[1][:2], range(6)), 0)
    return pi, len(a4), l_set




def relabel(a):
    """Relable a list or a numpy array

    Let a be a list or a numpy array of integers, and assume that
    the entries of a are a labels. We map the set of these labels
    to a set of contiguous small integers, starting with 1.

    We replace the entries of a by their images under that mapping.
    The images of the labels are sorted by their freqency as entries
    of array a as primary key, and then by the values of the labels.   
    """
    d = defaultdict(int)
    a1 = a.ravel() if isinstance(a, np.ndarray) else a
    for x in a1:
        d[x] += 1
    lst = sorted((n, x) for x, n in d.items())
    d = {}
    for i, (n, x) in enumerate(lst):
        d[x] = i + 1    
    for i in range(len(a1)):
        a1[i] = d[int(a1[i])]



# global variables used un function duad_table()
SH = 35
DUAD_TABLE = None
det_rng = Random(42)
RAND_N64 = np.array([det_rng.randint(1, (1 << SH) - 1) for i in range(64+8)],
               dtype = np.int64)
rand_n64_vectorize = np.vectorize(lambda x: RAND_N64[x])

def duad_table():
    """Return auxiliary table for function make_duad

    The function returns an array ``a`` of dimension 64 times 4. The
    index of the array corresponds to the entries of standard suboctad
    as defined in function find_good_type4() and used in function
    std_partition_suboctad().

    For any of these suboctad array ``a`` contains a quadruple
    (f, d0, d1, s) to be used with the result ``r``  of function
    partition_suboctad() applied to an axis. Here we compute a
    watermarking of the entries of the standard suboctad for a
    given axis using function partition_suboctad(axis).

    if d0 and d1 are nonzero for an entry of the standard suboctad 
    then the corrsponding entry of ``r`` contributes to the result
    result duads[d0, d1] with weight f. In case d0 > 0, d1 = 0
    or d0 = 0, d1 > 0 it contributes to all entries
    duads[d0, :]  or  duads[:, d1], repectively, instead. 

    For computing the bias returned by function make_duad() we
    sum up all entry of ``r`` with weight s and we return -1, 0,
    or 1, depending on the sign of that sum.
    """
    global DUAD_TABLE
    if DUAD_TABLE  is not None:
        return DUAD_TABLE
    SET0, SET1 = set([0,1,2,3]), set([4,5,6,7])
    D0 , D1 = {}, {}
    DUAD_TABLE = np.zeros((64,4), dtype = np.int64)
    for i in range(4):
        for j in range(4):
            D0[tuple((i,j))] = i ^ j
            D1[tuple((i ^ 4,j ^ 4))] = i ^ j
    F = [(RAND_N64[i] & 0xffff) for i in range(64, 72)]
    fsum = 0
    for i in range(64):
        vsyn = Cocode(map_e8_leech2(i)).syndromes_llist()[0]
        if len(vsyn) == 2:
            t = tuple(vsyn)
            if t in D0:
                DUAD_TABLE[i] = [F[0], D0[t], 0, F[2]]
                fsum += F[2]
            if t in D1:
                DUAD_TABLE[i] = [F[0], 0, D1[t], -F[2]]
                fsum -= F[2]
        if len(vsyn) == 4 and  vsyn[1] < 4 and vsyn[2] >= 4:
            t0, t1 = tuple(vsyn[:2]),  tuple(vsyn[2:])
            DUAD_TABLE[i] = [F[1], D0[t0], D1[t1], 0]
    assert fsum == 0, fsum
    return DUAD_TABLE




MONAD_TABLE = None

def monad_table():
    """Return auxiliary table for function make_duad

    The function returns an array ``a`` of dimension 64 times 3. The
    index of the array corresponds to the entries of standard suboctad
    as in function duad_table().

    For any of these suboctad array ``a`` contains a quadruple
    (d0, d1, f) to be used with the result ``r``  of function
    partition_suboctad() applied to an axis. Here we compute a
    watermarking of the entries of the standard suboctad for a
    given axis using function partition_suboctad(axis).

    Here the corrsponding entry of ``r`` contributes to the result
    result monads[d0, d1] of function make_duads() with weight f.
    """
    global MONAD_TABLE
    if MONAD_TABLE  is not None:
         return MONAD_TABLE
    omega = Cocode([0,1,2,3]).ord
    MONAD_TABLE = np.zeros((64,3), dtype = np.int8)
    for i in range(64):
        v = map_e8_leech2(i)
        if gen_leech2_type(v) == 2:
            cl = Cocode(v).syndrome_list()
            if cl[0] < 4 and cl[1] >= 4:
                  MONAD_TABLE[i] = [cl[0], cl[1] - 4, 1]   
        if gen_leech2_type(v ^ omega) == 2:
            cl = Cocode(v ^ omega).syndrome_list()
            if cl[0] < 4 and cl[1] >= 4:
                  MONAD_TABLE[i] = [cl[0], cl[1] - 4, 32]   
    return MONAD_TABLE



def make_duad(axis):
    """Compute information for sorting entries of the standard octad

    Given an axis, the funtion returns a triple (bias, d, m, diag)
    that can be used for rearranging the entries of the standard
    octad in the Mathieu group M_24. Here we assume that functions
    find_good_type4() and find_good_tetrad() have already been called
    so that it suffices to swap the tetrads (0,1,2,3) and (4,5,6,7)
    and to rearrange the entries inside each of these terads.

    We use function partition_suboctad() to obtain data for
    rearranging the octad for the axis ``axis``. 
    The returned value 'bias' is 0 if we cannot obtain any order
    of the tetrads form these data. Otherwise we will have bias = 1
    or -1, indicating the two possible orders of the tetrad. Here
    the sign does not indicacte a preference; it just disambiguates
    the two possible orders of the tetrad.

    The 4 x 4 matrix 'duads' labels the three possible pairs of
    duads in each tetrad. Entry duads[0][0] is irrelevant. Entry
    duads[i][0] labels the pair of duads in the first tetrad
    containing the duad (0, i), Entry duads[0]][j] labels the pair
    of duads in the second tetrad containing the duad (4, 4 ^ j).
    Entry (i, j), i, j > 0 labels an edge in the bipartite graph
    with the six pairs of duads mentioned above as vertices.

    The 4 x 4 matrix 'monads' labels pairs of entries of the standard
    octad 0, with one entry in the first and one entry in the second
    tetrad. More specifcally, monads[i,j] refers to the pair
    (i, 4+j). 

    'diag' contains the first 8 elements of the diagonal of part 'A'
    of the axis.
    """
    a = rand_n64_vectorize(partition_suboctad(axis)[:,3])
    d = np.zeros((4, 4), dtype = np.int64)
    bias = 0
    for h, (f, d0, d1, s) in zip(a, duad_table()):
        if d0:
            if d1:
                d[d0, d1] += int(h) * f
            else:
                d[d0,:] += int(h) * f
        elif d1:
            d[:,d1] += int(h) * f 
        bias += int(h) * s
    relabel(d)
    bias = max(-1, min(1, bias))

    m = np.zeros((4, 4), dtype = np.int64)
    for h, (d0, d1, f) in zip(a, monad_table()):
       m[d0, d1] += int(h) * int(f)
    relabel(m)
    diag = [int(axis['A',i,i]) for i in range(8)]
    return  bias, d, m, diag



def xch_points(i, j, duads, monads, diag, perm, condition = True):
    """Exchange entry i with entry j in the standard octad o

    Here an octad is given by a triple (duads, monads, diag, perm),
    with 'duads', 'monads', and 'diag' as returned by function
    make_duad() and 'perm' describing the permutation. These three
    entries are updated when entries i and j are exchanged. Then
    perm[i] is exchanged with perm[j]. 

    The permutation is done only if 'condition' is True (default).

    The function returns the parity of the performed permutation.
    """
    if i == j or not condition:
        return 0
    assert 0 <= i ^ j < 4
    diag[i], diag[j] = diag[j], diag[i]
    perm[i], perm[j] = perm[j], perm[i]
    col = i >= 4
    if col:
        i, j = i - 4, j - 4
        duads, monads =  duads.T, monads.T
    exchange = [x for x in range(1,4) if x != i ^ j] 
    x0, x1 = min(exchange), max(exchange)
    #print("XCH", i + 4 * col, j + 4 * col, x0 + 4 * col, x1 + 4 * col)
    duads[[x0, x1]] = duads[[x1, x0]]             
    monads[[i,j]] = monads[[j,i]]
    if col:
        duads, monads =  duads.T, monads.T                  
    return 1






            
def xch_tetrads(bias, duads, monads, diag):
    """Exchange upper and lower tetrad of octad o if appropriate

    Here the triple (bias, duads, monads, diag) must be as returned by
    function make_duad(). The function possibly exchanges the upper
    and the lower tetrad of octad o depending on the information
    contained in that triple. As a general rule, the tetrad
    containing smaller orbits is mapped to the lower tetrad.   

    The function updates 'duads' and 'monads' when the tetrads are
    exchanged and returns the triple (perm, duads, monads). Here
    'perm' is a list of length 8 representing the identity or a
    permutation exchanging the two tetrads.
    """
    l0, l4 = len(set(duads[:,0])), len(set(duads[0,:]))
    xch = False
    if l0 < l4:
        xch = True
    elif l0 == l4:
        m0 = sorted(len(set(monads[:,i])) for i in range(4))
        m4 = sorted(len(set(monads[i,:])) for i in range(4))
        if m0 != m4:
            xch = m0 < m4
        else:
            xch = bias < 0  
    #print("lllllll", l0, l4, bias)
    #xch = l0 < l4 or (l0 == l4 and bias < 0)
    #print(l0, l4, bias, xch)
    if xch:
        diag = diag[4:] + diag[:4]
        return [4,5,6,7,0,1,2,3], duads.T, monads.T, diag
    else:     
        return [0,1,2,3,4,5,6,7], duads, monads, diag





def duad_pair_bias(monads, diag, start, length):
    """Strategy for flipping elements of a duad or a pair of duads

    Let o be the standard octad given as the set [0,...,7].
    The function decides whether to exchange whether to exchange
    the subset o[start : start + length] with the subset
    [start + length : start + 2*length] or not. It uses the pair
    'monads', 'diag' returned by function make_duad() for making that
    decision. Here length must be 1 or 2, and start must be a
    multiple of 2*length.

    The function returns 1 if exchange is recommended, -1 if not, and
    0 if no such recommendation ven be derived from input 'monads'.
    """
    st1 = start
    if start >= 4:
        start -= 4
        monads = (monads.copy()).T
        diag = diag[4:]
    data = [sorted(list(map(int, monads[i]))) for i in range(4)]
    if length == 2:
        d0, d1 = sorted(data[:2]), sorted(data[2:])
        if d0 == d1:
            d0, d1 = sorted(diag[:2]), sorted(diag[2:])
    elif length == 1:
        d0, d1 = data[start], data[start + 1]
    result = 0 if d0 == d1 else (-1) ** (d0 < d1)
    #print("cxxx!", st1, length,  result, d0, d1, diag)
    return result



def xch_duads(duads, monads, perm, diag):
    """Exchange entries inside the two standard tetrads

    Exchange entries of inside the lower tetrad (0,1,2,3) and the 
    upper tetrad (4,5,6,7) of the standard octad o. Here we assume
    that function xch_tetrads() has been called and that parameters
    perm, duads, monads  are as returned by that function.

    Our strategy is as follows:

    First use parameter 'duads' to select the best pairs of duads
    in the two tetrads, and permute the entries of o so that all
    duads in these pairs are dajacent.

    Then use function duad_pair_bias() with parameter 'monads' and
    'diag' first for pairs of adjacent duads and then for the two
    entries of each duad, and check if any of these objects is to
    be exchanged. We exchange these objects if approprate.

    As a general strategy, 'best' objects are those with few orbits 
    as indicated by the input parameters. Such 'best' objects are
    selected and swapped into lowes possible positions. 
    """
    def augment_count(lst):
        d = defaultdict(int)
        for x in lst:
            d[x] += 1
        return sorted([(d[x], x, i) for i, x in enumerate(lst)])      
    par = 0
    l1 = augment_count(duads[:,0]) 
    if l1[1][0] == 1:
        d = l1[1][2]
        #print("dddd", d, l1)
        if d != 1:   
            par ^= xch_points(1, d, duads, monads, diag, perm)
    l1 = augment_count(duads[0,:]) 
    if l1[1][0] == 1:
        d = l1[1][2]
        #print("dddd", d+4, l1)
        if d != 1:   
            par ^= xch_points(5, 4+d, duads, monads, diag, perm)
    #display_duads(8, duads, monads)
    cond = duad_pair_bias(monads, diag, 0, 2) > 0
    #print("cc0", cond, duad_pair_bias(monads, diag, 0, 4))
    xch_points(0, 2, duads, monads, diag, perm, cond)
    xch_points(1, 3, duads, monads, diag, perm, cond)
    cond = duad_pair_bias(monads, diag, 4, 2) > 0
    #print("cc4", cond, duad_pair_bias(monads, diag, 4, 4))
    xch_points(4, 6, duads, monads, diag, perm, cond)
    xch_points(5, 7, duads, monads, diag, perm, cond)
    for j in range(0, 8, 2):
        cond = duad_pair_bias(monads, diag, j, 1) > 0
        par ^= xch_points(j, j+1, duads, monads, diag, perm, cond)
    #display_duads(9, duads, monads)
    if par == 1:
        par ^= xch_points(6, 7, duads, monads, diag, perm)  
    return par 



def display_duads(bias, duads, monads):
    print("  bias: %2d, duads: %3d%3d%3d%3d, monads: %3d%3d%3d%3d" %
           ((bias,) + tuple(duads[0]) + tuple(monads[0])) )
    fmt = "                   %3d%3d%3d%3d          %3d%3d%3d%3d"
    for i in range(1,4):
        print(fmt %  (tuple(duads[i]) + tuple(monads[i])))







 




def get_y_octad_equation(axis):
    """Find y_d such that transforming with y_d adjusts the signs

    This function assumes that function xch_duads() has been called
    to rearrange the the entries of the standard octad o. Then this
    function tries to find Golay code element d such that
    transforming the axis ``axis`` with y_d adjusts the signs of 
    axis['A',:8,:8], axis['B',:8,:8], and axis['C',:8,:8].

    y_d is retured as an instance of class Xsp2_Co1.
    """
    a = axis['A',:8,:8]
    coeff = 0
    nrows = 0
    m = np.zeros(24, dtype = np.uint64)
    for i in range(8):
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
    
       
# Data for function do_bad_case()
_BAD_LIST1 = [
  ([1, 14, 8, 7, 0, 0, 0, 0], [0,1,3,2,4,5]),
  ([2,  0, 8, 7, 7, 7, 7, 7], [0,1,3,2,4,5]),
]
_T2 = MM('t', 2)

def do_bad_case(axis):
    """Ad hoc auxiliary function for function beautify_axis_octad()

    The function deals with some cases for function
    beautify_axis_octad() that cannot be done by the previous
    functions in this section.

    This function will map axis['A',:8,:8], axis['B',:8,:8], and
    axis['C',:8,:8] to a standard form also in the diffcult cases.
    """
    a2 = list((axis * _T2)['A',0,:8])
    for symptom, repair in _BAD_LIST1:
        if a2 == symptom:
            pi = AutPL(0, zip(repair, range(6)), 0)
            axis *= MM(pi)
            break
    return axis
 
#######################################################################
# correct data related to octad
#######################################################################

_OCTAD_PREIMAGE = None

def octad_preimage():
    """Map suboctads to octad preimages.

    The function returns a dictionary d that maps the suboctads of
    the standard octad o (as enumerated by the indices of the vector
    axis['t', o] of an axis) to a preimage in the Golay code.
    For a suboctad  s  of size 4 we have o & d[s] == s.
    We do not compute preimages of suboctads of lengths not divisible
    by four.
    """
    global _OCTAD_PREIMAGE
    if _OCTAD_PREIMAGE is not None:
        return _OCTAD_PREIMAGE
    _OCTAD_PREIMAGE = {} # np.zeros((64,2), dtype = np.uint32)
    vect = MMV(255)()
    std_octad =  GCode(list(range(8)))
    std_octad_no = std_octad.octad
    for i in range(64):
        vect['T', std_octad_no, i] = i
    for octad in range(759):        
        o = Octad(octad)
        _OCTAD_PREIMAGE[0] = 0
        if int((o & std_octad)/2) == 0:
            v_img = vect * Xsp2_Co1('y', o)
            img = v_img['T', std_octad_no, 0]
            assert 0 <= img < 64
            if img not in _OCTAD_PREIMAGE:
               _OCTAD_PREIMAGE[img] = o.ord & 0xfff
    assert len(_OCTAD_PREIMAGE) == 36
    return _OCTAD_PREIMAGE



"""Dictionary for final reduction

We want to perform  a final reduction of an axis with function 
final_correction_y_octad(). After that reduction, entry axis['t',o]
(where o is the standard octad) should be the same for all axes in
the same orbit. Such a reduction is necessary for the orbits with
numbers 33, 35, and 37. Here the orbit is identified by the
concatentation of the tuples axis['B',0,:8]) and axis['C',0,:8].
Unfortuantely, we cannot disambiguate orbtis 33 and 35 by using
this or a similar tuple.

So the dictionary FINAL_CORRECTION_DATA maps the relevant tuples as
above to a list corresponding to the parts axis['t',o] of reduced
axes. So for the tuple (axis['B',0,:8], axis['C',0,:8]) corresponding
to both orbits, 33 and 35, the value in the dictionary contains two
entrires axis['t',o] corresponding to the two orbits.

Use function test_beautify_axis_octad() with parameter verbose = 1
axes axes in orbits 33, 35, and 37 for obtaining these data.
"""

FINAL_CORRECTION_DATA = {
(0,11,0,0,0,0,0,0, 0,4,0,0,0,0,0,0) :
[
 # for orbits 33, 35
 [int(x) for x in """
  0 11  0 11 11  0 11  0  0 11 11  0  0  4  4  0 11  0  0  4 11  0  0  4
 11  0  4  0  0  4  0 11  0 11 11  0  0  4  4  0 11  0  4  0  0  4  0 11
  0  4  0  4  4  0  4  0  4  0  0 11  4  0  0 11""".split()],
 [int(x) for x in """
   0 11  0 11 11  0 11  0 11  0  0  4 11  0  0  4  0 11 11  0  0  4  4  0
 11  0  4  0  0  4  0 11  0 11 11  0  0  4  4  0  0  4  0  4  4  0  4  0
 11  0  4  0  0  4  0 11  4  0  0 11  4  0  0 11""".split()],
],

(0,2,13,13,13,13,13,13, 0,13,2,2,2,2,2,2) :
[
 # for orbit 37
 [int(x) for x in """
 4 11  0  0  0  0 11  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 11  4  0  0  0  0  4 11  0  0  0  0  0  0  0  0  0  0  4  4  4  4  0  0
  0  0  4  4  4  4  0  0  0  0  0  0  0  0  0  0""".split()],
],
}



FINAL_CORRECTION_DICT = None

def final_correction_dict():
    """Precompute a table used by function final_correction_y_octad()

    Function final_correction_y_octad() uses an element ('y', y) of
    G_x0 to transform an axis to its reduced form. Here y depends
    on a key 'k' in the dictionary FINAL_CORRECTION_DATA obtained from
    the axis and also on the value of axis['T', o], which is a vector
    of length 64. In total there are 36 possible values of y to be
    tested. For reducing the number of tests we compute the support
    of axis['T', o], i.e. a bit vector with the positions of the
    nonzero entries of axis['T', o], as a 64-bit integer. Then

    FINAL_CORRECTION_DICT[k][support(axis['T', o])]

    is the list of entries y that must be tested.
    """
    global FINAL_CORRECTION_DICT
    if FINAL_CORRECTION_DICT is not None:
        return FINAL_CORRECTION_DICT
    FINAL_CORRECTION_DICT = {}
    for k, lst in FINAL_CORRECTION_DATA.items():
         d = defaultdict(set)
         for data in lst:
             assert  len(data) == 64
             occ_list = [int(x != 0) for x in data]
             for i, y in octad_preimage().items():
                 t_occ_list = [occ_list[j ^ i] for j in range(64)]
                 t_occ = sum(x << j for j, x in enumerate(t_occ_list))
                 d[t_occ].add(y)
         FINAL_CORRECTION_DICT[k] = d
    return FINAL_CORRECTION_DICT



def need_final_correction(axis):
    """Check if application of final_correction_y_octad() is required

    After calling function beautify_axis_octad(axis) we also want to
    map axis['T', o] to a fixed vector depending on the G_x0 orbit of
    the axis only. Here o is the standard octad.

    If such a (non-trivial) mapping is required we return the vector
    (axis['B',0,:8], axis['C',0,:8]) as a tuple. Otherwise we return
    None. If a tuple is returned, this is useful for finding a
    suitable mapping.
    """
    k = tuple(map(int, axis['B',0,:8])) + tuple(map(int, axis['C',0,:8]))
    data = axis['T', G_STD_OCTAD]
    if k in FINAL_CORRECTION_DATA and max(data) > 0:
        return k
    else:
        return None

def final_correction_y_octad(axis, case = None, verbose = 0):
    """Final reduction after calling function beautify_axis_octad()

    After calling function beautify_axis_octad(axis) we also want to
    map axis['T', o] to a fixed vector depending on the G_x0 orbit of
    the axis only. Here o is the standard octad.

    This function performs such a final reduction. It uses the
    following strategy.

    It first calls function need_final_correction(axis). A final 
    reduction is required if that function contains a tuple of 
    entries of the axis. Then that tuple is a key in dictionary
    FINAL_CORRECTION_DATA mapping such a tuple to a list of
    possible reduced vectors axis['T', 0]. In this case we compute
    a suitable element Xsp2_Co1('y', y) of G_x0 and transform
    the axis swith that element as required.

    Dictionary FINAL_CORRECTION_DATA can be obtained manually by
    calling function test_beautify_axis_octad() with parameter
    verbose = 1.
    """
    k = need_final_correction(axis)
    if k:
        d = final_correction_dict()[k]
        refdata_list = FINAL_CORRECTION_DATA[k]
        data = [int(x) for x in axis['T',G_STD_OCTAD]]
        occ = sum(int(x != 0) << i for i, x in enumerate(data))
        for y in d[occ]:
            g_y = Xsp2_Co1('y', y)
            ax1 = axis * g_y
            newdata = [int(x) for x in ax1['T', G_STD_OCTAD]]
            new_occ = sum((x != 0) << i for i, x in enumerate(newdata))
            for refdata in refdata_list:
                if newdata  == refdata:
                    #print("ok",  case)
                    axis *= g_y
                    return axis         
        raise ValueError("WTF", case)  
    return axis         



#######################################################################
# Dictionary for obtaining orbit from part ['B',:8,:8] - ['C',:8,:8]
#######################################################################

DICT_C8_ORBIT = defaultdict(list)
DICT_ORBIT_C8 = {}
det_h_rng = Random(57)
RAND_C8_ORBIT = np.array([det_h_rng.randint(1, (1 << 50) - 1) 
                  for i in range(64)],  dtype = np.int64)


def hash_dict_c8(axis):
    """Return hash value computed from of axis

    This hash value is computed from  axis['B',:8,:8] - axis['C',:8,:8]
    """
    a = ((15 + axis['B',:8,:8] - axis['C',:8,:8]) % 15).ravel()
    h1 = sum(a * RAND_C8_ORBIT)
    d = [int(x) for x in np.diagonal(axis['A',8:,8:])] 
    h2 = sum(1 << (4 * x) for x in d)
    a0 = axis['A',:8,:8].ravel()
    a0 = np.where(a0 > 7, 15 - a0, a0)
    h3 = sum(a0 * RAND_C8_ORBIT)
    return (h1, h2, h3)
    

def store_dict_c8(axis, case):
    """Store hash value for axis with orbit 'case'"""
    global DICT_C8_ORBIT, DICT_ORBIT_C8
    if case is None:
        return
    h = hash_dict_c8(axis)
    if case in DICT_ORBIT_C8:
        assert DICT_ORBIT_C8[case] == h
    else:
        DICT_ORBIT_C8[case] = h
        DICT_C8_ORBIT[h].append(case) 

def case_from_c8(axis):
    """Return list of possible orbits of an axis"""
    return DICT_C8_ORBIT[hash_dict_c8(axis)]


def display_non_disambiguated_cases():
    """Display orbits that could not be disambiguated"""
    ll = [str(x) for x in DICT_C8_ORBIT.values() if len(x) > 1]
    S = "Orbits that could not be disambiguated:\n  %s"
    if len(ll):
       print(S % (", ".join(ll)))


#######################################################################
# The main function beautify_axis_octad()
#######################################################################




_HEX="0123456789abcdef"

def get_abs_abc(axis, high = False, sign = True,  display = False):
    """Return array of certain entries of the axis

    The function returns an array A of shape (3, n, n) with entries
    of the axis. Here n = 8 if high == False, and n = 16 otherwise.
    A[e,:,:] contains entries of part 'A' of the axis * T**e, where T
    is the triality element. In case high == False the entries
    0:8, 0:8 are returned; otherwise the entries 8:24, 8:24 are
    returned. If sign == False the absolute values of the entries
    A[e,:,:] are returned instead.

    If display == True then the returned entries are diplayed.
    """
    T1, T2 = MM('t', 1), MM('t', 2)
    lo, hi = (8, 24) if high else (0, 8)
    a = axis['A', lo:hi, lo:hi]
    a1 = (axis.copy() * T1)['A', lo:hi, lo:hi]
    a2 = (axis.copy() * T2)['A', lo:hi, lo:hi]
    if not sign:
       a = np.where(a > 7, 15 - a, a)
       a1 = np.where(a1 > 7, 15 - a1, a1)
       a2 = np.where(a2 > 7, 15 - a2, a2)
    a_all = np.array([a, a1, a2], dtype = np.uint8)
    if display:
        n = hi - lo
        def f(s, a):
            return (s,) + tuple(_HEX[x] for x in a)  
        fmt = ["%3s " + "%2s" * n] * 3
        fmt = "  ".join(fmt)
        s0, s1, s2 = "A0:", "A1:", "A2:"
        for i in range(n):
           data = f(s0, a_all[0, i])  
           data += f(s1, a_all[1, i])  
           data += f(s2, a_all[2, i])
           print(fmt % data)
           s0 = s1 = s2 = ""
        print("")
    return a_all
    





def beautify_axis_octad(axis, check = 1, case = None, verbose = 0):
    r"""Beautify an axis orthogonal to axis AXIS_Y

    Let H be the centralizer of axis AXIS_Y in G_x0. The function
    transforms the axis ``axis`` with an element of H so that the parts
    axis['A',:8,:8], axis['B',:8,:8], and axis['C',:8,:8] are equal
    for each axis orthogonal to axis AXIS_Y in the same G_x0 orbit.

    The general transformation strategy is:

    - Use find_good_type4() to find an element of Q_x0 that is
      transformed to Omega

    - Use find_good_tetrad() to find a sextet that is transformed to
      the standard sextet.

    - Use functions xch_tetrads() and xch_duads() to rearrange the
      tetrads in a sextet and the entries inside a tetrad.

    - Use function get_y_octad_equation() to find a Golay code element
      d such that transforming with y_d adjusts the signs of 
      axis['A',:8,:8], axis['B',:8,:8], axis['C',:8,:8].

    - Finally use the ad hoc function do_bad_case() to correct some
      remaining mismatches.

    After calling this function, the function
    final_correction_y_octad() should be called.

    If 'case' is not None we check a hash value for the orbit given by
    'case'. When a 'case' occurs the first time, that hash value
    is computed and stored.
    """
    axis.rebase()
    length, t, v4 = find_good_type4(axis)
    axis = axis * map_omega(v4)
    pi, length4, muliple4 =  find_good_tetrad(axis)
    axis *= Xsp2_Co1('p', pi)
    c = "!" if muliple4 > 1 else ""
    if verbose:
        print("  beautifying candidate: ", length, t, ", l4 =", length4, c)
    bias, duads, monads, diag = make_duad(axis)
    perm, duads, monads, diag = xch_tetrads(bias, duads, monads, diag)
    xch_duads(duads, monads, perm, diag)
    #print("perm", perm)
    pi = AutPL(0, zip(perm[:6], range(6)), 0)
    axis *= MM(pi)
    axis *= get_y_octad_equation(axis)
    if check:
        bias_new, duads_new, monads_new, _ = make_duad(axis)
        assert (duads_new == duads).all()
        assert (monads_new == monads).all()
    do_bad_case(axis)
    if verbose:
        bias_new, duads_new, monads_new, _ = make_duad(axis)
        display_duads(bias_new, duads_new, monads_new)
        get_abs_abc(axis, high = False, sign = True,  display = True)
    assert AXIS_Y * axis.g1 == AXIS_Y, (AXIS_Y * axis.g1).g_axis * Y
    store_dict_c8(axis, case)
    return axis



#######################################################################
# Testing function beautify_axis_octad()
#######################################################################



def test_beautify_axis_octad(axis, ntests = 80, case = None, verbose = 0):
    """Test function beautify_axis_octad()

    That function is tested with the given 'axis'. 'ntests' tests are
    performed. An optional 'case' information can be given for
    displaying.

    If 'verbose' is True then the test is performed in such a way 
    that the data required for function final_correction_y_octad()
    are displayed. Otherwise the combination of functions
    beautify_axis_octad() and final_correction_y_octad() is tested.
    """
    oct = G_STD_OCTAD
    if isinstance(axis, str):
        axis = Axis(MM(axis))
    for n in range(ntests):
        ax1 = axis * rand_y()
        ax1 = beautify_axis_octad(ax1, case = case, verbose = 0)
        assert AXIS_Y * ax1.g1 == AXIS_Y
        abc = get_abs_abc(ax1, sign = True)
        data = ax1['T', oct]
        if n == 0:
            ref_data = data
        if verbose:
            # expand and display operation of final_correction_y_octad() 
            if need_final_correction(ax1):
                ok = False
                for i, y in octad_preimage().items():
                    ddd = np.array([data[j ^ i] for j in range(64)], 
                          dtype = np.uint8)
                    if ((ddd != 0) == (ref_data != 0)).all():
                        ax1 *= Xsp2_Co1('y', y)
                        newdata = ax1['T', oct]
                        ok =  (ref_data == newdata).all()
                        print("TTTT", case, ok,
                             ",".join(map(str,ax1['B',0,:8])) + ", " +
                             ",".join(map(str,ax1['C',0,:8])),
                             "\n", newdata, "\n")
                        break
                assert ok, "case %d" % case
        else:
            axis = final_correction_y_octad(ax1, case)
        assert AXIS_Y * ax1.g1 == AXIS_Y
        if n == 0:
            ref_data = ax1['T', oct]
            ref_abc =  get_abs_abc(ax1, sign = True)
        else:
            abc = get_abs_abc(ax1, sign = True)
            assert (abc == ref_abc).all(), (case, abc, ref_abc) 
            data = ax1['T', oct]
            assert (data == ref_data).all(), (case, data[-11:], ref_data[-11:])

    return "ok"



     

    