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
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup import MM, AutPL, PLoop, Cocode, XLeech2, Xsp2_Co1, MMV, GcVector
from mmgroup.axes import Axis, BabyAxis

from axis_class_2B_axes import Y , Y_Gx0, AXIS_Y
from axis_class_2B_axes import CONJ_Y_INV, NEG_AXIS
from axis_class_2B_axes import short_E8_vectors, inverse_E8, type2_4_i
from axis_class_2B_axes import map_omega
from axis_class_2B_axes import rand_y

##################### marking an axis ##############################################

def prep_analyse_axis():
    """Compute vectors in Leech lattice mod 2 for involution Y

    There are 256 elements v of the Leech lattice mod 2 related to
    involution Y in a standard way, with 135 elements of type 4.

    The function returns a dictionary mapping each of these 135
    vectors v of type 4 to a pair (a, e), such that for the
    positive preimage v_p of v in Q_x0 we have

        v_p * MM('a', a) * MM('t', e) = x_{-1} .

    Here MM('a', a) is in G_x0, and  x_{-1} is the central
    involution in G_x0. 
    """
    r = np.zeros(1000, dtype = np.uint32)
    mm_reduce_analyze_2A_axis(AXIS_Y.v15.data, r)
    assert r[0] == 0x22
    assert r[1] == 0x21
    assert r[3] == 0x100
    a = np.zeros((135,6), dtype = np.uint32) 
    t = np.zeros(135, dtype = np.uint32)
    i = 0
    d = {}
    for v in r[4:4+0x100]:
        if gen_leech2_type(v) == 4:
            ax = AXIS_Y.v15.copy()
            length = gen_leech2_reduce_type4(v, a[i])
            ax *= MM('a', a[i, :length])
            ok = False
            for e in (1,2):
                g = MM('t',e)
                #print(e, (ax * g).axis_type())
                if (ax * g).axis_type() == '2A':
                    t[i] = e
                    assert not ok
                    ok = True
                    break
            assert ok
            d[v] = a[i], t[i]                     
            i += 1
    assert i == 135
    return d # list(zip(a, t))

A_MODE = prep_analyse_axis()


def map_axis_type(ax_type):
    return str(ax_type >> 4) + "?ABCDEFGHIJKLMNO"[ax_type & 15]






def analyse_axis_mark(ax):
    r"""Watermark an axis with respect to G_x0 \cap Cent(Y)

    Count classes of involution(a) * t, where t runs over the
    135 type-4 vectors related to Y, as computed in function 
    prep_analyse_axis(). The function returns a sorted
    tuple of pairs (class_name, n), where class_name is the
    name of a class an n is the number of elements found in
    that class.
    """
    d = defaultdict(int)
    for a, t in A_MODE.values():
        ax_type = mm_reduce_op_2A_axis_type(ax.v15.data, a, 
            len(a), 0x10 + (1 << t))
        #print(hex(ax_type), t, a)
        ax_type = (ax_type >> (8 * t)) & 0xff
        #print(hex(ax_type))
        d[ax_type] += 1
        #assert map_axis_type(ax_type) == (ax * MM('a', a) * MM('t', t)).axis_type()
    d1 = []
    for key in sorted(d):
        d1.append((map_axis_type(key), d[key]))
    return tuple(d1)



##################### alternative marking of an axis ############################



def axis_mark_ab(axis):
    """Mark an axis with information modulo 3

    Let ``axis`` be an axis orthogonal to AXIS_Y. Let SHORT_E8 be
    the list of 120 short vectors related to AXIS_Y as returned by
    function ``short_E8_vectors`` For each v in SHORT_E0 let

        mark(axis, v)

    be the pair containing the value of the A part of axis applied to
    v  and the (sign-adjusted) part of the rep 98280_x corresponding
    to v. Both values are taken modulo 3. The function counts the 9
    possible results of mark(axis, v) for all v in SHORT_E0 and
    returns the list of 9 frequencies of these results. 
    """
    assert isinstance(axis, Axis)
    d = [0] * 9
    v15d = axis.v15.data
    for w, sign, _ in short_E8_vectors()[1]:
        a = mm_op_eval_A(15, v15d, w)
        b = mm_aux_get_mmv_leech2(15, v15d, w)
        assert a >= 0 and b >= 0
        d[3 * ((b * sign) % 3) + a % 3] += 1
    return tuple(d)



##################### reduction ############################




# Will be 256 times 256 hadamard matrix with coefficents +- 1
H256 = None

def hadamard():
    """Return 256 x 256 Hdamard matrix with entries +- 1"""
    global H256
    if H256 is not None:
        return H256
    from scipy.linalg import hadamard
    H256 = hadamard(256, dtype = np.int32) 
    return H256



# Coefficients for function data_type4
AA20 = [1968711, 1628775, 1917407, 1381083, 1423795,
        1069935, 2063131, 1780467, 1521439]
AA18 = [317171, 285987, 331571, 411867, 312995, 504387,
        448399, 408627, 477751]
AA_SHORT = AA18
AA = AA20


def data_type4(axis, aa, mask = -1):
    """Distinguish the 135 type-4 vectors related to axis AXIS_Y

    There are 120 type-2 vectors and 135 type-4 vectors in the Leech
    lattice mod 2 that are related to the axis AXIS_Y. These vectors
    are the nonzero vectors of an 8-dimensional space E8.
    For the reduction of an element of the Monster described by the
    axis ``axis`` and AXIS_Y is is important to find 'good' type-4
    vectors related to AXIS_Y.
   
    This function watermarks such type-4 vectors w by counting
    the occurences of the values mark(axis, v) for vectors v with
    (halved) scalar products <v, w> = 0 and <v, w> = 1 (mod 2)
    separately. Counting these values directly is way to much effort.
    So we compute hash values h(axis, w) for axis ``axis`` and
    type-4 vectors w as above  based on counting these occurences.

    There are 9 possible values mark(axis, v). For each of these 9
    values we select a large integer aa[i], where i runs over the
    possible values of mark(axis, v). We compute the hash values:

    h(axis, w) =   sum     aa[mark(axis, v)] * (-1) ** (v, w) .      

    for all w of type 4 in E8. Here the sum runs over all v in E8 of
    type 2. If ``mask`` is a power of two minus 1 then the hash
    value is reduced modulo ``mask``. The vector ``vt`` of all such
    hash values can be obtained by mulitplying a certain vector with
    a 256 times 256 Hadamard matrix, which is fast.

    The function returns a vector ``vt`` of length 256 such that
    vt[i] is the hash value for w = TYPE4I[i] in case TYPE4I[i] > 0.          
    """
    vt = np.zeros(256, dtype = np.int32)
    assert isinstance(axis, Axis)
    d = [0] * 9
    v15d = axis.v15.data
    for w, sign, n in short_E8_vectors()[1]:
        a = mm_op_eval_A(15, v15d, w)
        b = mm_aux_get_mmv_leech2(15, v15d, w)
        value = 3 * ((b * sign) % 3) + a % 3
        vt[n] = aa[value] 
    vt = (vt @ hadamard()) & mask
    return vt


def partition_type4(axis, aa = AA, mask = -1):
    """Distinguish the 135 type-4 vectors related to axis AXIS_Y

    Let E8, h(axis, w) be as in function data_type4(), where w is a
    type-4 vector in E8. The function partitions the set of these
    vectors w according to h(axis, w) and returns the sorted list
    of the sizes of the parts of the partition.
    """
    _, TYPE4I = type2_4_i()
    vt = data_type4(axis, aa, mask)
    d = defaultdict(int)
    for i, x in enumerate(TYPE4I):
        if x:
            d[vt[i]] += 1
    return sorted(d.values())

def partition_type4_nonzero(axis, aa = AA, mask = -1):
    """Distinguish the 135 type-4 vectors related to axis AXIS_Y

    Same as function partition_type4(), but ignoring a partition
    corresponding to  h(axis, w) = 0. Considering this kind of
    partitions only will probably speed up a live version of the
    reduction algorithm im mmgroup.
    """
    _, TYPE4I = type2_4_i()
    vt = data_type4(axis, aa, mask)
    d = defaultdict(int)
    for i, x in enumerate(TYPE4I):
        if x and vt[i]:
            d[vt[i]] += 1
    return sorted(d.values())
    
        
def get_axis_type(v, axis):
    """Get type of axis with respect to type-4 vector in Leech lattice mod 2"""
    a, t = A_MODE[v]
    ax_type = mm_reduce_op_2A_axis_type(axis.v15.data, a, 
            len(a), 0x10 + (1 << t))
    ax_type = (ax_type >> (8 * t)) & 0xff
    ax_type = map_axis_type(ax_type)
    assert ax_type == (axis * MM('a', a) * MM('t', t)).axis_type()
    return ax_type

    
def small_type4(axis, aa = AA, mask = -1, select_first = False):
    """Small version of function partition_type4()

    The function returns all partitions of minimum size only. For
    any such partition it returns a dictionary counting the values
    of function get_axis_type(w, axis) for all w in that partition.
    So the function returns a list of dictionaries mapping the
    axis types found by function get_axis_type() to the number of
    occurrences in that partition.
    
    If ``select_first`` is True then only the first of the
    partition of minimum size if considered.
    """
    _, TYPE4I = type2_4_i()
    vt = data_type4(axis, aa, mask)
    d = defaultdict(int)
    for i, x in enumerate(TYPE4I):
        if x:
            d[vt[i]] += 1
    mu = min(d.values())
    mu_list = sorted([x for x in d if d[x] == mu])
    #print("  mmm", sorted(d.values()), mu_list)
    types_list = []
    for k in mu_list:
        v_list = [TYPE4I[i] for i, k1 in enumerate(vt) 
                   if TYPE4I[i] != 0 and k == k1]
        types_dict = defaultdict(int)
        for v in v_list:
            types_dict[get_axis_type(v, axis)] += 1
        types_list.append(dict(types_dict))
        if select_first:
            break
    return types_list


# Coefficients for function data_type4_large, similar to array AA
det_rng = Random(42)
AA15 = np.reshape(np.array([det_rng.randint(1, 0xffffffffffff) | 1 
       for i in range(15*15)], dtype = np.int64), (15, 15))


def data_type4_large(axis):
    """Large version of function data_type4()

    This ia a large version of function data_type4(). It performs
    the same action as function data_type4(); but uses information
    modulo 15 instead of information modulo 3.

    This can be used to confirm that information modulo 3 is 
    sufficient for reduction in the mmgroup algorithm.
    """
    vt = np.zeros(256, dtype = np.int64)
    assert isinstance(axis, Axis)
    d = [0] * 9
    v15d = axis.v15.data
    for w, sign, n in short_E8_vectors()[1]:
        a = mm_op_eval_A(15, v15d, w)
        b = mm_aux_get_mmv_leech2(15, v15d, w)
        vt[n] = AA15[(b * sign) % 15,  a % 15]
    vt = (vt @ hadamard())
    return vt


def partition_type4_large(axis):
    """Large version of function partition_type4()

    This ia a large version of function partition_type4(). It performs
    an action similar to function partition_type4(). It uses uses
    information modulo 15 instead of information modulo 3.
    It distinguishes the 135 type-4 vectors related to axis AXIS_Y
    also by the result of get_axis_type() applied to these vectors.

    It returns a pair of sorted lists of the sizes of the parts of
    two partitions. 

    The first partition is a partition of the type-2 vectors; and the
    second one is a partition of the type-4 vectors related to Y_AXIS.
    """
    TYPE2I, TYPE4I = type2_4_i()
    vt = data_type4_large(axis)
    d2 = defaultdict(int)
    d4 = defaultdict(int)
    for i, x in enumerate(TYPE2I):
        if x:
            d2[vt[i]] += 1
    for i, x in enumerate(TYPE4I):
        if x:
            ax_t = get_axis_type(x, axis)
            d4[(vt[i], ax_t)] += 1
            #d4[(vt[i],)] += 1
    return sorted(d2.values()), sorted(d4.values())




######## adjust partition for the case that Omega is fixed ####################


_PART_OMEGA = None


def std_partition_suboctad():
    """Return data for partitioning the standard suboctad

    The function returns an array ``a`` of dimension 64 times 4.
    ``a[i]`` is a quadruple ``(v, type, w1, w2)``. Here ``i`` is the
    i-th suboctad.  ``v`` is Golay cocode vector corresponding to that
    suboctad in *Leech lattice encoding*.  ``type`` is the type  of
    ``v`` in the Leech lattice. Let ``vt`` the vector returned by
    function ``data_type4(axis,...)`` that computes data for an axis
    ``axis`` to be used for watermarking the vectors in the Leech
    lattice mod 2 associated with the 2A involution Y . Then ``w1``
    and ``w2`` are the watermarks for the vectors ``v`` and
    ``v + Omega``.
    """
    global _PART_OMEGA
    if _PART_OMEGA is not None: 
        return _PART_OMEGA
    a = np.zeros((64, 4), dtype = np.uint16)
    a[:,0] = lin_table(short_E8_vectors()[0][:6])
    a[:,1] = [gen_leech2_type(x) for x in a[:,0]]
    T4, T2 = type2_4_i()
    d = defaultdict(list)
    for i, v in enumerate(T4 + T2):
        if v and v & 0x7ff800 == 0: 
            d[v & 0x7ff].append(i & 0xff)
    for i, v in enumerate(a[:,0]):
        if v:
            assert len(d[v]) == 2, (i, hex(v), d[v])
            a[i, 2:] = d[v]
    a[0] = 0
    assert set(a[1:,2]) == set(range(1,64))
    assert set(a[1:,3]) == set(range(129,192))
    _PART_OMEGA = a
    #for i, b in enumerate(a): print(i, b)
    return _PART_OMEGA       
           


def partition_suboctad(axis):
    """Mark entries of an axis corrsponding to the standard suboctad

    Given a axis ``axis``, the function returns an array ``a`` of
    dimension 64 times 4. ``a[i]`` is a quadruple ``(v, type, f, k)``.
    Here ``i`` is the i-th suboctad, and ``v``, ``type`` are as in 
    function  ``std_partition_omega``. ``k`` is a number identifying
    the orbit of the suboctad under the axis, as far as it can be
    distinguished form the entries of the axis. ``f`` is the number
    of suboctads contained in that orbit.
    """
    data = data_type4_large(axis)
    d = defaultdict(int)
    lst = []
    for v, t, w1, w2 in std_partition_suboctad():
        q1, q2 =  data[w1], data[w2]
        entry = (t, min(q1,q2), max(q1,q2))
        lst.append(entry)
        d[entry] += 1
    d1 = {}
    for k, (entry, f) in enumerate(sorted(d.items())):
        d1[entry] = [f, k]
    b = np.copy(std_partition_suboctad())
    for i, entry in enumerate(lst):
        b[i, 2:] = d1[entry] 
    return b

    
def partition_suboctad_freq(axis):
    b = partition_suboctad(axis)[:, 1:]
    s = sorted(set(map(tuple, b)))
    d = defaultdict(list)
    for t, f, _ in s:
        d[t].append(int(f))         
    return d[2], d[4]

#partition_suboctad_freq(AXIS_Y)
#1/0

######## partition of A part of matrix ####################################



def find_mapper(axis):
    _, TYPE4I = type2_4_i()
    vt = data_type4(axis, aa=AA)
    d = defaultdict(int)
    for i, x in enumerate(TYPE4I):
        if x:
            d[vt[i]] += 1
    mu = min(d.values())
    h_l = [h for h in d if d[h] == mu]
    v_l = [TYPE4I[i] for i, h in enumerate(vt) if h in h_l and TYPE4I[i]]
    for v in v_l:
        assert gen_leech2_type(v) == 4
    ax_l = [(v, get_axis_type(v, axis)) for v in v_l]
    d_ax = defaultdict(int)
    for v, ax in ax_l:
         d_ax[ax] += 1
    mu = min(d_ax.values())
    mu_l = [x for x in d_ax if d_ax[x] == mu]
    o_l = [(int(x[:-1]), x[-1:], v) for v, x in ax_l if x in mu_l]
    o_l.sort()
    v = o_l[0][2]
    #print("xx",  o_l, hex(v))
    return map_omega(v)
    



def display_ABC8(axis):
    fmt = ((" %2d" * 8 + "  ") * 3)[:-2]
    abc = [(axis * MM('t', e)).v15['A', :8, :8] for e in range(3)]
    for i in range(8):
        d = list(abc[0][i]) + list(abc[1][i]) + list(abc[2][i])
        print(fmt % tuple(d))



_HASH8 = np.array([det_rng.randint(0, 1 << 57) for i in range(0x1010)],
                            dtype = np.int64)
_HASH8_VECT = None


def hash_coeffients():
    return _HASH8

def hash_vectorize():
    global _HASH8_VECT
    if _HASH8_VECT is None:
        _HASH8_VECT = np.vectorize(lambda x: _HASH8[x]) 
    return _HASH8_VECT


def relabel_a(a):
    la = len(a)
    diag = [a[i,i] for i in range(la)]
    for i in range(la):
        a[i,i] = 0
    s = sorted(set(a.ravel()))
    d = dict(zip(s, range(1, len(s) + 1)))
    h0 = len(d) + 1 
    for i in range(la):
        for j in range(la):
            a[i,j] =  d[a[i,j]]
    s = sorted(set(diag))
    d = dict(zip(s, range(h0, h0 + len(s))))
    for i in range(la):
        a[i,i] =  d[diag[i]]
    return a    



def hash_a(axis):
    a0, a1, a2 = np.array(
       [(axis * MM('t', e)).v15['A'] for e in range(3)], dtype = np.int32)
    s0 = np.where(a0 > 7, 15 - a0, a0)
    s1 = np.where(a1 > 7, 15 - a1, a1)
    s2 = np.where(a2 > 7, 15 - a2, a2)
    sgn = ((a0 * a1 * a2) % 15) & 1
    s2[:8, :8] = a2[:8, :8]
    a = sgn * 512 + s0 * 256 + s1 * 16 + s2
    for i in range(24):
         a[i,i] = 4096 + a0[i,i]
    #relabel_a(a)
    return a
    

def hash_diag(a, indices = None):
    if indices is None:
        indices = range(len(a))
    for i in indices:
        a[i,i] += sum(a[i,j] for j in indices if j != i)
    return relabel_a(a) 


def co_affine(lin_list_16):
    lst = lin_list_16[:]
    while len(lst) > 2:
        x = lst[0] ^ lst[1] ^ lst[2]
        lst = lst[3:]
        try:
            lst.remove(x)
        except ValueError:
            lst.append(x)
    if len(lst) == 2:
       lst[0], lst[1] = 0, lst[0] ^ lst[1]
    return lst



def hash_diag_lin(a):
    la = len(a)
    done = False
    cnt = 0
    while not done:
        done = True
        hash_diag(a)
        d = defaultdict(list)
        for i in range(la):
            d[a[i,i]].append(i)
        #print(list(d.values()))
        diff = la
        singletons = [l1[0] for l1 in d.values() if len(l1) == 1]
        for lst in d.values(): 
            if len(lst) & 1 and len(lst) > 1:
                rem = co_affine(lst)
                if len(rem) == 1 and not rem[0] in singletons:
                   done = False
                   cnt += 1
                   assert cnt < 10 
                   a[rem[0],rem[0]] += diff
                   diff += la
    return a


def count_hash_partition(hlist):
    d = defaultdict(int)
    for i in hlist:
        d[i] += 1
    return sorted(list(d.values()))



def subpartition(a, indices):
    """Partition a graph represented by a matrix

    Let ``a`` be a symmetric square matrix encoding an edge-labelled
    graph. Labelling of vertices is igorned. We also assume the
    multiset of colors of edges adjacent to a vertex is the same 
    for each vertex. We consider only vertices of the graph occuring
    as indices in the set  ``indices``.

    We try to find a partition of these vertices into parts of
    equal sizes that can be distinguished by analyzing the graph.
    We return a triple  ``(ind, k, e)`` if a partition into ``e``
    parts of equal size ``k`` has been found. Thus ``e == 1`` means
    no parition found. ``ind`` is a  reordered list of  the set
    ``indices``, containing the  concatenation of the ``e``
    (ordered) parts found. We always have ``len(ind) == k * e``.
    """
    indices = sorted(list(set(indices)))
    li = len(indices)
    if li < 3 or (li < 9 and li & 1):
        return indices, li, 1
    b = np.zeros((li, li), dtype = a.dtype)
    for i, ai in enumerate(indices):
        for j, aj in enumerate(indices):
            b[i, j] = a[ai, aj]
    d = defaultdict(int)
    for i in range(li - 1):
        d[b[i,li - 1]] += 1
    mu = min(d.values())
    if mu + mu >= li:
        return indices, li, 1
    b_mu = [x for x in d if d[x] == mu][0]
    for i, row in enumerate(b):
        s = sum([x == b_mu for x in row]) -  (row[i] == b_mu)
        if s != mu:
            return indices, li, 1
    setlist = []
    table = np.zeros(li, dtype = np.uint32)
    assert gen_ufind_init(table, li) == 0
    for i in range(1, li):
        for j in range(i):
            if  b[i,j] == b_mu:
                gen_ufind_union(table, li, i, j)
    n_part = gen_ufind_find_all_min(table, li)
    assert n_part > 0
    if n_part == 1:
        return  indices, li, 1
    uf_map = np.zeros(li, dtype = np.uint32)
    assert gen_ufind_make_map(table, li, uf_map) == 0           
    partition = [(uf_map[i], i) for i in range(li)]
    partition.sort()
    h_part = count_hash_partition([uf_map[i] for i in range(li)])
    if len(set(h_part)) == 1:
         k = h_part[0]
         e = li // k
         if k > 1 and e > 1 and k * e == li:
             return [indices[i] for _, i in partition], k, e
    return indices, li, 1 



def partition_from_matrix(a):
    h = [a[i,i] for i in range(len(a))]
    d = defaultdict(int)
    for x in h:
        d[x] += 1
    d1 = defaultdict(list)
    for i, x in enumerate(h):
        d1[(d[x], x)].append(i)
    part_list = []
    partition = []
    for (n, x), d_list in sorted(d1.items()):
        indices, k, e = subpartition(a, d_list)
        part_list += indices
        partition.append((k, e))
    return part_list, partition    
    
def str_partition(partition):
    plist = [f"{k}^{e}" if e > 1 else str(k) for k, e in partition]
    return "[" + ", ".join(plist) + "]"



def is_affine_set(part, k, e):
    if k not in (2, 4, 8) or e not in (2, 4, 8):
        return False
    translated = set()
    for i in range(0, k * e, k):
        d = part[0] ^ part[i]
        translated.add(tuple(sorted(d ^ x for x in part[i:i+k])))
    if len(translated) > 1:
        return False
    if k == 2:
        return True
    return len(co_affine(part[:k])) == []

def str_partition_lin(part_list, partition):
    assert len(part_list) == 16
    lengths_list = [k * e for k, e in partition]
    plist = []
    s = 0
    last_entries = []
    for i, length in enumerate(lengths_list):
        k, e = partition[i]
        p_entries = part_list[s : s + length]
        if is_affine_set(p_entries, k, e):
            k = f"<{k}>"
        entry = f"{k}^{e}" if e > 1 else str(k)
        if len(co_affine(p_entries)) == 0:
            entry += '*'
        elif len(plist) and  entry == plist[-1]:
            if len(co_affine(p_entries + last_entries)) == 0:
                entry = f"({entry}, {entry})*" 
                plist.pop()
        plist.append(entry)
        s += length
        last_entries = p_entries
    return "[" + ", ".join(plist) + "]"



def test_find_mapper(axis, trials = 1000, verbose = 1):
    ref = None
    for i in range(trials):
        g = Xsp2_Co1('r')
        i8 = Y_Gx0 ** g
        c, h = i8.conjugate_involution_G_x0()
        assert c == "2A_o+"
        g1 = g*h
        ax = axis * g1
        mp = find_mapper(ax)
        ax *= mp
        g_prod = g1 * mp

        f_hash = hash_vectorize()
        a1 = f_hash(hash_a(ax))
        a8, a16 = a1[:8,:8], a1[8:,8:]
        hash_diag(a8)
        hash_diag_lin(a16)
        ind8, part8 = partition_from_matrix(a8)
        ind16, part16 = partition_from_matrix(a16)
        str_part8 = str_partition(part8)
        str_part16 = str_partition_lin(ind16, part16)
        pi = AutPL(0, zip(ind8[:6], range(6)), 0)
        g_pi = Xsp2_Co1('p', pi) 
        ax *= pi
        g_prod *= g_pi
        assert AXIS_Y * g_prod == AXIS_Y
        assert (ax.v15['A', :8, 8:] == 0).all()
        soct2, soct4 = partition_suboctad_freq(ax)
        
        entry = str_part8, str_part16, tuple(soct2), tuple(soct4)
        if ref is None:
            ref = entry
            if verbose:
                print("  partition: %-12s %s" % (str_part8, str_part16))
                if verbose > 1:
                    print("  suboctads type 2:", soct2, ", type 4:", soct4)
                    if verbose > 2:
                        display_ABC8(ax)
                        if verbose > 3:
                            print((ax.v15 * MM('t',2))['A', 8:, 8:])
        else:
            assert ref == entry, (ref,  entry)
    return 0    
  




         
        
