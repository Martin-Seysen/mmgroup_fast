import sys
from pathlib import Path
import time
import os
from math import floor
from random import randint, shuffle, sample, Random
from collections import defaultdict
import inspect
import pprint


import numpy as np


from mmgroup.bitfunctions import lin_table, bit_mat_inverse, bit_mat_mul
from mmgroup.bitfunctions import iter_bitweight
from mmgroup import mat24, octad_entries
from mmgroup import MM, AutPL, PLoop, Cocode, Xsp2_Co1, MMV, GcVector
from mmgroup import XLeech2, Octad, SubOctad, GCode
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_reduce_type4
from mmgroup.generators import gen_ufind_init, gen_ufind_union
from mmgroup.generators import gen_ufind_find_all_min, gen_ufind_make_map 
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.axes import Axis, BabyAxis

from axis_class_2B_axes import Y , Y_Gx0, AXIS_Y
from axis_class_2B_axes import CONJ_Y_INV, NEG_AXIS
from axis_class_2B_axes import short_E8_vectors, inverse_E8, type2_4_i
from axis_class_2B_axes import type_4_i_basis
from axis_class_2B_axes import map_omega
from axis_class_2B_axes import rand_y
from axis_class_2B_axes import map_e8_leech2, map_leech2_e8

from axis_class_2B_sub import data_type4
from axis_class_2B_sub import prep_analyse_axis




script_dir = Path(__file__).resolve().parent
GENERATED_FILE = "py_process_axis_2B.py"
GENERATED_PATH = os.path.join(script_dir, GENERATED_FILE)





##################### reduction ############################









###############################################################



HEADER = r'''# This file has been generate automatically; do not change!

"""Reduce a pair of orthognal 2A axes, on of them in G_x0 orbit '2B'

The basic word shortening algorithm for the Monster group in the
mmgroup package acts on pairs of orthogonal 2A axes, see [1].
In principle, we first transform one of these axes to the axis v^+.
Then we fix that axis and transform the other axis to the axis v^-.
Here v^+ and v^- are fixed axes defined as in [1], Both, v^+ and v^-,
are in the G_x0 orbit of axes labelled '2A'. This kind of a
transformation is called a reduction of a pair of axes.

In many cases, during the reduction of the first axis we arrive at an
axis in the G_x0 orbit '2B'. Then there ere are many ways to reduce the
first axis into an axis in the final orbit '2A'; and some of these ways
lead to a faster redcution of the second axis than others. In this
module we assume that the first axis has been reduced to the fixed
axis AXIS_Y defined below, which is in the G_x0 orbit '2B'. The case
of dealing with axis AXIS_Y can easily be generalized to the general
case of dealing with an axis in the G_x0 orbit '2B'.

There is an 8-dimensional subspace E_8 of the Leech lattice mod 2
fixed by the involution associated with the axis AXIS_Y, and also a
(unique) preimage Q_8 of size 256 of E_8 in Q_x0 associated with that
axis. Q_8 is an Abelian subgroup of Q_x0. The group Q_8 has 255
nonzero elements. 120 of them are of type 2, and 135 are of type 4.
It usually suffices to deal with the image of Q_8 in E_8. The
subspace of the Leech lattice corresponding to E_8 is the space 
spanned by the first eight unit vectors of the Leech lattice.

When reducing AXIS_Y to an axis in the orbit '2A' we first have to
transform one of the type-4 elements of Q_8 to the central involution
x_-1 of G_x0. Afterwards we have to apply a power of the triality
element \tau. Depending on the second axis, some of these type-4
elements lead to a faster reduction than others. Given such a second
axis, function ``small_type4`` returns a list of suitable type 4
vectors in Q_8 for that axis. Once a type-4 vector v has been
selected from that list, the next reduction step is determined.
Given v, we may read from the array ``A_MODE `` how to proceed.
  
For a given second axis, the test function ``test_axis`` checks
that all vectors v in the list returned by function ``small_type4``
have the required properties. This function als shows how to use
the array ``A_MODE ``.

We strongly conjecture that the method discussed in this module
may decrease the maximum number of trialtiy elements required
inside a reduced word from 7 to 6.

[1] M. Seysen. A fast implementation of the Monster group.
    arXiv e-prints, pages arXiv:2203.04223, 2022. 

"""



from collections import defaultdict

import numpy as np

from mmgroup import MM, PLoop
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup.axes import Axis



# Y is the standard 2A involution such that x_{-1} * y is in class 2B
# We have Y = y_o, where o is the standard octad.
Y = MM('y', PLoop(range(8)))
# AXIS_Y is the axis of involution Y
AXIS_Y = Axis('i', Y)





# Will be 256 times 256 hadamard matrix with coefficents +- 1
H256 = None

def hadamard():
    """Return 256 x 256 Hadamard matrix with entries +- 1"""
    global H256
    if H256 is not None:
        return H256
    from scipy.linalg import hadamard
    H256 = hadamard(256, dtype = np.int32) 
    return H256


'''


###############################################################


Comment_BASIS_E8 = '''"""
BASIS_E8 is our selected basis of the subspace E_8 of the Leech
lattice mod 2 in *Leech lattice encoding*
"""

'''




Comment_SH_E9 = '''"""
SH_E9 is a list of 120 triples describing the type-2 vector in
the subspace E_8 of the Leech lattice mod 2, or in the preimage
Q_8 of E_8 in Q_x0.

An entry of that list is a triple (v, sign, n). Here v is
the short vector in standard Leech lattice encoding. The
components of part 98280_x of axis Y corresponding to the vectors 
in that space are equal up to sign. Entry 'sign' of a triple is
the sign of the corresponding component. Entry n is a bit vector
(i.e. an int) containing  the co-ordinates of v with respect to
the basis BASIS_E8.
"""
'''


Comment_TYPE4I = '''"""
TYPE4I is a list of 256 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above.

For an 8-bit vector b, define TYPE_I(b) as the (unique) vector in
E_8 as follows:

The halved scalar product of TYPE_I(b) and E8_BASIS[i] in the
Leech lattice is equal to b[i] (mod 2).

Then TYPE4I[b] is TYPE_I(b) if TYPE_I(b) is of type 4 and 0 otherwise. 

"""
'''


Comment_COEFF = '''
"""
COEFF is a list of 9 coefficients which is (yet to be commented)
"""
'''




###############################################################

source_data_type4 = inspect.getsource(data_type4)


def str_basis_E8():
    b = [hex(x) for x in short_E8_vectors()[0]]
    str_b = ", ".join(b)
    return """BASIS_E8  = [
%s]

""" % str_b



def str_short_E8_vectors():
    s = "SH_E9 = " + pprint.pformat(
         short_E8_vectors()[1], compact = True, width = 66)
    s += '''\n\ndef short_E8_vectors():
    """For compatibility with some automatically generated stuff"""
    return None, SH_E9 

'''
    return s



def str_TYPE4I():
    _, t4 = type2_4_i()
    return  "TYPE4I = " + pprint.pformat(
        t4, compact = True, width = 66) + "\n\n"

def str_COEFF(aa, mask):
    return ("COEFF = " + str(aa) + "\n" + "MASK = "
       + hex(mask) + "\n\n" )



MAIN_FUNC = '''

def small_type4(axis):
    """Return list of suitable type-4 vectors for reducing a pair of axes

    Let AXIS_Y be the standard axis of axis type 2B, and let ``axis``
    be an axis such that the product of the two involutions
    corresponding the axes AXIS_Y and ``axis`` is of class 2B in the
    Monster. 

    The first step to reduce the pair (AXIS_Y, axis) is to map one of
    the type-4 vectors in the table ``TYPE4I`` (which is associated
    with the axis AXIS_Y) to the standard type-4 vector in the Leech
    lattice mode 2.

    Depending on the axis ``axis``, some of the type-4 vectors in table
    ``TYPE4I`` may lead to a faster reduction process than others. The
    function returns a list of suitable type-4 vectors in that table.

    This function uses function ``data_type4`` to compute a partition
    of these type-4 vectors. This partition is invariant under 
    transformations stabilizing both axes, AXIS_Y and ``axis``. 
    In general, the vectors in the smaller sets of the partition lead
    to faster reduction pocesses.  

    The function returns the list of type-4 vectors in table ``TYPE4I``
    contained in one of the smallest sets of the partition. That
    smallest set is selected in a repreducible way.
    """
    vt = data_type4(axis, COEFF, MASK)
    d = defaultdict(int)
    for i, x in enumerate(TYPE4I):
        if x:
            d[vt[i]] += 1
    mu = min(d.values())
    best = min([x for x in d if d[x] == mu])
    v_list = [TYPE4I[i] for i, k in enumerate(vt) 
                   if TYPE4I[i] != 0 and k == best]
    return v_list
'''



def a_mode():
    A_MODE = prep_analyse_axis()
    lst = []
    for key in sorted(A_MODE):
        a, e = A_MODE[key]
        assert (a[3:] == 0).all()
        d = [key] + [x for x in a[:3]] 
        d += [0] * (4 - len(d)) + [0x50000000 + e]
        lst.append(d)
    return np.array(lst, dtype = np.uint32)

def str_a_mode():
    lst = []
    lst.append('''"""
Array A_MODE maps a type-4 element v of the group Q_x0 to an element
g(v) of the Monster to be used for reducing an axis as dicussed in the
header of this module. More precisely, v is given as en element of the
Leech lattice mod 2 in *Leech lattice encoding*, which is sufficent.
The Monster element g(v) is given as a word of generators of length
4, with the last generator a power of the triality, and the other 
generators in G_x0, possibly padded with zeros.

Column A_MODE[:,0] is a sorted array of the type-4 elements v
associated with axis AXIS_Y. In case A_MODE[i,0] = v, the row
A_MODE[i,1:] is the word of generators of the Monster representing 
the element g(v).  
"""''')
    A = a_mode()
    lst.append("A_MODE = np.array([")
    fmt = "[0x%06x," + "0x%08x," * 3 + "0x%08x],"
    for x in A:
        lst.append(fmt % tuple(x))
    lst.append("], dtype = np.uint32)\n\n")    
    return "\n".join(lst)   
     

####### Generate tables for the C programs ############################




Comment_TYPE2BASIS = '''"""

# Table for the C programs 

TYPE2BASIS is a list of 120 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above.

Then TYPE2BASIS is the list of all 8.bit vectors b such that the
vector b * E8_BASIS is of type 2.


The order of the type-2 vectors in E8 is:

   Cocode([i,j]),             0 <= j < i < 8;
   Cocode([i,j]) + Omega,     0 <= j < i < 8;
   Suboctad([0,1,...,7], k),  0 <= k < 64;

where pairs (i,j) and indices k are traversed in lexical order.
"""

'''


def str_TYPE2BASIS():
    t2, t2Omega = [], []
    for i in range(1,8):
        for j in range(i):
            c = Cocode([i,j]).ord
            t2.append(map_leech2_e8(c))
            t2Omega.append(map_leech2_e8(c ^ 0x800000))
    t2 += t2Omega
    oct = Octad(range(8))
    for i in range(64):
        t2.append(map_leech2_e8(SubOctad(oct, i).ord))
    assert len(t2) == 120
    return  "TYPE2BASIS = " + pprint.pformat(
        t2, compact = True, width = 66) + "\n\n"





Comment_TYPE4IBASIS = '''"""

# Table for the C programs 

TYPE4IBASIS is a list of 256 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above.

For an 8-bit vector b, define TYPE_I(b) as the (unique) vector in
E_8 as follows:

The halved scalar product of TYPE_I(b) and E8_BASIS[i] in the
Leech lattice is equal to b[i] (mod 2).

Then TYPE4I[b] is TYPE_I(b) if TYPE_I(b) is of type 4 and 0 otherwise.

TYPE4I_BASIS[i] is a bit vector b if TYPE_I(i) is of type 4 and equal
to  b * E8_BASIS. We put TYPE4I_BASIS[i] = 0 if TYPE_I(i) is not of
type 4.
"""

'''

def str_TYPE4IBASIS():
    t4 = type_4_i_basis()
    return  "TYPE4IBASIS = " + pprint.pformat(
        t4, compact = True, width = 66) + "\n\n"





def str_subspace(i, j, others = False):
    E8_BASIS = short_E8_vectors()[0]
    lst0 = []
    lst1 = []
    coc = Cocode([i,j]).ord
    for k in range(256):
        v = bit_mat_mul(k, E8_BASIS)
        if gen_leech2_type(v) == 4:
            lst = lst0 if gen_leech2_type(v ^ coc) == 2 else lst1
            lst.append(k)
    length = len(lst0)    
    s = f'''"""

# Table for the C programs 

TYPE4_SCAL_{i}_{j} is a list of {length} integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above. Let c be the Golay cocode vector  [{i}, {j}].
For an 8-bit vector b, define E(b) with co-ordinates in that basis
given by b.

Then this list is the list of all bit vectors b such that E(b) is of
type 4, and E(b) + c is of type 2.
"""
'''
    if others:
        lst0 += lst1
        s += f'''"""
Actually, we append the remaining 135 - {length} bit vectors b with
E(b) of type 4 to the list, so that the list has a total length of 135.
"""
'''
    s += f"\nTYPE4_SCAL_{i}_{j} = "    
    s+= pprint.pformat(
        lst0, compact = True, width = 66) + "\n\n"
    return s


def leech2_table_to_hex_list(name, vlist):
    data = []
    for i, x in enumerate(vlist):
        s = "0x%06x" % x + ("," if (i + 1 < len(vlist)) else "")
        s += "\n" if (i % 5 == 4 or i + 1 == len(vlist)) else " "
        data.append(s)
    #print(len(vlist), data)
    return  "%s = [\n" % name + "".join(data) + "]\n"


def table_E8_subspace_expand(name, cond = 0):
    V4LIST = [Cocode(GcVector(x)).ord for x in iter_bitweight(4, 128)]
    if cond == 0:
        COC = XLeech2(0, Cocode([8,9])).ord
        VLIST = [v for v in V4LIST if gen_leech2_type(v ^ COC) == 2]
    else:
        b = [Cocode(GcVector(x << 8)).ord for x in [0xf, 0x33, 0x55]]
        VLIST = lin_table(b)[1:]
    VLIST = [int(x) for x in VLIST] # f.. numpy!
    VLIST.sort()
    VLIST += [0x800000] + [v ^ 0x800000 for v in VLIST]
    return leech2_table_to_hex_list(name, VLIST)

 

Comment_E8_SUBSPACE_8_9 = r'''
"""
Let E_8 be the subspace the Leech lattice mod 2 as above.
Let c be the vector ``x_delta`` in the Leech lattice mod 2 where
``delta`` is the Golay cocode element [8,9]. 
 
Table E8_SUBSPACE_8_9 is the set of all vectors v in E_8 such that
v + x_delta is of type 2.

Table entries are given in Leech lattice encoding.
"""
'''


def table_E8_subset_trio(name):
    def x_octad(o_set):
        gc = GCode(list(o_set))
        return XLeech2(PLoop(gc), 0)
    o0, o8 = x_octad(range(0,8)), x_octad(range(8,16))
    vlist = [0x800000]
    vlist += [Cocode(GcVector(x)).ord for x in [0xf, 0x33, 0x55]]
    vlist += [Cocode([0,1]).ord, o0.ord, Cocode([8,9]).ord, o8.ord]
    def check_type2(v):
        assert gen_leech2_type(v) == 2, (hex(v), gen_leech2_type(v))
    for v in vlist[4:]:
        check_type2(v)
    for i, v in enumerate(vlist[:4]):
        j = (i != 0)
        check_type2(v ^ vlist[4+j]) 
        check_type2(v ^ vlist[6+j]) 
    return leech2_table_to_hex_list(name, vlist)
     

Comment_E8_SUBSET_TRIO = r'''
"""
Let E_8 be the subspace the Leech lattice mod 2 as above. Let E_8' be
the image of the E_8 unter a transformation that exchanges the basis
vector i with the basis vector i + 8 for i < 16. So the subspace of
the Leech lattice corresponding to E_8' is spanned by the basis
vectors of the Leech lattice with indices 8,9,10,11,12,13,14,15.

Table E8_SUBSPACE_TRIO deals with the intersection of the set of the
nonzero vectors in E_8 and E_8'.

The first four vectors e_0,...,e_3 of E8_SUBSPACE_TRIO are a basis
of that intersection; they are equal to

   ``Omega``, [0,1,2,3], [0,1,4,5], [0,2,4,6] .

Here ``Omega`` is the standard co-ordinate frame of the Leech lattice;
and a vector in square brackets corresponds to a Golay cocode word of
weight 4. Thus all vectors e_i are of type 4. For each e_i we need
a type-2 v_i0 vector in E_8 and a type-2 vector v_i8 in E_8' such that
(e_i + v_i0) and (e_i + v_i8) are also of type 2. The remaining four
vectors in the list E8_SUBSPACE_TRIO are vectors satisfying the
conditions for:

    e_00, e_08, e_10, e_18 .

Here the chosen vectors  e_10  and  e_18  also satify the conditions
for the vectors  e_j0  and  e_j8, (with j = 2, 3), respectively. 
 
Entries of the tabl are given in Leech lattice encoding.
"""
'''





####### Auxiliary table for the C programs ##########################



TABLE_AUX_E8_DESCRIPTION = '''"""

Let s(i) be the suboctad Suboctad([0,1,2,3,4,5,6,7], i), for
0 <= i < 64, considered as an element of the Leech lattice (up to
sign), scaled so that it has squared norm 32.

We want to compute s(i) * A * s(i) for a symmetric 24 times 24 matrix A
of integers (modulo 3). By defnition of s(i), it suffices to know the
first eight rows and columns of matrix A. Let v the vector of entries
A[k,l], 0 <= l < k < 8, with pairs (k, l) arranged in lexical order. 
Let matrix A' be the matrix obtained by zeroing the diagonal entries
of A. Then we are almost done if we have computed s(i) * A' * s(i). 

We have s(i) * A' * s(i) = (-1) ** S[i] * v, where S(i) is a vector
with entries 0 or 1. For accelerating that computation with vector
operations, we precompute a matrix AUX_E8[j,i], 0 <= j < 28,
0 <= i < 32; with 

AUX_E8[j, i] = 3 * S[i, j] + 0x30 * S[i + 32, j].
"""

TABLE_AUX_E8 = [
'''

def make_table_aux_e8(verbose = 0):
    o = mat24.vect_to_octad(0xff)
    assert 0 < o < 759
    assert octad_entries(Octad(o)) == [0,1,2,3,4,5,6,7]
    syn_table = np.array([2 * i + (i.bit_count() & 1)
        for i in range(64)], dtype = np.uint8)
    index_table = np.array([(1 << i) + (1 << j)
       for i in range(8) for j in range(i)], dtype = np.uint8)
    signs = [3]
    while len(signs) < 256:
        signs += [x ^ 3 for x in signs]
    signs = np.array(signs, dtype = np.uint8)
    aux_t = np.zeros((28, 64), dtype = np.uint8)
    for i in range(28):
        for j in range(64):
            aux_t[i, j] = signs[index_table[i] & syn_table[j]]
    if verbose:
         print(syn_table)
         print(index_table)
         print(signs)
         print(aux_t[:20,:20])
    aux_table = [0] * 28
    for i in range(28):
        t = aux_t[i, :32] + 16 * aux_t[i, 32:]
        aux_table[i] = [[int(x) for x in t]]
    return aux_table

def str_table_aux_e8():
    s = TABLE_AUX_E8_DESCRIPTION
    aux = make_table_aux_e8()
    data = [str(aux[i]) + ',' for i in range(28)]
    s += "\n".join(data)
    return s + "]\n"



####### Test ########################################################


def test_axis(axis):
    """Test the function ``small_type4``

    Here parameter ``axis`` has the same meaning as in function
    ``small_type4``. This function tests if the list of type-4 vectors
    returned by function ``small_type4`` (when called with argument
    ``axis``) has the required properties.
    """
    vlist = small_type4(axis)
    assert  AXIS_Y.product_class(axis) == "2B", "Bad axis"
    for v in vlist:
        assert v in A_MODE[:,0], hex(v)
        ind = np.searchsorted(A_MODE[:,0], v) 
        assert A_MODE[ind][0] == v, hex(v)
        g_d = A_MODE[ind,1:] 
        t = mm_reduce_op_2A_axis_type(axis.v15.data, g_d, len(g_d), 0x11)
        assert 0x11 < t <= 0x67
        t0 = mm_reduce_op_2A_axis_type(AXIS_Y.v15.data, g_d, len(g_d), 0x11) 
        assert t0 == 0x21



source_test_axis = inspect.getsource(test_axis)



def generate_code(aa, mask):
    with open(GENERATED_PATH, "wt") as f:
        f.write(HEADER)
        f.write(Comment_BASIS_E8)
        f.write(str_basis_E8())
        f.write(Comment_SH_E9)
        f.write(str_short_E8_vectors())
        f.write(source_data_type4 + "\n")
        f.write(Comment_TYPE4I)
        f.write(str_TYPE4I())
        f.write(Comment_COEFF)
        f.write(str_COEFF(aa, mask))
        f.write(str_a_mode())

        f.write(Comment_TYPE2BASIS)
        f.write(str_TYPE2BASIS())
        f.write(Comment_TYPE4IBASIS)
        f.write(str_TYPE4IBASIS())
        f.write(str_subspace(0, 1, others = True))
        f.write(str_subspace(8, 9))
        f.write("#" * 70 + "\n")
        f.write(str_table_aux_e8())
        f.write('\n' + "#" * 70 + "\n")
        f.write(Comment_E8_SUBSPACE_8_9)
        f.write(table_E8_subspace_expand("E8_SUBSPACE_8_9", 0))
        #f.write(table_E8_subspace_expand("E8_SUBSPACE_TRIO", 1))
        f.write(Comment_E8_SUBSET_TRIO)
        f.write(table_E8_subset_trio("E8_SUBSET_TRIO"))
        f.write(MAIN_FUNC) 
        f.write("\n\n")
        f.write(source_test_axis)      
    return GENERATED_FILE 


                
