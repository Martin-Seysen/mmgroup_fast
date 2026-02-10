# This file has been generate automatically; do not change!

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


"""
BASIS_E8 is our selected basis of the subspace E_8 of the Leech
lattice mod 2 in *Leech lattice encoding*
"""

BASIS_E8  = [
0x600, 0x500, 0x700, 0x40f, 0x8f, 0x4f, 0x800000, 0x80f00f]

"""
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
SH_E9 = [(1536, 1, 1), (1280, 1, 2), (768, 1, 3), (1792, 1, 4),
 (256, 1, 5), (512, 1, 6), (1039, 1, 8), (527, 1, 9),
 (271, 1, 10), (783, 1, 12), (143, 1, 16), (1679, 1, 17),
 (1423, 1, 18), (1935, 1, 20), (1152, 1, 24), (128, 1, 31),
 (79, 1, 32), (1615, 1, 33), (1359, 1, 34), (1871, 1, 36),
 (1088, 1, 40), (64, 1, 47), (192, 1, 48), (1216, 1, 55),
 (1999, 1, 59), (1487, 1, 61), (1743, 1, 62), (207, 1, 63),
 (8390144, -1, 65), (8389888, -1, 66), (8389376, -1, 67),
 (8390400, -1, 68), (8388864, -1, 69), (8389120, -1, 70),
 (8389647, -1, 72), (8389135, -1, 73), (8388879, -1, 74),
 (8389391, -1, 76), (8388751, -1, 80), (8390287, -1, 81),
 (8390031, -1, 82), (8390543, -1, 84), (8389760, -1, 88),
 (8388736, -1, 95), (8388687, -1, 96), (8390223, -1, 97),
 (8389967, -1, 98), (8390479, -1, 100), (8389696, -1, 104),
 (8388672, -1, 111), (8388800, -1, 112), (8389824, -1, 119),
 (8390607, -1, 123), (8390095, -1, 125), (8390351, -1, 126),
 (8388815, -1, 127), (8450063, -1, 128), (8451087, -1, 135),
 (8451840, -1, 139), (8451328, -1, 141), (8451584, -1, 142),
 (8450048, -1, 143), (8450944, -1, 147), (8450432, -1, 149),
 (8450688, -1, 150), (8451200, -1, 151), (8450703, -1, 153),
 (8450447, -1, 154), (8451983, -1, 155), (8450959, -1, 156),
 (8451471, -1, 157), (8451727, -1, 158), (8450880, -1, 163),
 (8450368, -1, 165), (8450624, -1, 166), (8451136, -1, 167),
 (8450639, -1, 169), (8450383, -1, 170), (8451919, -1, 171),
 (8450895, -1, 172), (8451407, -1, 173), (8451663, -1, 174),
 (8451791, -1, 177), (8451535, -1, 178), (8451023, -1, 179),
 (8452047, -1, 180), (8450511, -1, 181), (8450767, -1, 182),
 (8451264, -1, 184), (8450752, -1, 185), (8450496, -1, 186),
 (8451008, -1, 188), (62991, 1, 193), (62735, 1, 194),
 (62223, 1, 195), (63247, 1, 196), (61711, 1, 197),
 (61967, 1, 198), (62464, 1, 200), (61952, 1, 201),
 (61696, 1, 202), (62208, 1, 204), (61568, 1, 208),
 (63104, 1, 209), (62848, 1, 210), (63360, 1, 212),
 (62607, 1, 216), (61583, 1, 223), (61504, 1, 224),
 (63040, 1, 225), (62784, 1, 226), (63296, 1, 228),
 (62543, 1, 232), (61519, 1, 239), (61647, 1, 240),
 (62671, 1, 247), (63424, 1, 251), (62912, 1, 253),
 (63168, 1, 254), (61632, 1, 255)]

def short_E8_vectors():
    """For compatibility with some automatically generated stuff"""
    return None, SH_E9 

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

"""
TYPE4I is a list of 256 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above.

For an 8-bit vector b, define TYPE_I(b) as the (unique) vector in
E_8 as follows:

The halved scalar product of TYPE_I(b) and E8_BASIS[i] in the
Leech lattice is equal to b[i] (mod 2).

Then TYPE4I[b] is TYPE_I(b) if TYPE_I(b) is of type 4 and 0 otherwise. 

"""
TYPE4I = [0, 0, 0, 0, 0, 0, 0, 1231, 0, 0, 0, 1984, 0, 1472, 1728, 15, 0,
 0, 0, 832, 0, 320, 576, 1167, 0, 591, 335, 1920, 847, 1408, 1664,
 0, 0, 0, 0, 896, 0, 384, 640, 1103, 0, 655, 399, 1856, 911, 1344,
 1600, 0, 0, 1551, 1295, 960, 1807, 448, 704, 0, 1024, 719, 463,
 0, 975, 0, 0, 0, 0, 8451776, 8451520, 8450831, 8452032, 8450319,
 8450575, 0, 8451279, 8450560, 8450304, 0, 8450816, 0, 0, 0,
 8450127, 8451712, 8451456, 0, 8451968, 0, 0, 0, 8451215, 0, 0, 0,
 0, 0, 0, 8450112, 8450191, 8451648, 8451392, 0, 8451904, 0, 0, 0,
 8451151, 0, 0, 0, 0, 0, 0, 8450176, 8450255, 0, 0, 0, 0, 0, 0,
 8451072, 0, 0, 0, 8451855, 0, 8451343, 8451599, 8450240, 8388608,
 0, 0, 0, 0, 0, 0, 8389839, 0, 0, 0, 8390592, 0, 8390080, 8390336,
 8388623, 0, 0, 0, 8389440, 0, 8388928, 8389184, 8389775, 0,
 8389199, 8388943, 8390528, 8389455, 8390016, 8390272, 0, 0, 0, 0,
 8389504, 0, 8388992, 8389248, 8389711, 0, 8389263, 8389007,
 8390464, 8389519, 8389952, 8390208, 0, 0, 8390159, 8389903,
 8389568, 8390415, 8389056, 8389312, 0, 8389632, 8389327, 8389071,
 0, 8389583, 0, 0, 0, 61455, 0, 0, 0, 0, 0, 0, 62656, 0, 0, 0,
 63439, 0, 62927, 63183, 61440, 0, 0, 0, 62287, 0, 61775, 62031,
 62592, 0, 62016, 61760, 63375, 62272, 62863, 63119, 0, 0, 0, 0,
 62351, 0, 61839, 62095, 62528, 0, 62080, 61824, 63311, 62336,
 62799, 63055, 0, 0, 62976, 62720, 62415, 63232, 61903, 62159, 0,
 62479, 62144, 61888, 0, 62400, 0, 0, 0]


"""
COEFF is a list of 9 coefficients which is (yet to be commented)
"""
COEFF = [93, 91, 39, 46, 10, 53, 41, 45, 32]
MASK = 0x7f

"""
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
"""
A_MODE = np.array([
[0x00000f,0xa0b04780,0xe0000001,0x00000000,0x50000001],
[0x000140,0xa0076d40,0xe0000001,0x00000000,0x50000001],
[0x00014f,0xa0005a00,0xe0000001,0x00000000,0x50000001],
[0x000180,0xa0076980,0xe0000001,0x00000000,0x50000001],
[0x00018f,0xa0005640,0xe0000001,0x00000000,0x50000001],
[0x0001c0,0xa0077100,0xe0000001,0x00000000,0x50000001],
[0x0001cf,0xa0005dc0,0xe0000001,0x00000000,0x50000001],
[0x000240,0xa3338580,0xe0000001,0x00000000,0x50000001],
[0x00024f,0xa0222900,0xe0000001,0x00000000,0x50000001],
[0x000280,0xa3cf2d00,0xe0000001,0x00000000,0x50000001],
[0x00028f,0xa01b6480,0xe0000001,0x00000000,0x50000001],
[0x0002c0,0xa32cc100,0xe0000001,0x00000000,0x50000001],
[0x0002cf,0xa028ed80,0xe0000001,0x00000000,0x50000001],
[0x000340,0xa00e31c0,0xe0000001,0x00000000,0x50000001],
[0x00034f,0xa0000b40,0xe0000001,0x00000000,0x50000001],
[0x000380,0xa00e2e00,0xe0000001,0x00000000,0x50000001],
[0x00038f,0xa0000780,0xe0000001,0x00000000,0x50000001],
[0x0003c0,0xa00e3580,0xe0000001,0x00000000,0x50000001],
[0x0003cf,0xa0000f00,0xe0000001,0x00000000,0x50000001],
[0x000400,0xe0000001,0x00000000,0x00000000,0x50000001],
[0x00044f,0xa0bdd080,0xe0000001,0x00000000,0x50000001],
[0x00048f,0xa0b70c00,0xe0000001,0x00000000,0x50000001],
[0x0004cf,0xa0c49500,0xe0000001,0x00000000,0x50000001],
[0x00050f,0xa0005280,0xe0000001,0x00000000,0x50000001],
[0x000540,0xa007bfc0,0xe0000001,0x00000000,0x50000001],
[0x000580,0xa0080e80,0xe0000001,0x00000000,0x50000001],
[0x0005c0,0xa007bc00,0xe0000001,0x00000000,0x50000001],
[0x00060f,0xa014a000,0xe0000001,0x00000000,0x50000001],
[0x000640,0xa2911980,0xe0000001,0x00000000,0x50000001],
[0x000680,0xa28a5500,0xe0000001,0x00000000,0x50000001],
[0x0006c0,0xa297de00,0xe0000001,0x00000000,0x50000001],
[0x00070f,0xa00003c0,0xe0000001,0x00000000,0x50000001],
[0x000740,0xa00e8440,0xe0000001,0x00000000,0x50000001],
[0x000780,0xa00ed300,0xe0000001,0x00000000,0x50000001],
[0x0007c0,0xa00e8080,0xe0000001,0x00000000,0x50000001],
[0x00f000,0xe0000002,0xa0b04780,0xe0000001,0x50000001],
[0x00f00f,0xe0000001,0xa0b04780,0xe0000001,0x50000001],
[0x00f140,0xe0000002,0xa0005a00,0xe0000001,0x50000001],
[0x00f14f,0xe0000001,0xa0005a00,0xe0000001,0x50000001],
[0x00f180,0xe0000002,0xa0005640,0xe0000001,0x50000001],
[0x00f18f,0xe0000001,0xa0005640,0xe0000001,0x50000001],
[0x00f1c0,0xe0000002,0xa0005dc0,0xe0000001,0x50000001],
[0x00f1cf,0xe0000001,0xa0005dc0,0xe0000001,0x50000001],
[0x00f240,0xe0000002,0xa0222900,0xe0000001,0x50000001],
[0x00f24f,0xe0000001,0xa0222900,0xe0000001,0x50000001],
[0x00f280,0xe0000002,0xa01b6480,0xe0000001,0x50000001],
[0x00f28f,0xe0000001,0xa01b6480,0xe0000001,0x50000001],
[0x00f2c0,0xe0000002,0xa028ed80,0xe0000001,0x50000001],
[0x00f2cf,0xe0000001,0xa028ed80,0xe0000001,0x50000001],
[0x00f340,0xe0000002,0xa0000b40,0xe0000001,0x50000001],
[0x00f34f,0xe0000001,0xa0000b40,0xe0000001,0x50000001],
[0x00f380,0xe0000002,0xa0000780,0xe0000001,0x50000001],
[0x00f38f,0xe0000001,0xa0000780,0xe0000001,0x50000001],
[0x00f3c0,0xe0000002,0xa0000f00,0xe0000001,0x50000001],
[0x00f3cf,0xe0000001,0xa0000f00,0xe0000001,0x50000001],
[0x00f40f,0xe0000001,0xa0b04780,0xe0000002,0x50000002],
[0x00f440,0xe0000002,0xa0bdd080,0xe0000002,0x50000001],
[0x00f480,0xe0000002,0xa0b70c00,0xe0000002,0x50000001],
[0x00f4c0,0xe0000002,0xa0c49500,0xe0000002,0x50000001],
[0x00f500,0xe0000002,0xa0005280,0xe0000002,0x50000001],
[0x00f54f,0xe0000001,0xa0005a00,0xe0000002,0x50000002],
[0x00f58f,0xe0000001,0xa0005640,0xe0000002,0x50000002],
[0x00f5cf,0xe0000001,0xa0005dc0,0xe0000002,0x50000002],
[0x00f600,0xe0000002,0xa014a000,0xe0000002,0x50000001],
[0x00f64f,0xe0000001,0xa0222900,0xe0000002,0x50000002],
[0x00f68f,0xe0000001,0xa01b6480,0xe0000002,0x50000002],
[0x00f6cf,0xe0000001,0xa028ed80,0xe0000002,0x50000002],
[0x00f700,0xe0000002,0xa00003c0,0xe0000002,0x50000001],
[0x00f74f,0xe0000001,0xa0000b40,0xe0000002,0x50000002],
[0x00f78f,0xe0000001,0xa0000780,0xe0000002,0x50000002],
[0x00f7cf,0xe0000001,0xa0000f00,0xe0000002,0x50000002],
[0x800000,0x00000000,0x00000000,0x00000000,0x50000002],
[0x80000f,0xa0b04780,0xe0000002,0x00000000,0x50000001],
[0x800140,0xa0076d40,0xe0000002,0x00000000,0x50000001],
[0x80014f,0xa0005a00,0xe0000002,0x00000000,0x50000001],
[0x800180,0xa0076980,0xe0000002,0x00000000,0x50000001],
[0x80018f,0xa0005640,0xe0000002,0x00000000,0x50000001],
[0x8001c0,0xa0077100,0xe0000002,0x00000000,0x50000001],
[0x8001cf,0xa0005dc0,0xe0000002,0x00000000,0x50000001],
[0x800240,0xa3338580,0xe0000002,0x00000000,0x50000001],
[0x80024f,0xa0222900,0xe0000002,0x00000000,0x50000001],
[0x800280,0xa3cf2d00,0xe0000002,0x00000000,0x50000001],
[0x80028f,0xa01b6480,0xe0000002,0x00000000,0x50000001],
[0x8002c0,0xa32cc100,0xe0000002,0x00000000,0x50000001],
[0x8002cf,0xa028ed80,0xe0000002,0x00000000,0x50000001],
[0x800340,0xa00e31c0,0xe0000002,0x00000000,0x50000001],
[0x80034f,0xa0000b40,0xe0000002,0x00000000,0x50000001],
[0x800380,0xa00e2e00,0xe0000002,0x00000000,0x50000001],
[0x80038f,0xa0000780,0xe0000002,0x00000000,0x50000001],
[0x8003c0,0xa00e3580,0xe0000002,0x00000000,0x50000001],
[0x8003cf,0xa0000f00,0xe0000002,0x00000000,0x50000001],
[0x800400,0xe0000002,0x00000000,0x00000000,0x50000001],
[0x80044f,0xa0bdd080,0xe0000002,0x00000000,0x50000001],
[0x80048f,0xa0b70c00,0xe0000002,0x00000000,0x50000001],
[0x8004cf,0xa0c49500,0xe0000002,0x00000000,0x50000001],
[0x80050f,0xa0005280,0xe0000002,0x00000000,0x50000001],
[0x800540,0xa007bfc0,0xe0000002,0x00000000,0x50000001],
[0x800580,0xa0080e80,0xe0000002,0x00000000,0x50000001],
[0x8005c0,0xa007bc00,0xe0000002,0x00000000,0x50000001],
[0x80060f,0xa014a000,0xe0000002,0x00000000,0x50000001],
[0x800640,0xa2911980,0xe0000002,0x00000000,0x50000001],
[0x800680,0xa28a5500,0xe0000002,0x00000000,0x50000001],
[0x8006c0,0xa297de00,0xe0000002,0x00000000,0x50000001],
[0x80070f,0xa00003c0,0xe0000002,0x00000000,0x50000001],
[0x800740,0xa00e8440,0xe0000002,0x00000000,0x50000001],
[0x800780,0xa00ed300,0xe0000002,0x00000000,0x50000001],
[0x8007c0,0xa00e8080,0xe0000002,0x00000000,0x50000001],
[0x80f040,0xe0000002,0xa0bdd080,0xe0000001,0x50000002],
[0x80f04f,0xe0000001,0xa0bdd080,0xe0000002,0x50000002],
[0x80f080,0xe0000002,0xa0b70c00,0xe0000001,0x50000002],
[0x80f08f,0xe0000001,0xa0b70c00,0xe0000002,0x50000002],
[0x80f0c0,0xe0000002,0xa0c49500,0xe0000001,0x50000002],
[0x80f0cf,0xe0000001,0xa0c49500,0xe0000002,0x50000002],
[0x80f100,0xe0000002,0xa0005280,0xe0000001,0x50000002],
[0x80f10f,0xe0000001,0xa0005280,0xe0000002,0x50000002],
[0x80f200,0xe0000002,0xa014a000,0xe0000001,0x50000002],
[0x80f20f,0xe0000001,0xa014a000,0xe0000002,0x50000002],
[0x80f300,0xe0000002,0xa00003c0,0xe0000001,0x50000002],
[0x80f30f,0xe0000001,0xa00003c0,0xe0000002,0x50000002],
[0x80f400,0xe0000002,0xa0b04780,0xe0000002,0x50000002],
[0x80f44f,0xe0000001,0xa0bdd080,0xe0000001,0x50000001],
[0x80f48f,0xe0000001,0xa0b70c00,0xe0000001,0x50000001],
[0x80f4cf,0xe0000001,0xa0c49500,0xe0000001,0x50000001],
[0x80f50f,0xe0000001,0xa0005280,0xe0000001,0x50000001],
[0x80f540,0xe0000002,0xa0005a00,0xe0000002,0x50000002],
[0x80f580,0xe0000002,0xa0005640,0xe0000002,0x50000002],
[0x80f5c0,0xe0000002,0xa0005dc0,0xe0000002,0x50000002],
[0x80f60f,0xe0000001,0xa014a000,0xe0000001,0x50000001],
[0x80f640,0xe0000002,0xa0222900,0xe0000002,0x50000002],
[0x80f680,0xe0000002,0xa01b6480,0xe0000002,0x50000002],
[0x80f6c0,0xe0000002,0xa028ed80,0xe0000002,0x50000002],
[0x80f70f,0xe0000001,0xa00003c0,0xe0000001,0x50000001],
[0x80f740,0xe0000002,0xa0000b40,0xe0000002,0x50000002],
[0x80f780,0xe0000002,0xa0000780,0xe0000002,0x50000002],
[0x80f7c0,0xe0000002,0xa0000f00,0xe0000002,0x50000002],
], dtype = np.uint32)

"""

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

TYPE2BASIS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34,
 36, 40, 48, 63, 62, 61, 59, 55, 47, 31, 65, 66, 67, 68, 69, 70,
 72, 73, 74, 76, 80, 81, 82, 84, 88, 96, 97, 98, 100, 104, 112,
 127, 126, 125, 123, 119, 111, 95, 128, 193, 194, 195, 196, 197,
 198, 135, 200, 201, 202, 139, 204, 141, 142, 143, 208, 209, 210,
 147, 212, 149, 150, 151, 216, 153, 154, 155, 156, 157, 158, 223,
 224, 225, 226, 163, 228, 165, 166, 167, 232, 169, 170, 171, 172,
 173, 174, 239, 240, 177, 178, 179, 180, 181, 182, 247, 184, 185,
 186, 251, 188, 253, 254, 255]

"""

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

TYPE4IBASIS = [0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 52, 0, 50, 49, 15, 0, 0, 0, 44,
 0, 42, 41, 23, 0, 38, 37, 27, 35, 29, 30, 0, 0, 0, 0, 28, 0, 26,
 25, 39, 0, 22, 21, 43, 19, 45, 46, 0, 0, 14, 13, 51, 11, 53, 54,
 0, 7, 57, 58, 0, 60, 0, 0, 0, 0, 190, 189, 131, 187, 133, 134, 0,
 183, 137, 138, 0, 140, 0, 0, 0, 175, 145, 146, 0, 148, 0, 0, 0,
 152, 0, 0, 0, 0, 0, 0, 160, 159, 161, 162, 0, 164, 0, 0, 0, 168,
 0, 0, 0, 0, 0, 0, 144, 176, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0, 132,
 0, 130, 129, 191, 64, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 116, 0,
 114, 113, 79, 0, 0, 0, 108, 0, 106, 105, 87, 0, 102, 101, 91, 99,
 93, 94, 0, 0, 0, 0, 92, 0, 90, 89, 103, 0, 86, 85, 107, 83, 109,
 110, 0, 0, 78, 77, 115, 75, 117, 118, 0, 71, 121, 122, 0, 124, 0,
 0, 0, 192, 0, 0, 0, 0, 0, 0, 248, 0, 0, 0, 244, 0, 242, 241, 207,
 0, 0, 0, 236, 0, 234, 233, 215, 0, 230, 229, 219, 227, 221, 222,
 0, 0, 0, 0, 220, 0, 218, 217, 231, 0, 214, 213, 235, 211, 237,
 238, 0, 0, 206, 205, 243, 203, 245, 246, 0, 199, 249, 250, 0,
 252, 0, 0, 0]

"""

# Table for the C programs 

TYPE4_SCAL_0_1 is a list of 63 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above. Let c be the Golay cocode vector  [0, 1].
For an 8-bit vector b, define E(b) with co-ordinates in that basis
given by b.

Then this list is the list of all bit vectors b such that E(b) is of
type 4, and E(b) + c is of type 2.
"""
"""
Actually, we append the remaining 135 - 63 bit vectors b with
E(b) of type 4 to the list, so that the list has a total length of 135.
"""

TYPE4_SCAL_0_1 = [7, 11, 13, 19, 21, 25, 30, 35, 37, 41, 46, 49, 54, 58, 60, 64,
 71, 75, 77, 83, 85, 89, 94, 99, 101, 105, 110, 113, 118, 122,
 124, 129, 134, 138, 140, 146, 148, 152, 159, 162, 164, 168, 175,
 176, 183, 187, 189, 192, 199, 203, 205, 211, 213, 217, 222, 227,
 229, 233, 238, 241, 246, 250, 252, 14, 15, 22, 23, 26, 27, 28,
 29, 38, 39, 42, 43, 44, 45, 50, 51, 52, 53, 56, 57, 78, 79, 86,
 87, 90, 91, 92, 93, 102, 103, 106, 107, 108, 109, 114, 115, 116,
 117, 120, 121, 130, 131, 132, 133, 136, 137, 144, 145, 160, 161,
 190, 191, 206, 207, 214, 215, 218, 219, 220, 221, 230, 231, 234,
 235, 236, 237, 242, 243, 244, 245, 248, 249]

"""

# Table for the C programs 

TYPE4_SCAL_8_9 is a list of 15 integers defined as follows.

Let E8_BASIS be the basis of the subspace E_8 of the Leech lattice
mod 2 as above. Let c be the Golay cocode vector  [8, 9].
For an 8-bit vector b, define E(b) with co-ordinates in that basis
given by b.

Then this list is the list of all bit vectors b such that E(b) is of
type 4, and E(b) + c is of type 2.
"""

TYPE4_SCAL_8_9 = [7, 25, 30, 43, 44, 50, 53, 64, 71, 89, 94, 107, 108, 114, 117]

######################################################################
"""

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
[[3, 3, 48, 48, 48, 48, 3, 3, 48, 48, 3, 3, 3, 3, 48, 48, 48, 48, 3, 3, 3, 3, 48, 48, 3, 3, 48, 48, 48, 48, 3, 3]],
[[3, 48, 3, 48, 48, 3, 48, 3, 48, 3, 48, 3, 3, 48, 3, 48, 48, 3, 48, 3, 3, 48, 3, 48, 3, 48, 3, 48, 48, 3, 48, 3]],
[[51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51]],
[[3, 48, 48, 3, 3, 48, 48, 3, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 3, 48, 48, 3, 3, 48, 48, 3]],
[[51, 0, 51, 0, 0, 51, 0, 51, 51, 0, 51, 0, 0, 51, 0, 51, 51, 0, 51, 0, 0, 51, 0, 51, 51, 0, 51, 0, 0, 51, 0, 51]],
[[51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51]],
[[3, 48, 48, 3, 48, 3, 3, 48, 3, 48, 48, 3, 48, 3, 3, 48, 48, 3, 3, 48, 3, 48, 48, 3, 48, 3, 3, 48, 3, 48, 48, 3]],
[[51, 0, 51, 0, 51, 0, 51, 0, 0, 51, 0, 51, 0, 51, 0, 51, 51, 0, 51, 0, 51, 0, 51, 0, 0, 51, 0, 51, 0, 51, 0, 51]],
[[51, 51, 0, 0, 51, 51, 0, 0, 0, 0, 51, 51, 0, 0, 51, 51, 51, 51, 0, 0, 51, 51, 0, 0, 0, 0, 51, 51, 0, 0, 51, 51]],
[[51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 51, 51, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 51, 51]],
[[3, 48, 48, 3, 48, 3, 3, 48, 48, 3, 3, 48, 3, 48, 48, 3, 3, 48, 48, 3, 48, 3, 3, 48, 48, 3, 3, 48, 3, 48, 48, 3]],
[[51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51]],
[[51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51]],
[[51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51]],
[[51, 51, 51, 51, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 51, 51, 51, 51, 51, 51]],
[[51, 0, 0, 51, 0, 51, 51, 0, 0, 51, 51, 0, 51, 0, 0, 51, 0, 51, 51, 0, 51, 0, 0, 51, 51, 0, 0, 51, 0, 51, 51, 0]],
[[3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48]],
[[3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48, 3, 3, 48, 48]],
[[3, 3, 3, 3, 48, 48, 48, 48, 3, 3, 3, 3, 48, 48, 48, 48, 3, 3, 3, 3, 48, 48, 48, 48, 3, 3, 3, 3, 48, 48, 48, 48]],
[[3, 3, 3, 3, 3, 3, 3, 3, 48, 48, 48, 48, 48, 48, 48, 48, 3, 3, 3, 3, 3, 3, 3, 3, 48, 48, 48, 48, 48, 48, 48, 48]],
[[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]],
[[3, 48, 48, 3, 48, 3, 3, 48, 48, 3, 3, 48, 3, 48, 48, 3, 48, 3, 3, 48, 3, 48, 48, 3, 3, 48, 48, 3, 48, 3, 3, 48]],
[[51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0, 51, 0]],
[[51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0, 51, 51, 0, 0]],
[[51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0, 51, 51, 51, 51, 0, 0, 0, 0]],
[[51, 51, 51, 51, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 51, 51, 51, 51, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0]],
[[51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
[[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],]

######################################################################

"""
Let E_8 be the subspace the Leech lattice mod 2 as above.
Let c be the vector ``x_delta`` in the Leech lattice mod 2 where
``delta`` is the Golay cocode element [8,9]. 
 
Table E8_SUBSPACE_8_9 is the set of all vectors v in E_8 such that
v + x_delta is of type 2.

Table entries are given in Leech lattice encoding.
"""
E8_SUBSPACE_8_9 = [
0x0001c0, 0x000280, 0x000340, 0x000400, 0x0005c0,
0x000680, 0x000740, 0x800000, 0x8001c0, 0x800280,
0x800340, 0x800400, 0x8005c0, 0x800680, 0x800740
]

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
E8_SUBSET_TRIO = [
0x800000, 0x000400, 0x000280, 0x000140, 0x000600,
0x80f00f, 0x000420, 0x003403
]


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
