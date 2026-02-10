import sys
import os
import re
from collections import defaultdict, OrderedDict
from functools import reduce
import operator
import numpy as np

from mmgroup import mat24, GcVector, GCode, AutPL, Cocode, PLoop
from mmgroup import MM, XLeech2, Xsp2_Co1, Octad, SubOctad
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_subtype, gen_leech3_op_vector_word
from mmgroup.generators import gen_leech3_op_vector_atom
from mmgroup.generators import gen_leech3_reduce, gen_leech3to2_short
from mmgroup.generators import gen_leech3_add, gen_leech2to3_abs, gen_leech3_neg
from mmgroup.generators import gen_leech3_reduce_leech_mod3, gen_leech3_neg
from mmgroup.generators import gen_leech2_op_word
from mmgroup.generators import gen_leech2_op_word_leech2_many
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_echelon_eqn
from mmgroup.clifford12 import leech2_matrix_orthogonal
from mmgroup.clifford12 import leech2_matrix_radical
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_subspace_eqn
from mmgroup.clifford12 import bitmatrix64_vmul
from mmgroup.axes import Axis
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))


from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import parse_mat24_orbits
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import py_prep_fixed_leech2_set
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import axis_count_BpmC

try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_prep_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_map_Case6A
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_find_case_2A
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False



#########################################################################
# Tables for code generation
#########################################################################


_BW16_basis = None

def BW16_basis():
    """Basis of Barnes-Wall lattice BW16 mod 2

    The function returns a basis of the 16 dimensional Barnes-Wall
    lattice BW16 modulo 2 with basis vectors given in Leech lattice
    encoding. Here the lattice BW16 is orthogonal to the Standard E8
    sublattice of the  Leech lattice. It is the intersection of the
    Leech lattice with its subspace spanned by its last 16 basis
    vectors.   

    In the Leech lattice, BW16 is orthogonal to E8. In the Leech
    lattice modulo 2, E8 is a subspace of BW16. In our returned basis
    of BW16, the first 8 basis vectors are a basis of E8.
    """
    global _BW16_basis
    if _BW16_basis is None:
        _BW16_basis = [0x800000]
        for i in range(1,7):
            _BW16_basis.append(Cocode(GcVector([0, i])).ord)
        _BW16_basis.append(XLeech2(PLoop(range(8))).ord) 
        for i in range(4):
            _BW16_basis.append(Cocode(GcVector([8, 8 + (1 << i)])).ord)
        W = [[x+8 for x in range(16) if x & m == 0] for m in (1,2,4,8)]   
        _BW16_basis += [XLeech2(PLoop(w)).ord for w in W]
    return _BW16_basis


_BW16_basis_solve = None
ERR_BW16 = "Vector 0x%6x is not in sublattice BW16 of Leech lattice mod 2"


def BW16_basis_coords(v):
    """Return cooordinates of a vector in BW16 in basis vectors 

    Here v must be a vector in the subspace BW16 of the Leech lattice
    modulo 2. The function returns the co-ordinates of v in the basis
    BW16_basis() of BW16 as a bit vector.

    The function raises valueError if v is not in BW16.
    """
    global _BW16_basis_solve
    if _BW16_basis_solve is None:
        basis = BW16_basis()
        m = np.zeros(16, dtype = np.uint64)
        n = 0
        for b in basis:
            n += leech2matrix_add_eqn(m, n, 24, b)
        assert n == 16, n
        _BW16_basis_solve = m
    coord = leech2matrix_subspace_eqn(_BW16_basis_solve, 16, 24, v)
    if v < 0:
        raise ValueError(ERR_BW16 % v)
    return coord      

def do_test_BW16_basis():
    # print([hex(x) for x in BW16_basis()])
    for i, b in enumerate(BW16_basis()):
        assert BW16_basis_coords(b) == 1 << i


_BW16_type2_list = None 

def BW16_type2_list():
    """Return list of shortest vectors i sublattice BW16

    The Barnes-Wall lattice BW16 modulo 2 containes 2160 (shortest)
    vectors of type 2. The function returns a sorted list of these
    vectors in co-ordinates of the basis BW16_basis().
    
    These type-2 vectores fall into 135 subsets of size 16. Due to the
    structure of the basis BW16_basis(), the members of these subsets
    are adjacent i the returend list.
    """
    global _BW16_type2_list
    if _BW16_type2_list is None:
        l2 = []
        for i in range(8,24):
            for j in range(8, i):
                c = Cocode([i,j]).ord
                l2.append(BW16_basis_coords(c))
                l2.append(BW16_basis_coords(c + 0x800000))
        for i in range(759):
           oct = Octad(i)
           if GCode(oct).vector & 0xff == 0:
               for j in range(64):
                   v = XLeech2(SubOctad(oct, j)).ord
                   l2.append(BW16_basis_coords(v))
        l2.sort()
        _BW16_type2_list = [int(x) for x in l2]
    return _BW16_type2_list                


def BW16_type2_basis_list():
    """Yet to be documented!

    """
    ldiff = []
    l2 = BW16_type2_list()
    for i in range(0, 135 * 16, 16):
        st = l2[i]
        t2 = [(x ^ st) & 0xff for x in l2[i+1 : i+16]]
        m = np.zeros(4, dtype = np.uint64)
        n = 0
        for b in t2:
            indep = leech2matrix_add_eqn(m, n, 8, b)
            if indep:
                ldiff.append(b)
                n += 1
                if n == 4:
                    break
        if n < 4:
            ERR = "Basis of subspace of E8 not found"
            raise ValueError(ERR)
    return ldiff





class Tables:
    directives = {}
    def __init__(self):
        self.tables = {
            "MM_AXIS3_CASE_4C_BASIS_E8":
                BW16_basis(),
            "MM_AXIS3_CASE_4C_SHORT_BW16":
                BW16_type2_list()[0:135*16:16],
            "MM_AXIS3_CASE_4C_SHORT_SUB_E8":
                BW16_type2_basis_list()
        }

class MockupTables:
    directives = {}
    tables = defaultdict(lambda x: [0])

def do_test_BW16(verbose = 0):
    if verbose:
        print("Basis of BW16")
        print([hex(x) for x in BW16_basis()])
    for i, b in enumerate(BW16_basis()):
        assert BW16_basis_coords(b) == 1 << i
    t2 = BW16_type2_list()
    if verbose > 1:
        print("Type-2 vectors in BW16, %d entries" % len(t2))
    assert len(t2) == 135*16
    for i in range(0, 135*16, 16):
        t2i = t2[i:i+16]
        tdiff = [t2i[i] ^ t2i[0] for i in range(16)]
        assert reduce(operator.or_, tdiff) & -0x100 == 0 
        s = " ".join(["%04x" % v for v in  t2i])
        if verbose > 1:
            print(s)
    t = Tables().tables["MM_AXIS3_CASE_4C_SHORT_SUB_E8"]    
    if verbose > 1:
        print("Bases of corresponding E8 sbspaces")
        for i in range(0, 135*4, 5*4):
            print(" ".join(["%02x" % v for v in t[i : i+5*4]]))
       


#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################

def axis_to_MMOpFastMatrix(axis):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0)
    return amod3



#########################################################################
# Check case 4C
#########################################################################


def iter_testcases_4C(n_cases = 100):
    std_axis_4C = Axis.representatives()["4C"]
    for i in range(n_cases):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis_4C * g



def check_case_4C(n_tests = 100, verbose = 0):
    if use_mmgroup_fast:
        print("Testing axis type 4C")
        for n, axis in enumerate(iter_testcases_4C(n_tests)):
            if verbose:
                print("Test", n+1)
            if verbose > 1:
                amod3a = axis_to_MMOpFastMatrix(axis)
                buf = amod3a.analyze_v4()
                print(amod3a.data[1])
                print([hex(x) for x in buf])
            amod3 = axis_to_MMOpFastMatrix(axis)
            ax_type, v = amod3.find_v4()
            assert ax_type == "4C", (ax_type, hex(v))
            amod31 = axis_to_MMOpFastMatrix(axis)
            ax_type1, v1 = amod31.find_v4()
            assert ax_type == ax_type1
            assert v == v1
            ax1 = axis * Xsp2_Co1('c', v) **-1
            t_markers = axis_count_BpmC(ax1)
            if verbose:
                print(t_markers)           
            assert set(t_markers) == {0, 136}
            e = t_markers.index(136) + 1
            assert ax1.axis_type(e) == '2B', ax1.axis_type(e)
        print("passed")      

  
#########################################################################
# Main program
#########################################################################


def test_4C_all(n_tests=100):
    do_test_BW16(verbose = 0)
    check_case_4C(n_tests = 100, verbose = 0)

if __name__ == "__main__":
    print("start")
    test_4C_all()

